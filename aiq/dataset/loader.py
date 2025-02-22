import abc
import os
from typing import List

import pandas as pd
import mysql.connector


class DataLoader(abc.ABC):
    """
    DataLoader is designed for loading raw dataset from original dataset source.
    """

    @staticmethod
    def load_instruments(data_dir, market, start_time=None, end_time=None):
        """
        Args:
            data_dir (str): dataset directory
            market (str): market name
            start_time (str): start time
            end_time (str): end_time

        Returns:
            List[Tuple[str]]: list of instrument's name and list date
        """
        instruments = set()
        if data_dir is not None:
            file_path = os.path.join(data_dir, "instruments", market + ".csv")
            df = pd.read_csv(file_path)
            if start_time is not None:
                df = df[df["Date"] >= start_time]
            if end_time is not None:
                df = df[df["Date"] <= end_time]
        else:
            connection = mysql.connector.connect(
                host="127.0.0.1",
                user="zcs",
                passwd="2025zcsdaydayup",
                database="stock_info",
            )
            with connection.cursor() as cursor:
                query = (
                    "SELECT DISTINCT ts_code, trade_date "
                    "FROM ts_idx_index_weight "
                    "WHERE index_code=%s AND trade_date >= %s AND trade_date <= %s"
                )
                cursor.execute(
                    query,
                    (market, start_time.replace("-", ""), end_time.replace("-", "")),
                )

                # Fetch all rows and create a DataFrame
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=["Instrument", "Date"])

                # Convert 'Date' column to datetime format
                df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

                # Format 'Date' column to 'YYYY-MM-DD'
                df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        for _, row in df.iterrows():
            instruments.add(row["Instrument"])

        return list(instruments)

    @staticmethod
    def load_features(
        data_dir,
        instrument,
        timestamp_col="Date",
        start_time=None,
        end_time=None,
        column_names=None,
    ) -> pd.DataFrame:
        """
        Args:
            data_dir (str): dataset directory
            instrument (str):  instrument's name
            timestamp_col (str): column name of timestamp
            start_time (str): start of the time range.
            end_time (str): end of the time range.
            column_names (List[str]): output column names

        Returns:
            pd.DataFrame: dataset load from the files
        """
        file_path = os.path.join(data_dir, "features", instrument + ".csv")
        if not os.path.exists(file_path):
            return None
        df = pd.read_csv(file_path)
        if start_time is not None:
            df = df[(df[timestamp_col] >= start_time)]
        if end_time is not None:
            df = df[(df[timestamp_col] <= end_time)]
        if column_names is not None:
            df = df[column_names]
        df = df.sort_values(by=timestamp_col, ascending=True)
        return df

    @staticmethod
    def load_calendars(
        data_dir, timestamp_col="Trade_date", start_time=None, end_time=None
    ) -> List[str]:
        """

        Args:
            data_dir (str): dataset directory
            timestamp_col (str): column name of timestamp
            start_time (str): start of the time range.
            end_time (str): end of the time range.

        Returns:

        """
        file_path = os.path.join(data_dir, "calendars", "calendar.csv")
        if not os.path.exists(file_path):
            return None
        df = pd.read_csv(file_path)
        if start_time is not None:
            df = df[(df[timestamp_col] >= start_time)]
        if end_time is not None:
            df = df[(df[timestamp_col] <= end_time)]

        trade_dates = df[df["Exchange"] == "SSE"]["Trade_date"].tolist()
        return trade_dates
