import os
from typing import List, Dict

import pandas as pd

from aiq.utils.db import init_db_connection


class DataLoader:
    """
    DataLoader is designed for loading raw dataset from original dataset source.
    """

    @staticmethod
    def load_market_instruments(
        data_dir, market, start_time=None, end_time=None
    ) -> List[str]:
        """
        Load constituent instruments of a specific market/index.

        Args:
            data_dir (str): dataset directory
            market (str): market name
            start_time (str): start time
            end_time (str): end_time

        Returns:
            List[str]: list of instrument's name
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
            db_connection = init_db_connection()
            try:
                with db_connection.cursor() as cursor:
                    query = (
                        "SELECT DISTINCT ts_code, trade_date "
                        "FROM ts_idx_index_weight "
                        "WHERE index_code=%s AND trade_date>=%s AND trade_date<=%s"
                    )
                    cursor.execute(
                        query,
                        (
                            market,
                            start_time.replace("-", ""),
                            end_time.replace("-", ""),
                        ),
                    )

                    # Fetch all rows and create a DataFrame
                    data = cursor.fetchall()
                    df = pd.DataFrame(data, columns=["Instrument", "Date"])

                    # Convert 'Date' column to datetime format
                    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

                    # Format 'Date' column to 'YYYY-MM-DD'
                    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            finally:
                db_connection.close()

        instruments = df["Instrument"].unique().tolist()
        return list(instruments)

    @staticmethod
    def load_instrument_features(
        data_dir,
        instrument,
        timestamp_col="Date",
        start_time=None,
        end_time=None,
        column_names=None,
    ) -> pd.DataFrame:
        """
        Load feature data for a single instrument.

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
        if data_dir is not None:
            file_path = os.path.join(data_dir, "features", instrument + ".csv")
            if not os.path.exists(file_path):
                return None
            df = pd.read_csv(file_path)
            if start_time is not None:
                df = df[(df[timestamp_col] >= start_time)]
            if end_time is not None:
                df = df[(df[timestamp_col] <= end_time)]
        else:
            db_connection = init_db_connection()
            try:
                with db_connection.cursor() as cursor:
                    query = (
                        "SELECT daily.*, daily_basic.turnover_rate, daily_basic.turnover_rate_f, "
                        "daily_basic.volume_ratio, daily_basic.pe, daily_basic.pe_ttm, "
                        "daily_basic.pb, daily_basic.ps, daily_basic.ps_ttm, daily_basic.dv_ratio, "
                        "daily_basic.dv_ttm, daily_basic.total_share, daily_basic.float_share, daily_basic.free_share, "
                        "daily_basic.total_mv, daily_basic.circ_mv, factor.adj_factor "
                        "FROM ts_quotation_daily daily "
                        "JOIN ts_quotation_daily_basic daily_basic ON "
                        "daily.ts_code=daily_basic.ts_code AND "
                        "daily.trade_date=daily_basic.trade_date "
                        "JOIN ts_quotation_adj_factor factor ON "
                        "daily.ts_code=factor.ts_code AND "
                        "daily.trade_date=factor.trade_date "
                        "WHERE daily.ts_code=%s AND daily.trade_date>=%s AND daily.trade_date<=%s LIMIT 50000 "
                    )
                    cursor.execute(
                        query,
                        (
                            instrument,
                            start_time.replace("-", ""),
                            end_time.replace("-", ""),
                        ),
                    )

                    # Fetch all rows and create a DataFrame
                    data = cursor.fetchall()
                    df = pd.DataFrame(
                        data,
                        columns=[
                            "Instrument",
                            "Date",
                            "Open",
                            "Close",
                            "High",
                            "Low",
                            "Pre_Close",
                            "Change",
                            "Pct_Chg",
                            "Volume",
                            "AMount",
                            "Turnover_rate",
                            "Turnover_rate_f",
                            "Volume_ratio",
                            "Pe",
                            "Pe_ttm",
                            "Pb",
                            "Ps",
                            "Ps_ttm",
                            "Dv_ratio",
                            "Dv_ttm",
                            "Total_share",
                            "Float_share",
                            "Free_share",
                            "Total_mv",
                            "Circ_mv",
                            "Adj_factor",
                        ],
                    )

                    # Convert 'Date' column to datetime format
                    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

                    # Format 'Date' column to 'YYYY-MM-DD'
                    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            finally:
                db_connection.close()

        if column_names is not None:
            df = df[column_names]
        df = df.sort_values(by=timestamp_col, ascending=True)
        return df

    @staticmethod
    def load_instruments_features(
        data_dir, instruments, start_time, end_time
    ) -> List[pd.DataFrame]:
        """
        Batch load feature data for multiple instruments.

        Args:
            data_dir (str): dataset directory
            instruments: single or list of instrument symbols
            start_time (str): start time (YYYY-MM-DD)
            end_time (str): end time (YYYY-MM-DD)

        Returns:
            List[pd.DataFrame]: list of feature DataFrames
        """
        if isinstance(instruments, str):
            instruments = DataLoader.load_market_instruments(
                data_dir, instruments, start_time, end_time
            )
        dfs = []
        for instrument in instruments:
            df = DataLoader.load_instrument_features(
                data_dir, instrument=instrument, start_time=start_time, end_time=end_time
            )
            if df is not None:
                dfs.append(df)
        return dfs

    @staticmethod
    def load_market_features(
        data_dir,
        market,
        timestamp_col="Date",
        start_time=None,
        end_time=None,
        column_names=None,
    ) -> pd.DataFrame:
        """
        Args:
            data_dir (str): dataset directory
            market (str):  market name
            timestamp_col (str): column name of timestamp
            start_time (str): start of the time range.
            end_time (str): end of the time range.
            column_names (List[str]): output column names

        Returns:
            pd.DataFrame: dataset load from the files
        """
        if data_dir is not None:
            file_path = os.path.join(data_dir, "features", market + ".csv")
            if not os.path.exists(file_path):
                return None
            df = pd.read_csv(file_path)
            if start_time is not None:
                df = df[(df[timestamp_col] >= start_time)]
            if end_time is not None:
                df = df[(df[timestamp_col] <= end_time)]
        else:
            db_connection = init_db_connection()
            try:
                with db_connection.cursor() as cursor:
                    query = "SELECT * FROM ts_idx_index_daily WHERE index_code=%s AND trade_date>=%s AND trade_date<=%s LIMIT 50000"
                    cursor.execute(
                        query,
                        (
                            market,
                            start_time.replace("-", ""),
                            end_time.replace("-", ""),
                        ),
                    )

                    # Fetch all rows and create a DataFrame
                    data = cursor.fetchall()
                    df = pd.DataFrame(
                        data,
                        columns=[
                            "Instrument",
                            "Date",
                            "Close",
                            "Open",
                            "High",
                            "Low",
                            "Pre_Close",
                            "Change",
                            "Pct_Chg",
                            "Volume",
                            "AMount",
                        ],
                    )

                    # Convert 'Date' column to datetime format
                    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

                    # Format 'Date' column to 'YYYY-MM-DD'
                    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            finally:
                db_connection.close()

        if column_names is not None:
            df = df[column_names]
        df = df.sort_values(by=timestamp_col, ascending=True)
        return df

    @staticmethod
    def load_markets_features(
        data_dir, markets, start_time, end_time
    ) -> Dict[str, pd.DataFrame]:
        """
        Batch load feature data for multiple markets.

        Args:
            data_dir (str): dataset directory
            markets (List[str]): market names
            start_time (str): start time (YYYY-MM-DD)
            end_time (str): end time (YYYY-MM-DD)

        Returns:
            Dict[str, pd.DataFrame]: dictionary of market data with market names as keys
        """

        dfs = {}
        for market in markets:
            df = DataLoader.load_market_features(data_dir, market=market, start_time=start_time, end_time=end_time)
            if df is not None:
                dfs[market] = df
        return dfs

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
        if data_dir is not None:
            file_path = os.path.join(data_dir, "calendars", "day.csv")
            if not os.path.exists(file_path):
                return None
            df = pd.read_csv(file_path)
            if start_time is not None:
                df = df[(df[timestamp_col] >= start_time)]
            if end_time is not None:
                df = df[(df[timestamp_col] <= end_time)]
        else:
            db_connection = init_db_connection()
            try:
                with db_connection.cursor() as cursor:
                    query = (
                        "SELECT DISTINCT exchange, DATE_FORMAT(cal_date, '%Y-%m-%d') FROM ts_basic_trade_cal "
                        "WHERE is_open=1 AND cal_date >= %s AND cal_date <= %s"
                    )

                    cursor.execute(
                        query,
                        (start_time.replace("-", ""), end_time.replace("-", "")),
                    )

                    # Fetch all rows and create a DataFrame
                    data = cursor.fetchall()
                    df = pd.DataFrame(data, columns=["Exchange", "Trade_date"])
            finally:
                db_connection.close()

        trade_dates = df[df["Exchange"] == "SSE"]["Trade_date"].tolist()
        return trade_dates
