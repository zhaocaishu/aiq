import abc
import os
from datetime import datetime
from typing import List

import pandas as pd

from aiq.utils.date import now_date, date_diff


class DataLoader(abc.ABC):
    """
    DataLoader is designed for loading raw dataset from original dataset source.
    """

    @staticmethod
    def load_symbols(data_dir, instruments, start_time=None, end_time=None):
        """
        Args:
            data_dir (str): dataset directory
            instruments (str): index names separated by ','
            start_time (str): start time
            end_time (str): end_time

        Returns:
            List[Tuple[str]]: list of symbol's name and list date
        """
        symbols = set()
        instrument_names = instruments.split(',')
        for instrument_name in instrument_names:
            file_path = os.path.join(data_dir, 'instruments', instrument_name + '.csv')
            df = pd.read_csv(file_path)
            if start_time is not None:
                df = df[df['Date'] >= start_time]
            if end_time is not None:
                df = df[df['Date'] <= end_time]
            for index, row in df.iterrows():
                symbols.add((row['Symbol'], row['List_date']))

        return list(symbols)

    @staticmethod
    def load_features(data_dir, symbol, timestamp_col='Date', start_time=None, end_time=None,
                      column_names=None) -> pd.DataFrame:
        """
        Args:
            data_dir (str): dataset directory
            symbol (str):  ticker symbol
            timestamp_col (str): column name of timestamp
            start_time (str): start of the time range.
            end_time (str): end of the time range.
            column_names (List[str]): output column names

        Returns:
            pd.DataFrame: dataset load from the files
        """
        file_path = os.path.join(data_dir, 'features', symbol + '.csv')
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
    def load_calendars(data_dir, timestamp_col='Trade_date', start_time=None, end_time=None) -> List[str]:
        """

        Args:
            data_dir (str): dataset directory
            timestamp_col (str): column name of timestamp
            start_time (str): start of the time range.
            end_time (str): end of the time range.

        Returns:

        """
        file_path = os.path.join(data_dir, 'calendars', 'days.csv')
        if not os.path.exists(file_path):
            return None
        df = pd.read_csv(file_path)
        if start_time is not None:
            df = df[(df[timestamp_col] >= start_time)]
        if end_time is not None:
            df = df[(df[timestamp_col] <= end_time)]

        trade_dates = df[df['Exchange'] == 'SZ']['Trade_date'].tolist()
        return trade_dates
