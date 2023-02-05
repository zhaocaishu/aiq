import abc
import os

import pandas as pd


class DataLoader(abc.ABC):
    """
    DataLoader is designed for loading raw dataset from original dataset source.
    """
    @staticmethod
    def load(data_dir, symbol, timestamp_col='Date', start_time=None, end_time=None) -> pd.DataFrame:
        """
        Args:
            data_dir (str): dataset directory
            symbol (str):  ticker symbol
            timestamp_col (str): column name of timestamp
            start_time (str): start of the time range.
            end_time (str): end of the time range.

        Returns:
            pd.DataFrame: dataset load from the files
        """
        file_path = os.path.join(data_dir, symbol + '.csv')
        df = pd.read_csv(file_path)
        if start_time is not None:
            df = df[(df[timestamp_col] >= start_time)]
        if end_time is not None:
            df = df[(df[timestamp_col] <= end_time)]
        df.index = df[timestamp_col]
        df.sort_index(inplace=True)
        return df
