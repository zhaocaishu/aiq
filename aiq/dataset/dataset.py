import abc

import pandas as pd

from .loader import DataLoader


class Dataset(abc.ABC):
    """
    Preparing data for model training and inferencing.
    """

    def __init__(self, data_dir, symbols, start_time=None, end_time=None, handler=None):
        df_list = []
        for symbol in symbols:
            df = DataLoader.load(data_dir, symbol, start_time=start_time, end_time=end_time)
            if handler is not None:
                df = handler.fetch(df)
            df_list.append(df)
        self.df = pd.concat(df_list)

    def to_dataframe(self):
        return self.df

    def __getitem__(self, index):
        return self.df.iloc[[index]]

    def __len__(self):
        return self.df.shape[0]