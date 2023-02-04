import abc
import os

import pandas as pd

from .loader import DataLoader


class Dataset(abc.ABC):
    """
    Preparing data for model training and inferencing.
    """

    def __init__(
            self,
            data_dir,
            start_time=None,
            end_time=None,
            min_periods=30,
            handler=None,
            shuffle=False
    ):
        with open(os.path.join(data_dir, 'instruments/code.txt'), 'r') as f:
            line = f.readline().strip()
            symbols = [symbol for symbol in line.split(',')]

        df_list = []
        for symbol in symbols:
            df = DataLoader.load(os.path.join(data_dir, 'features'), symbol_name=symbol, start_time=start_time,
                                 end_time=end_time)
            if df.shape[0] < min_periods:
                continue
            if handler is not None:
                df = handler.fetch(df)
            df_list.append(df)
        self.df = pd.concat(df_list)

        if shuffle:
            self.df = self.df.sample(frac=1)

    def to_dataframe(self):
        return self.df

    def __getitem__(self, index):
        return self.df.iloc[[index]]

    def __len__(self):
        return self.df.shape[0]
