import abc
import os

import numpy as np
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
            self.symbols = [symbol for symbol in line.split(',')]

        df_list = []
        for symbol in self.symbols:
            df = DataLoader.load(os.path.join(data_dir, 'features'), symbol=symbol, start_time=start_time,
                                 end_time=end_time)
            # skip ticker of small periods
            if df.shape[0] < min_periods:
                continue

            # extract ticker factors
            if handler is not None:
                df = handler.fetch(df)

            # append ticker symbol
            df['Symbol'] = symbol
            df_list.append(df)
        self.df = pd.concat(df_list)

        if shuffle:
            self.df = self.df.sample(frac=1)

    def to_dataframe(self):
        return self.df

    def add_column(self, name: str, data: np.array):
        self.df[name] = data

    def dump(self, output_dir: str=None):
        if output_dir is None:
            return

        if not os.path.exists(path=output_dir):
            os.makedirs(output_dir)

        for symbol in self.symbols:
            df_symbol = self.df[self.df['Symbol'] == symbol]
            if df_symbol.shape[0] > 0:
                df_symbol.to_csv(os.path.join(output_dir, symbol + '.csv'), na_rep='NaN')

    def __getitem__(self, index):
        return self.df.iloc[[index]]

    def __len__(self):
        return self.df.shape[0]
