import abc
import os

import numpy as np
import pandas as pd

from .loader import DataLoader
from .processor import CSZScoreNorm


class Dataset(abc.ABC):
    """
    Preparing data for model training and inference.
    """

    def __init__(
        self,
        data_dir,
        instruments,
        start_time=None,
        end_time=None,
        handler=None,
        min_periods=60,
        adjust_price=False,
        shuffle=False
    ):
        with open(os.path.join(data_dir, 'instruments/%s.txt' % instruments), 'r') as f:
            self.symbols = [line.strip().split()[0] for line in f.readlines()]

        df_list = []
        for symbol in self.symbols:
            df = DataLoader.load(os.path.join(data_dir, 'features'), symbol=symbol, start_time=start_time,
                                 end_time=end_time)

            # skip ticker of non-existed or small periods
            if df is None or df.shape[0] < min_periods:
                continue

            # append ticker symbol
            df['Symbol'] = symbol

            # adjust price with factor
            if adjust_price:
                df = self.adjust_price(df)

            # extract ticker factors
            if handler is not None:
                df = handler.fetch(df)

            df_list.append(df)
        # concat and reset index
        self.df = pd.concat(df_list)
        self.df.reset_index(inplace=True)
        print('Loaded %d symbols to build dataset' % len(df_list))

        # random shuffle
        if shuffle:
            self.df = self.df.sample(frac=1)

    @staticmethod
    def adjust_price(df):
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            df[col] = df[col] * df['Adj_factor'] / df['Adj_factor'].iloc[-1]
        return df

    def to_dataframe(self):
        return self.df

    def add_column(self, name: str, data: np.array):
        self.df[name] = data

    def dump(self, output_dir: str = None):
        if output_dir is None:
            return

        if not os.path.exists(path=output_dir):
            os.makedirs(output_dir)

        for symbol in self.symbols:
            df_symbol = self.df[self.df['Symbol'] == symbol]
            if df_symbol.shape[0] > 0:
                df_symbol.to_csv(os.path.join(output_dir, symbol + '.csv'), na_rep='NaN', index=False)

    def __getitem__(self, index):
        return self.df.iloc[[index]]

    def __len__(self):
        return self.df.shape[0]
