import abc
import os
from typing import List

import numpy as np
import pandas as pd

from .loader import DataLoader
from .processor import CSLabelNorm


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
        adjust_price=True,
        training=False
    ):
        # feature and label names
        self._feature_names = None
        self._label_name = None

        # symbol of instruments
        with open(os.path.join(data_dir, 'instruments/%s.txt' % instruments), 'r') as f:
            self.symbols = [line.strip().split()[0] for line in f.readlines()]

        # process per symbol
        dfs = []
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

            dfs.append(df)

        # concat and reset index
        self.df = pd.concat(dfs)
        self.df.reset_index(inplace=True)

        # assign features and label name
        if handler is not None:
            self._feature_names = handler.feature_names
            self._label_name = handler.label_name

        # normalize label
        if self._label_name is not None:
            processor = CSLabelNorm(cols=[self._label_name])
            self.df = processor.transform(self.df)

        # random shuffle
        if training:
            self.df = self.df.sample(frac=1.0)

    @staticmethod
    def adjust_price(df):
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            df[col] = df[col] * df['Adj_factor']
        return df

    def to_dataframe(self):
        return self.df

    def add_column(self, name: str, data: np.array):
        self.df[name] = data

    def slice(self, start_time, end_time):
        return self.df[(self.df['Date'] >= start_time) & (self.df['Date'] <= end_time)].copy()

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def label_name(self):
        return self._label_name

    def __getitem__(self, index):
        return self.df.iloc[[index]]

    def __len__(self):
        return self.df.shape[0]


class Subset(Dataset):
    def __init__(self, dataset, start_time, end_time):
        self.df = dataset.slice(start_time, end_time)


def random_split(dataset: Dataset, segments: List[List[str]]):
    return [Subset(dataset, segment[0], segment[1]) for segment in segments]
