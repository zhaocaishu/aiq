import abc
import os
from typing import List
from datetime import timedelta, datetime

import numpy as np
import pandas as pd

from aiq.utils.date import date_add

from .loader import DataLoader
from .handler import Alpha101
from .processor import CSFillna, CSNeutralize, CSFilter, CSZScore

# turn off warnings
pd.options.mode.copy_on_write = True


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
        handlers=None,
        adjust_price=True,
        min_listing_days=365
    ):
        # feature and label names
        ts_handler, cs_handler = handlers
        self.feature_names_ = None
        self.label_name_ = None

        # symbol's name and list date
        self.symbols = DataLoader.load_symbols(data_dir, instruments, min_listing_days=min_listing_days)

        # process per symbol
        dfs = []
        for symbol, list_date in self.symbols:
            df = DataLoader.load_features(data_dir, symbol=symbol, start_time=start_time, end_time=end_time)

            # skip ticker of non-existed
            if df is None: continue

            # append ticker symbol
            df['Symbol'] = symbol

            # adjust price with factor
            if adjust_price:
                df = self.adjust_price(df)

            # extract time-series factors
            if ts_handler is not None:
                df = ts_handler.fetch(df)

            # keep data started from min_listing_days after list date
            cur_start_time = date_add(list_date, n_days=min_listing_days)
            if cur_start_time > start_time:
                df = df[(df['Date'] >= cur_start_time)]

            dfs.append(df)

        # concat dataframes and set index
        self.df = pd.concat(dfs, ignore_index=True)
        self.df = self.df.set_index(['Date', 'Symbol'])

        # assign features and label name
        if ts_handler is not None:
            self.feature_names_ = ts_handler.feature_names
            self.label_name_ = ts_handler.label_name

        # extract cross-sectional factors
        if cs_handler is not None:
            self.df = cs_handler.fetch(self.df)

            if self.feature_names_ is not None:
                self.feature_names_ += cs_handler.feature_names
            else:
                self.feature_names_ = cs_handler.feature_names
            self.label_name_ = cs_handler.label_name

        # processors
        if self.feature_names_ is not None:
            processors = [
                CSFillna(target_cols=self.feature_names_),
                CSFilter(target_cols=self.feature_names_)
            ]

            for processor in processors:
                self.df = processor(self.df)

        # recovery to original price
        if adjust_price:
            self.df = self.de_adjust_price(self.df)

        # reset index
        self.df.reset_index(inplace=True)

    @staticmethod
    def adjust_price(df):
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            df[col] = df[col] * df['Adj_factor']
        return df

    @staticmethod
    def de_adjust_price(df):
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            df[col] = df[col] / df['Adj_factor']
        return df

    def to_dataframe(self):
        return self.df

    def add_column(self, name: str, data: np.array):
        self.df[name] = data

    def slice(self, start_time, end_time):
        return self.df[(self.df['Date'] >= start_time) & (self.df['Date'] <= end_time)]

    @property
    def feature_names(self):
        return self.feature_names_

    @property
    def label_name(self):
        return self.label_name_

    def __getitem__(self, index):
        return self.df.iloc[[index]]

    def __len__(self):
        return self.df.shape[0]


class Subset(Dataset):
    def __init__(self, dataset, start_time, end_time):
        self.feature_names_ = dataset.feature_names_
        self.label_name_ = dataset.label_name_
        self.df = dataset.slice(start_time, end_time)


def ts_split(dataset: Dataset, segments: List[List[str]]):
    return [Subset(dataset, segment[0], segment[1]) for segment in segments]
