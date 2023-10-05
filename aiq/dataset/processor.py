import os
import abc
import pickle
from typing import Union, Text

import pandas as pd
import numpy as np

from aiq.utils.data import mad_filter, neutralize, zscore


class Processor(abc.ABC):
    def fit(self, df: pd.DataFrame = None):
        """
        learn data processing parameters
        Parameters
        ----------
        df : pd.DataFrame
            When we fit and process data with processor one by one. The fit function reiles on the output of previous
            processor, i.e. `df`.
        """

    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame):
        """
        process the data
        NOTE: **The processor could change the content of `df` inplace !!!!! **
        User should keep a copy of data outside
        Parameters
        ----------
        df : pd.DataFrame
            The raw_df of handler or result from previous processor.
        """


class CSFilter(Processor):
    """Outlier filter"""

    def __init__(self, target_cols=None, method="mad"):
        self.target_cols = target_cols
        if method == "mad":
            self.filter_func = mad_filter
        else:
            raise NotImplementedError(f"This type of method is not supported")

    def __call__(self, df):
        df[self.target_cols] = df[self.target_cols].groupby('Date', group_keys=False).apply(self.filter_func)
        return df


class CSNeutralize(Processor):
    """Factor neutralize by industry and market value"""

    def __init__(self, industry_num, industry_col=None, market_cap_col=None, target_cols=None):
        self.industry_num = industry_num
        self.industry_col = industry_col
        self.market_cap_col = market_cap_col
        self.target_cols = target_cols
        self.neutralize_func = neutralize

    def __call__(self, df):
        df = df.groupby('Date', group_keys=False).apply(self.neutralize_func,
                                                        industry_num=self.industry_num,
                                                        industry_col=self.industry_col,
                                                        market_cap_col=self.market_cap_col,
                                                        target_cols=self.target_cols)
        return df


class CSFillna(Processor):
    """Cross Sectional Fill Nan"""

    def __init__(self, target_cols=None):
        self.target_cols = target_cols

    def __call__(self, df):
        df[self.target_cols] = df[self.target_cols].groupby('Date', group_keys=False).apply(
            lambda x: x.fillna(x.mean()))
        return df


class CSZScore(Processor):
    """ZScore Normalization"""

    def __init__(self, target_cols=None):
        self.target_cols = target_cols
        self.norm_func = zscore

    def __call__(self, df):
        df[self.target_cols] = df[self.target_cols].groupby('Date', group_keys=False).apply(self.norm_func)
        return df


class TSStandardize(Processor):
    def __init__(self, target_cols=None, save_dir=None):
        self.target_cols = target_cols
        self.save_dir = save_dir

        if os.path.exists(os.path.join(self.save_dir, 'standardize.pkl')):
            with open(os.path.join(self.save_dir, 'standardize.pkl'), 'rb') as f:
                data = pickle.load(f)
                self.mean = data['mean']
                self.std = data['std']
        else:
            self.mean = None
            self.std = None

    def fit(self, df: pd.DataFrame = None):
        self.mean = df[self.target_cols].groupby('Symbol', group_keys=False).mean()
        self.std = df[self.target_cols].groupby('Symbol', group_keys=False).std()

        with open(os.path.join(self.save_dir, 'standardize.pkl'), 'wb') as f:
            pickle.dump({'mean': self.mean, 'std': self.std}, f)

    def __call__(self, df: pd.DataFrame = None):
        # filter symbols without mean and std data
        symbols = []
        for symbol in df.index.unique():
            if symbol in self.mean.index.unique():
                symbols.append(symbol)
        df = df.loc[symbols]

        # standardize transform
        for symbol in symbols:
            df.loc[symbol, self.target_cols] = (df.loc[symbol, self.target_cols] - self.mean.loc[symbol]) / \
                                               self.std.loc[symbol]
        return df
