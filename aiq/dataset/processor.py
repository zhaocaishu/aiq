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
        df[self.target_cols] = df[self.target_cols].groupby('Date', group_keys=False).apply(lambda x: x.fillna(x.mean()))
        return df


class CSZScore(Processor):
    """ZScore Normalization"""

    def __init__(self, target_cols=None):
        self.target_cols = target_cols
        self.norm_func = zscore

    def __call__(self, df):
        df[self.target_cols] = df[self.target_cols].groupby('Date', group_keys=False).apply(self.norm_func)
        return df


class CSProcessor(Processor):
    """
    This processor is designed for Alpha158. And will be replaced by simple processors in the future
    """

    def __init__(self, fillna_feature=True, clip_feature_outlier=True, shrink_feature_outlier=True,
                 norm_label=False, clip_label_outlier=False, fillna_label=False):
        # Options
        self.fillna_feature = fillna_feature
        self.clip_feature_outlier = shrink_feature_outlier
        self.shrink_feature_outlier = clip_label_outlier
        self.norm_label = norm_label
        self.fillna_label = clip_feature_outlier
        self.clip_label_outlier = fillna_label

    def __call__(self, df):
        return self._transform(df)

    def _transform(self, df):
        def _label_norm(x):
            x = x - x.mean()  # copy
            x /= x.std()
            if self.clip_label_outlier:
                x.clip(-3, 3, inplace=True)
            if self.fillna_label:
                x.fillna(0, inplace=True)
            return x

        def _feature_norm(x):
            x = x - x.median()  # copy
            x /= x.abs().median() * 1.4826
            if self.clip_feature_outlier:
                x.clip(-3, 3, inplace=True)
            if self.shrink_feature_outlier:
                x.where(x <= 3, 3 + (x - 3).div(x.max() - 3) * 0.5, inplace=True)
                x.where(x >= -3, -3 - (x + 3).div(x.min() + 3) * 0.5, inplace=True)
            if self.fillna_feature:
                x.fillna(0, inplace=True)
            return x

        # Label
        if self.norm_label:
            cols = df.columns[df.columns.str.contains("^LABEL")]
            df[cols] = df[cols].groupby('Date', group_keys=False).apply(_label_norm)

        # Features
        cols = df.columns[df.columns.str.contains("^KLEN|^KLOW|^KUP")]
        df[cols] = df[cols].apply(lambda x: x**0.25).groupby('Date', group_keys=False).apply(_feature_norm)

        cols = df.columns[df.columns.str.contains("^KLOW2|^KUP2")]
        df[cols] = df[cols].apply(lambda x: x**0.5).groupby('Date', group_keys=False).apply(_feature_norm)

        _cols = [
            "KMID",
            "KSFT",
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE",
            "ROC",
            "MA",
            "BETA",
            "RESI",
            "QTLU",
            "QTLD",
            "RSV",
            "SUMP",
            "SUMN",
            "SUMD",
            "VSUMP",
            "VSUMN",
            "VSUMD",
            "BPLF",
            "EPTTM",
            "DVTTM"
        ]
        pat = "|".join(["^" + x for x in _cols])
        cols = df.columns[df.columns.str.contains(pat) & (~df.columns.isin(["HIGH0", "LOW0"]))]
        df[cols] = df[cols].groupby('Date', group_keys=False).apply(_feature_norm)

        cols = df.columns[df.columns.str.contains("^STD|^VOLUME|^VMA|^VSTD")]
        df[cols] = df[cols].apply(np.log).groupby('Date', group_keys=False).apply(_feature_norm)

        cols = df.columns[df.columns.str.contains("^RSQR")]
        df[cols] = df[cols].fillna(0).groupby('Date', group_keys=False).apply(_feature_norm)

        cols = df.columns[df.columns.str.contains("^MAX|^HIGH0")]
        df[cols] = df[cols].apply(lambda x: (x - 1) ** 0.5).groupby('Date', group_keys=False).apply(_feature_norm)

        cols = df.columns[df.columns.str.contains("^MIN|^LOW0")]
        df[cols] = df[cols].apply(lambda x: (1 - x) ** 0.5).groupby('Date', group_keys=False).apply(_feature_norm)

        cols = df.columns[df.columns.str.contains("^CORR|^CORD")]
        df[cols] = df[cols].apply(np.exp).groupby('Date', group_keys=False).apply(_feature_norm)

        cols = df.columns[df.columns.str.contains("^WVMA")]
        df[cols] = df[cols].apply(np.log1p).groupby('Date', group_keys=False).apply(_feature_norm)

        return df
