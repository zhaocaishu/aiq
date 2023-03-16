import abc
from typing import Union, Text

import pandas as pd
import numpy as np


class Processor(abc.ABC):
    def fit(self, df: pd.DataFrame = None):
        """
        Learn data processing parameters

        Args:
            df (pd.DataFrame): When we fit and process data with processor one by one. The fit function reiles on the output of previous
            processor, i.e. `df`.

        """

    def transform(self, df: pd.DataFrame):
        """
        Process the data
        NOTE: **The processor could change the content of `df` inplace !!!!! **
        User should keep a copy of data outside

        Args:
            df (pd.DataFrame): The raw_df of handler or result from previous processor.
        """


class CSZScoreNorm(Processor):
    """Cross Sectional ZScore Normalization

     Use robust statistics for Z-Score normalization:
        mean(x) = median(x)
        std(x) = MAD(x) * 1.4826
    """
    def __init__(self, cols=None, clip_outlier=True):
        self.cols = cols
        self.clip_outlier = clip_outlier

    def fit(self, df):
        X = df[self.cols].values
        self.mean_train = np.nanmedian(X, axis=0)
        self.std_train = np.nanmedian(np.abs(X - self.mean_train), axis=0)
        self.std_train += 1e-12
        self.std_train *= 1.4826

        return df

    def transform(self, df):
        def normalize(x, mean_train=self.mean_train, std_train=self.std_train):
            return (x - mean_train) / std_train

        df.loc(axis=1)[self.cols] = normalize(df[self.cols].values)
        return df
