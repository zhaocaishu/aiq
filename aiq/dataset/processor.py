import os
import abc
import pickle
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


class CSLabelNorm(Processor):
    """Cross Sectional Label Normalization"""
    def __init__(self, cols=None, clip_outlier=True):
        self.cols = cols
        self.clip_outlier = clip_outlier

    def transform(self, df):
        X = df[self.cols]
        if self.clip_outlier:
            X = np.clip(X, -0.1, 0.1)
        df[self.cols] = X
        return df
