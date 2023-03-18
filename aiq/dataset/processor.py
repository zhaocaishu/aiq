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


class CSLabelClip(Processor):
    """Cross Sectional Label Outlier Clip"""

    def __init__(self, label_col=None, clip_outlier=True, high_limit=0.098, low_limit=-0.098):
        self.label_col = label_col
        self.clip_outlier = clip_outlier
        self.high_limit = high_limit
        self.low_limit = low_limit

    def transform(self, df):
        if self.clip_outlier:
            df = df[(df[self.label_col] <= self.high_limit) & (df[self.label_col] >= self.low_limit)]
        return df
