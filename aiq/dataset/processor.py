import abc

import pandas as pd
import numpy as np

from aiq.utils.data import robust_zscore, zscore


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


class DropnaProcessor(Processor):
    def __init__(self, target_cols=None):
        self.target_cols = target_cols

    def __call__(self, df):
        return df.dropna(subset=self.target_cols)


class Fillna(Processor):
    """Process NaN"""

    def __init__(self, target_cols=None, fill_value=0):
        self.target_cols = target_cols
        self.fill_value = fill_value

    def __call__(self, df):
        if self.target_cols is None:
            df.fillna(self.fill_value, inplace=True)
        else:
            # So we use numpy to accelerate filling values
            nan_select = np.isnan(df.values)
            nan_select[:, ~df.columns.isin(self.target_cols)] = False

            # FIXME: For pandas==2.0.3, the following code will not set the nan value to be self.fill_value
            # df.values[nan_select] = self.fill_value

            # lqa's method
            value_tmp = df.values
            value_tmp[nan_select] = self.fill_value
            df = pd.DataFrame(value_tmp, columns=df.columns, index=df.index)
        return df


class CSZScoreNorm(Processor):
    """Cross Sectional ZScore Normalization"""

    def __init__(self, target_cols=None, method="zscore"):
        self.target_cols = target_cols
        if method == "zscore":
            self.zscore_func = zscore
        elif method == "robust":
            self.zscore_func = robust_zscore
        else:
            raise NotImplementedError(f"This type of input is not supported")

    def __call__(self, df):
        df[self.target_cols] = df[self.target_cols].groupby("Date", group_keys=False).apply(self.zscore_func)
        return df