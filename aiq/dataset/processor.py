import abc
from typing import List, Optional

import pandas as pd
import numpy as np

from aiq.utils.functional import robust_zscore, zscore, neutralize


def get_group_columns(
    df: pd.DataFrame, group: str = None, exclude_cols: List[str] = []
):
    """
    get a group of columns from multi-index columns DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        with multi of columns.
    group : str
        the name of the feature group, i.e. the first level value of the group index.
    exclude_cols : List[str]
        List of column names (from the last level) to exclude from the result.
    """
    if group is None:
        cols = df.columns
    else:
        cols = df.columns[df.columns.get_loc(group)]

    if exclude_cols:
        cols = cols[~cols.get_level_values(-1).isin(exclude_cols)]

    return cols


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

    def is_for_infer(self) -> bool:
        """
        Is this processor usable for inference
        Some processors are not usable for inference.

        Returns
        -------
        bool:
            if it is usable for infenrece.
        """
        return True


class Dropna(Processor):
    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df):
        return df.dropna(subset=get_group_columns(df, self.fields_group))


class Fillna(Processor):
    """Process NaN values by filling with a constant."""

    def __init__(self, fields_group=None, fill_value=0):
        self.fields_group = fields_group
        self.fill_value = fill_value

    def __call__(self, df):
        cols = (
            get_group_columns(df, self.fields_group)
            if self.fields_group
            else df.columns
        )
        if len(cols) > 0:
            df[cols] = df[cols].fillna(self.fill_value)
        return df


class RobustZScoreNorm(Processor):
    """Robust ZScore Normalization

    Use robust statistics for Z-Score normalization:
        mean(x) = median(x)
        std(x) = MAD(x) * 1.4826

    Reference:
        https://en.wikipedia.org/wiki/Median_absolute_deviation.
    """

    def __init__(self, fields_group=None, clip_outlier=True, exclude_cols=None):
        self.fields_group = fields_group
        self.clip_outlier = clip_outlier
        self.exclude_cols = exclude_cols or []

    def fit(self, df: pd.DataFrame = None):
        self.cols = get_group_columns(df, self.fields_group, self.exclude_cols)
        X = df[self.cols].values
        self.mean_train = np.nanmedian(X, axis=0)
        self.std_train = np.nanmedian(np.abs(X - self.mean_train), axis=0)
        self.std_train += 1e-12
        self.std_train *= 1.4826

    def __call__(self, df):
        X = df[self.cols]
        X -= self.mean_train
        X /= self.std_train
        if self.clip_outlier:
            X = X.clip(-3, 3)
        df[self.cols] = X
        return df


class CSNeutralize(Processor):
    """Factors Neutralization"""

    def __init__(
        self, industry_col: str = None, cap_col: str = None, factor_cols: List[str] = []
    ):
        self.industry_col = industry_col
        self.cap_col = cap_col
        self.factor_cols = factor_cols

    def __call__(self, df):
        df = df.groupby(level="Date", group_keys=False).apply(
            neutralize, self.industry_col, self.cap_col, self.factor_cols
        )
        return df


class CSWinsorize(Processor):
    """Cross Sectional Winsorization: winsorize each variable within each date slice."""

    def __init__(
        self,
        fields_group=None,
        lower_quantile=0.01,
        upper_quantile=0.99,
        exclude_cols=None,
    ):
        """
        Parameters
        ----------
        fields_group: grouping key or pattern to select columns (passed to get_group_columns)
        lower_quantile: lower tail cutoff (e.g. 0.01 for 1%)
        upper_quantile: upper tail cutoff (e.g. 0.99 for 99%)
        """
        self.fields_group = fields_group
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.exclude_cols = exclude_cols or []

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        # Identify numeric columns to winsorize
        cols = get_group_columns(df, self.fields_group, self.exclude_cols)

        def winsorize(group: pd.DataFrame) -> pd.DataFrame:
            # Compute per-column quantiles
            lower_bounds = group.quantile(self.lower_quantile)
            upper_bounds = group.quantile(self.upper_quantile)
            # Clip values to bounds
            return group.clip(lower=lower_bounds, upper=upper_bounds, axis=1)

        # Apply winsorization within each date group
        df[cols] = df[cols].groupby(level="Date", group_keys=False).apply(winsorize)
        return df


class DropExtremeLabel(Processor):
    """
    Processor that drops extreme label values within each cross-sectional group.

    For each date, this processor groups the data using `fields_group` (on the label column),
    and removes the lowest `percent` fraction and the highest `percent` fraction of label values.

    Parameters
    ----------
    fields_group : str
        Column name whose values are the labels to filter.
    percent : float
        Fraction of data to drop at each tail (0 < percent < 0.5).
    """

    def __init__(self, fields_group=None, percent: float = 0.025):
        if not (0.0 < percent < 0.5):
            raise ValueError("percent must be between 0 and 0.5")
        self.fields_group = fields_group
        self.percent = percent

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = get_group_columns(df, self.fields_group)

        # Start with an all-True mask
        mask = pd.Series(True, index=df.index)
        for col in cols:
            # Compute per-date quantiles on the ORIGINAL data
            lower = (
                df[col]
                .groupby(level="Date", group_keys=False)
                .transform(lambda x: x.quantile(self.percent))
            )
            upper = (
                df[col]
                .groupby(level="Date", group_keys=False)
                .transform(lambda x: x.quantile(1 - self.percent))
            )
            # Combine conditions: row must be within bounds for this column
            mask &= df[col].between(lower, upper)

        # Apply the final mask once
        return df[mask]

    def is_for_infer(self) -> bool:
        return False


class CSZScoreNorm(Processor):
    """Cross Sectional ZScore Normalization"""

    def __init__(self, fields_group=None, method="zscore", exclude_cols=None):
        self.fields_group = fields_group
        if method == "zscore":
            self.zscore_func = zscore
        elif method == "robust":
            self.zscore_func = robust_zscore
        else:
            raise NotImplementedError(f"This type of input is not supported")
        self.exclude_cols = exclude_cols or []

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group, self.exclude_cols)
        df[cols] = (
            df[cols].groupby(level="Date", group_keys=False).apply(self.zscore_func)
        )
        return df


class CSRankNorm(Processor):
    """
    Cross Sectional Rank Normalization.

    This processor ranks values across all stocks for each day and
    then normalizes the ranks to have zero mean and unit variance.
    """

    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group)

        def normalize_group(group):
            ranks = group.rank(method="average")  # rank from 1
            ranks = (ranks - 1) / (len(ranks) - 1)  # scale to [0,1]
            return (ranks - ranks.mean()) / ranks.std(ddof=0)  # z-score

        for col in cols:
            df[col] = (
                df[col].groupby(level="Date", group_keys=False).apply(normalize_group)
            )

        return df
