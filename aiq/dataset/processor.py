import abc
from typing import List

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
    exclude_cols : List[str], optional
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
    """Process NaN"""

    def __init__(self, fields_group=None, fill_value=0):
        self.fields_group = fields_group
        self.fill_value = fill_value

    def __call__(self, df):
        if self.fields_group is None:
            df.fillna(self.fill_value, inplace=True)
        else:
            cols = get_group_columns(df, self.fields_group)

            # So we use numpy to accelerate filling values
            nan_select = np.isnan(df.values)
            nan_select[:, ~df.columns.isin(cols)] = False

            # FIXME: For pandas==2.0.3, the following code will not set the nan value to be self.fill_value
            # df.values[nan_select] = self.fill_value

            # lqa's method
            value_tmp = df.values
            value_tmp[nan_select] = self.fill_value
            df = pd.DataFrame(value_tmp, columns=df.columns, index=df.index)
        return df


class RobustZScoreNorm(Processor):
    """Robust ZScore Normalization

    Use robust statistics for Z-Score normalization:
        mean(x) = median(x)
        std(x) = MAD(x) * 1.4826

    Reference:
        https://en.wikipedia.org/wiki/Median_absolute_deviation.
    """

    def __init__(self, fields_group=None, clip_outlier=True, exclude_cols=[]):
        self.fields_group = fields_group
        self.clip_outlier = clip_outlier
        self.exclude_cols = exclude_cols

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
            X = np.clip(X, -3, 3)
        df[self.cols] = X
        return df


class RollingRobustZScoreNorm(Processor):
    """Rolling Robust Z-Score Normalization

    Normalizes each data point using a robust Z-score based on a rolling window of previous data points.
    For each data point at time t, the mean is estimated using the median of the previous `window_size` data points,
    and the standard deviation is estimated using the Median Absolute Deviation (MAD) multiplied by 1.4826 from the same window.
    This approach is robust to outliers and non-normal distributions.

    Parameters:
    -----------
    window_size : int
        The number of previous data points to use for calculating rolling statistics.
        A larger window provides more stable estimates but is less responsive to recent changes.
    fields_group : str or list, optional
        The column group or list of columns to normalize. If None, all columns in the dataframe are normalized.
    clip_outlier : bool, optional
        If True, clips the normalized values to the range [-3, 3] to limit the impact of extreme outliers. Default is True.
    """

    def __init__(self, window_size, fields_group=None, clip_outlier=True, exclude_cols=[]):
        self.window_size = window_size
        self.fields_group = fields_group
        self.clip_outlier = clip_outlier
        self.exclude_cols = exclude_cols

    def __call__(self, df):
        """Apply rolling robust Z-score normalization to the dataframe.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with a datetime index, assumed to be sorted by time.

        Returns:
        --------
        pd.DataFrame
            Dataframe with specified columns normalized.
        """
        # Get columns to normalize
        cols = get_group_columns(df, self.fields_group, self.exclude_cols)

        # Define MAD function handling NaNs
        def mad(x):
            median = np.nanmedian(x)
            abs_dev = np.abs(x - median)
            return np.nanmedian(abs_dev)

        # Compute rolling median
        rolling_median = df[cols].rolling(self.window_size, min_periods=1).median()

        # Compute rolling MAD
        rolling_mad = df[cols].rolling(self.window_size, min_periods=1).apply(mad, raw=True)

        # Compute robust standard deviation
        rolling_std = rolling_mad * 1.4826 + 1e-12

        # Normalize using statistics from the previous window
        normalized = (df[cols] - rolling_median) / rolling_std

        # Clip outliers if specified
        if self.clip_outlier:
            normalized = normalized.clip(-3, 3)

        # Assign normalized values back to dataframe
        df[cols] = normalized
        return df


class CSNeutralize(Processor):
    """Factors Neutralization"""

    def __init__(self, industry_col: str, cap_col: str, factor_cols: list):
        self.industry_col = industry_col
        self.cap_col = cap_col
        self.factor_cols = factor_cols

    def __call__(self, df):
        df = df.groupby("Date", group_keys=False).apply(
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
        exclude_cols=[],
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
        self.exclude_cols = exclude_cols

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        # Identify numeric columns to winsorize
        cols = get_group_columns(df, self.fields_group, self.exclude_cols)

        def winsorize_slice(slice_df: pd.DataFrame) -> pd.DataFrame:
            # Compute per-column quantiles
            lower_bounds = slice_df.quantile(self.lower_quantile)
            upper_bounds = slice_df.quantile(self.upper_quantile)
            # Clip values to bounds
            return slice_df.clip(lower=lower_bounds, upper=upper_bounds, axis=1)

        # Apply winsorization within each date group
        df[cols] = df[cols].groupby("Date", group_keys=False).apply(winsorize_slice)
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
        # Compute per-date lower and upper quantiles for each column
        for col in cols:
            # vectorized quantile computation via groupby-transform
            lower = (
                df[col]
                .groupby("Date", group_keys=False)
                .transform(lambda x: x.quantile(self.percent))
            )
            upper = (
                df[col]
                .groupby("Date", group_keys=False)
                .transform(lambda x: x.quantile(1 - self.percent))
            )
            # Keep only rows within [lower, upper]
            df = df[df[col].between(lower, upper)]
        return df

    def is_for_infer(self) -> bool:
        return False


class CSZScoreNorm(Processor):
    """Cross Sectional ZScore Normalization"""

    def __init__(self, fields_group=None, method="zscore", exclude_cols=[]):
        self.fields_group = fields_group
        if method == "zscore":
            self.zscore_func = zscore
        elif method == "robust":
            self.zscore_func = robust_zscore
        else:
            raise NotImplementedError(f"This type of input is not supported")
        self.exclude_cols = exclude_cols

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group, self.exclude_cols)
        df[cols] = df[cols].groupby("Date", group_keys=False).apply(self.zscore_func)
        return df


class CSRankNorm(Processor):
    """
    Cross Sectional Rank Normalization.
    "Cross Sectional" is often used to describe data operations.
    The operations across different stocks are often called Cross Sectional Operation.

    For example, CSRankNorm is an operation that grouping the data by each day and rank `across` all the stocks in each day.

    Explanation about 3.46 & 0.5

    .. code-block:: python

        import numpy as np
        import pandas as pd
        x = np.random.random(10000)  # for any variable
        x_rank = pd.Series(x).rank(pct=True)  # if it is converted to rank, it will be a uniform distributed
        x_rank_norm = (x_rank - x_rank.mean()) / x_rank.std()  # Normally, we will normalize it to make it like normal distribution

        x_rank.mean()   # accounts for 0.5
        1 / x_rank.std()  # accounts for 3.46

    """

    def __init__(self, fields_group=None):
        self.fields_group = fields_group

    def __call__(self, df):
        # try not modify original dataframe
        cols = get_group_columns(df, self.fields_group)
        t = df[cols].groupby("Date").rank(pct=True)
        t -= 0.5
        t *= 3.46  # NOTE: towards unit std
        df[cols] = t
        return df
