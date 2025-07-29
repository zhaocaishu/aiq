import abc
import warnings
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


class TSRobustZScoreNorm(Processor):
    """Timeseries Robust Z-Score Normalization for MultiIndex DataFrames

    Normalizes each data point using a robust Z-score based on a rolling window of previous data points,
    including multiple instruments in each time window.

    Statistics are computed over all instruments within the window of `window_size` dates.

    Parameters
    ----------
    window_size : int
        Number of dates in the rolling window to use for statistics.
    fields_group : str or list, optional
        Columns to normalize; defaults to all numeric columns.
    clip_outlier : bool, default True
        If True, clips normalized values to [-3, 3].
    exclude_cols : list, optional
        List of columns to exclude from normalization.
    """

    def __init__(
        self,
        window_size: int,
        fields_group: str = None,
        clip_outlier: bool = True,
        exclude_cols: list = None,
    ):
        self.window_size = window_size
        self.fields_group = fields_group
        self.clip_outlier = clip_outlier
        self.exclude_cols = exclude_cols or []

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        # Validate MultiIndex ['Date', 'Instrument']
        if not isinstance(df.index, pd.MultiIndex) or df.index.names[:2] != [
            "Date",
            "Instrument",
        ]:
            raise ValueError("DataFrame must have MultiIndex ['Date', 'Instrument']")

        # Determine columns to normalize
        cols = get_group_columns(df, self.fields_group, self.exclude_cols)

        # Sort by Date and extract values & dates
        df = df.sort_index(level="Date")
        values = df[cols].to_numpy()
        dates = df.index.get_level_values("Date")

        # Identify unique dates and their positions
        unique_dates, start_idxs, counts = np.unique(
            dates, return_index=True, return_counts=True
        )
        end_idxs = start_idxs + counts

        n_dates, n_cols = len(unique_dates), values.shape[1]
        med_arr = np.zeros((n_dates, n_cols))
        std_arr = np.zeros((n_dates, n_cols))

        left = 0
        for right in range(n_dates):
            # Slide window
            if right - left + 1 > self.window_size:
                left += 1

            # Aggregate block for current window
            block = values[start_idxs[left] : end_idxs[right]]

            # Compute median and scaled MAD
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                med = np.nanmedian(block, axis=0)
                mad = np.nanmedian(np.abs(block - med), axis=0)
                std = mad * 1.4826 + 1e-12

            med_arr[right] = med
            std_arr[right] = std

        # Map medians and stds back to each row via searchsorted
        idx_map = np.searchsorted(unique_dates, dates)
        med_full = med_arr[idx_map]
        std_full = std_arr[idx_map]

        # Normalize and clip outliers
        normed = (values - med_full) / std_full
        if self.clip_outlier:
            normed = np.clip(normed, -3, 3)

        # Assign back and return
        df.loc[:, cols] = normed
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
