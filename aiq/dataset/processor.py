import abc

import pandas as pd
import numpy as np

from aiq.utils.processing import robust_zscore, zscore, neutralize


def get_group_columns(
    df: pd.DataFrame, group: str = None, exclude_discrete: bool = False
):
    """
    get a group of columns from multi-index columns DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        with multi of columns.
    group : str
        the name of the feature group, i.e. the first level value of the group index.
    exclude_discrete : bool
        whether exclude discrete columns
    """
    if group is None:
        cols = df.columns
    else:
        cols = df.columns[df.columns.get_loc(group)]

    if exclude_discrete:
        filtered_cols = cols[~cols.get_level_values(-1).str.endswith("_CAT")]
    else:
        filtered_cols = cols

    return filtered_cols


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

    def __init__(self, fields_group=None, clip_outlier=True):
        self.fields_group = fields_group
        self.clip_outlier = clip_outlier

    def fit(self, df: pd.DataFrame = None):
        self.cols = get_group_columns(df, self.fields_group, exclude_discrete=True)
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

    def __init__(self, fields_group=None, lower_quantile=0.01, upper_quantile=0.99):
        """
        :param fields_group: grouping key or pattern to select columns (passed to get_group_columns)
        :param lower_quantile: lower tail cutoff (e.g. 0.01 for 1%)
        :param upper_quantile: upper tail cutoff (e.g. 0.99 for 99%)
        """
        self.fields_group = fields_group
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        # Identify numeric columns to winsorize, excluding discrete if specified
        cols = get_group_columns(df, self.fields_group, exclude_discrete=True)

        def winsorize_slice(slice_df: pd.DataFrame) -> pd.DataFrame:
            """
            Winsorize values in each column of the slice.

            :param slice_df: DataFrame subset for a single date
            :return: winsorized DataFrame
            """
            # Compute per-column quantiles
            lower_bounds = slice_df.quantile(self.lower_quantile)
            upper_bounds = slice_df.quantile(self.upper_quantile)
            # Clip values to bounds
            return slice_df.clip(lower=lower_bounds, upper=upper_bounds, axis=1)

        # Apply winsorization within each date group
        df[cols] = df[cols].groupby("Date", group_keys=False).apply(winsorize_slice)
        return df


class CSZScoreNorm(Processor):
    """Cross Sectional ZScore Normalization"""

    def __init__(self, fields_group=None, method="zscore"):
        self.fields_group = fields_group
        if method == "zscore":
            self.zscore_func = zscore
        elif method == "robust":
            self.zscore_func = robust_zscore
        else:
            raise NotImplementedError(f"This type of input is not supported")

    def __call__(self, df):
        cols = get_group_columns(df, self.fields_group, exclude_discrete=True)
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
