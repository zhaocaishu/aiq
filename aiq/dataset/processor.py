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
        # 1. 校验 MultiIndex
        if not isinstance(df.index, pd.MultiIndex) or df.index.names[0] != "Date":
            raise ValueError("DataFrame must have MultiIndex ['Date', 'Instrument']")

        # 2. 待归一化列
        cols = get_group_columns(df, self.fields_group, self.exclude_cols)

        # 1. 确保按日期升序
        df = df.sort_index(level="Date")
        # 2. 把原始值提取为 NumPy 数组
        values = df[cols].values  # shape = (n_rows, n_cols)
    
        # 3. 计算每个日期在原 df 中的起止行号（假定同一天的数据在 df 中是连续的）
        #    得到 unique_dates，以及对应每个日期在 values 中的 [start_idx, end_idx)
        date_index = df.index.get_level_values("Date")
        unique_dates, start_idxs, counts = np.unique(
            date_index, return_index=True, return_counts=True
        )
        # start_idxs 是第一个出现位置，counts 是每个日期的行数
        end_idxs = (
            start_idxs + counts
        )  # 对应每个日期切片是 values[start_idxs[i]:end_idxs[i]]
    
        n_dates = len(unique_dates)
        n_cols = values.shape[1]
    
        # 4. 准备用于存放滑窗结果的数组
        med_arr = np.zeros((n_dates, n_cols), dtype=float)
        std_arr = np.zeros((n_dates, n_cols), dtype=float)
    
        # 5. 滑动窗口：用两个指针维护窗口在 unique_dates 上的起止
        left = 0
        for right in range(n_dates):
            # 当窗口大小超过 window_size 时，左指针右移
            if right - left + 1 > window_size:
                left += 1
    
            # 本次窗口在 values 中的起止行号
            win_start = start_idxs[left]
            win_end = end_idxs[right]
            block = values[win_start:win_end]  # shape (~window_rows, n_cols)
    
            med = np.nanmedian(block, axis=0)  # C 实现
            mad = np.nanmedian(np.abs(block - med), axis=0)
            std = mad * 1.4826 + 1e-12
    
            med_arr[right] = med
            std_arr[right] = std
    
        # 6. 将 (n_dates, n_cols) 的 med/std 扩展回原始行数
        #    先构建以 unique_dates 为索引的 DataFrame
        med_df = pd.DataFrame(med_arr, index=unique_dates, columns=cols)
        std_df = pd.DataFrame(std_arr, index=unique_dates, columns=cols)
    
        #    然后按原始行的日期级别来重索引并获取底层值
        full_med = med_df.reindex(date_index).values  # shape = (n_rows, n_cols)
        full_std = std_df.reindex(date_index).values
    
        # 7. 最终归一化
        normed = (values - full_med) / full_std
        if self.clip_outlier:
            normed = np.clip(normed, -3, 3)
    
        # 8. 替换回原 df 并返回
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
