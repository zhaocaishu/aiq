import abc

import pandas as pd
import numpy as np

from .libs.rolling import rolling_slope, rolling_rsquare, rolling_resi
from .libs.expanding import expanding_slope, expanding_rsquare, expanding_resi


class NpPairOperator(abc.ABC):
    """Numpy Pair-wise operator"""

    def __init__(self, func):
        self.func = func

    def __call__(self, series_left, series_right):
        res = getattr(np, self.func)(series_left, series_right)
        return res


class Greater(NpPairOperator):
    """Greater Operator"""

    def __init__(self):
        super(Greater, self).__init__("maximum")


class Less(NpPairOperator):
    """Less Operator"""

    def __init__(self):
        super(Less, self).__init__("minimum")


class NpElemOperator(abc.ABC):
    """Numpy Element-wise Operator"""

    def __init__(self, func):
        self.func = func
        super(NpElemOperator, self).__init__()

    def __call__(self, series: pd.Series):
        return getattr(np, self.func)(series)


class Log(NpElemOperator):
    """Feature Log"""

    def __init__(self):
        super(Log, self).__init__("log")


class Abs(NpElemOperator):
    """Feature Absolute Value"""

    def __init__(self):
        super(Abs, self).__init__("abs")


class Rolling(abc.ABC):
    """Rolling Operator
    The meaning of rolling and expanding is the same in pandas.
    When the window is set to 0, the behaviour of the operator should follow `expanding`
    Otherwise, it follows `rolling`
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, series: pd.Series, N):
        # NOTE: remove all null check,
        # now it's user's responsibility to decide whether use features in null days
        # isnull = series.isnull() # NOTE: isnull = NaN, inf is not null
        if isinstance(N, int) and N == 0:
            series = getattr(series.expanding(min_periods=1), self.func)()
        elif isinstance(N, float) and 0 < N < 1:
            series = series.ewm(alpha=N, min_periods=1).mean()
        else:
            series = getattr(series.rolling(N, min_periods=1), self.func)()
        return series


class Ref(Rolling):
    def __init__(self):
        super(Ref, self).__init__("ref")

    def __call__(self, series: pd.Series, N):
        """
        Args:
            N (int):  N = 0, retrieve the first data; N > 0, retrieve data of N periods ago; N < 0, future data
        """
        # N = 0, return first day
        if series.empty:
            return series  # Pandas bug, see: https://github.com/pandas-dev/pandas/issues/21049
        elif N == 0:
            series = pd.Series(series.iloc[0], index=series.index)
        else:
            series = series.shift(N)  # copy
        return series


class Mean(Rolling):
    """Rolling Mean"""

    def __init__(self):
        super(Mean, self).__init__("mean")


class Sum(Rolling):
    """Rolling Sum"""

    def __init__(self):
        super(Sum, self).__init__("sum")


class Std(Rolling):
    """Rolling Std"""

    def __init__(self):
        super(Std, self).__init__("std")


class Var(Rolling):
    """Rolling Variance"""

    def __init__(self):
        super(Var, self).__init__("var")


class Skew(Rolling):
    """Rolling Skewness"""

    def __init__(self):
        super(Skew, self).__init__("skew")


class Kurt(Rolling):
    """Rolling Kurtosis"""

    def __init__(self):
        super(Kurt, self).__init__("kurt")


class Max(Rolling):
    """Rolling Max"""

    def __init__(self):
        super(Max, self).__init__("max")


class IdxMax(Rolling):
    """Rolling Max Index"""

    def __init__(self):
        super(IdxMax, self).__init__("idxmax")

    def __call__(self, series: pd.Series, N):
        if N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        else:
            series = series.rolling(N, min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        return series


class Min(Rolling):
    """Rolling Min"""

    def __init__(self):
        super(Min, self).__init__("min")


class IdxMin(Rolling):
    """Rolling Min Index"""

    def __init__(self):
        super(IdxMin, self).__init__("idxmin")

    def __call__(self, series, N):
        if N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        else:
            series = series.rolling(N, min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        return series


class Quantile(Rolling):
    """Rolling Quantile"""

    def __init__(self):
        super(Quantile, self).__init__("quantile")

    def __call__(self, series: pd.Series, N, qscore):
        if N == 0:
            series = series.expanding(min_periods=1).quantile(qscore)
        else:
            series = series.rolling(N, min_periods=1).quantile(qscore)
        return series


class Med(Rolling):
    """Rolling Median"""

    def __init__(self):
        super(Med, self).__init__("median")


class Mad(Rolling):
    """Rolling Mean Absolute Deviation"""

    def __init__(self):
        super(Mad, self).__init__("mad")

    def __call__(self, series: pd.Series, N):
        def mad(x):
            x1 = x[~np.isnan(x)]
            return np.mean(np.abs(x1 - x1.mean()))

        if N == 0:
            series = series.expanding(min_periods=1).apply(mad, raw=True)
        else:
            series = series.rolling(N, min_periods=1).apply(mad, raw=True)
        return series


class Rank(Rolling):
    """Rolling Rank (Percentile)"""

    def __init__(self):
        super(Rank, self).__init__("rank")

    # for compatiblity of python 3.7, which doesn't support pandas 1.4.0+ which implements Rolling.rank
    def __call__(self, series: pd.Series, N):
        rolling_or_expending = series.expanding(min_periods=1) if N == 0 else series.rolling(N, min_periods=1)
        return rolling_or_expending.rank(pct=True)


class Count(Rolling):
    """Rolling Count"""

    def __init__(self):
        super(Count, self).__init__("count")


class Delta(Rolling):
    """Rolling Delta"""

    def __init__(self):
        super(Delta, self).__init__("delta")

    def __call__(self, series: pd.Series, N):
        if N == 0:
            series = series - series.iloc[0]
        else:
            series = series - series.shift(N)
        return series


# TODO:
# support pair-wise rolling like `Slope(A, B, N)`
class Slope(Rolling):
    """Rolling Slope"""

    def __init__(self):
        super(Slope, self).__init__("slope")

    def __call__(self, series: pd.Series, N):
        if N == 0:
            series = pd.Series(expanding_slope(series.values), index=series.index)
        else:
            series = pd.Series(rolling_slope(series.values, N), index=series.index)
        return series


class Rsquare(Rolling):
    """Rolling R-value Square"""

    def __init__(self):
        super(Rsquare, self).__init__("rsquare")

    def __call__(self, _series: pd.Series, N):
        if N == 0:
            series = pd.Series(expanding_rsquare(_series.values), index=_series.index)
        else:
            series = pd.Series(rolling_rsquare(_series.values, N), index=_series.index)
            series.loc[np.isclose(_series.rolling(N, min_periods=1).std(), 0, atol=2e-05)] = np.nan
        return series


class Resi(Rolling):
    """Rolling Regression Residuals"""

    def __init__(self):
        super(Resi, self).__init__("resi")

    def __call__(self, series: pd.Series, N):
        if N == 0:
            series = pd.Series(expanding_resi(series.values), index=series.index)
        else:
            series = pd.Series(rolling_resi(series.values, N), index=series.index)
        return series


class WMA(Rolling):
    """Rolling WMA"""

    def __init__(self):
        super(WMA, self).__init__("wma")

    def __call__(self, series: pd.Series, N):
        def weighted_mean(x):
            w = np.arange(len(x)) + 1
            w = w / w.sum()
            return np.nanmean(w * x)

        if N == 0:
            series = series.expanding(min_periods=1).apply(weighted_mean, raw=True)
        else:
            series = series.rolling(N, min_periods=1).apply(weighted_mean, raw=True)
        return series


class EMA(Rolling):
    """Rolling Exponential Mean (EMA)"""

    def __init__(self):
        super(EMA, self).__init__("ema")

    def __call__(self, series: pd.Series, N):
        def exp_weighted_mean(x):
            a = 1 - 2 / (1 + len(x))
            w = a ** np.arange(len(x))[::-1]
            w /= w.sum()
            return np.nansum(w * x)

        if N == 0:
            series = series.expanding(min_periods=1).apply(exp_weighted_mean, raw=True)
        elif 0 < N < 1:
            series = series.ewm(alpha=N, min_periods=1).mean()
        else:
            series = series.ewm(span=N, min_periods=1).mean()
        return series


class PairRolling(abc.ABC):
    """Pair Rolling Operator"""

    def __init__(self, func):
        self.func = func

    def __call__(self, series_left: pd.Series, series_right: pd.Series, N):
        if N == 0:
            series = getattr(series_left.expanding(min_periods=1), self.func)(series_right)
        else:
            series = getattr(series_left.rolling(N, min_periods=1), self.func)(series_right)
        return series


class Corr(PairRolling):
    """Rolling Correlation"""

    def __init__(self):
        super(Corr, self).__init__("corr")

    def __call__(self, series_left, series_right, N):
        res: pd.Series = super(Corr, self).__call__(series_left, series_right, N)
        res.loc[
            np.isclose(series_left.rolling(N, min_periods=1).std(), 0, atol=2e-05)
            | np.isclose(series_right.rolling(N, min_periods=1).std(), 0, atol=2e-05)
            ] = np.nan
        return res


class Cov(PairRolling):
    """Rolling Covariance"""

    def __init__(self):
        super(Cov, self).__init__("cov")


# Cross Sectional Operators
class CSRank(abc.ABC):
    def __call__(self, series):
        return series.groupby(level=0).rank(pct=True)
