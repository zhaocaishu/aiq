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

    def __init__(self, N, func):
        """
        Args:
            N (int): rolling window size
            func (str): rolling method
        """
        self.N = N
        self.func = func

    def __call__(self, series: pd.Series):
        # NOTE: remove all null check,
        # now it's user's responsibility to decide whether use features in null days
        # isnull = series.isnull() # NOTE: isnull = NaN, inf is not null
        if isinstance(self.N, int) and self.N == 0:
            series = getattr(series.expanding(min_periods=1), self.func)()
        elif isinstance(self.N, float) and 0 < self.N < 1:
            series = series.ewm(alpha=self.N, min_periods=1).mean()
        else:
            series = getattr(series.rolling(self.N, min_periods=1), self.func)()
            # series.iloc[:self.N-1] = np.nan
        # series[isnull] = np.nan
        return series


class Ref(Rolling):
    def __init__(self, N):
        """
        Args:
            N (int):  N = 0, retrieve the first data; N > 0, retrieve data of N periods ago; N < 0, future data
        """
        super(Ref, self).__init__(N, "ref")

    def __call__(self, series: pd.Series):
        # N = 0, return first day
        if series.empty:
            return series  # Pandas bug, see: https://github.com/pandas-dev/pandas/issues/21049
        elif self.N == 0:
            series = pd.Series(series.iloc[0], index=series.index)
        else:
            series = series.shift(self.N)  # copy
        return series


class Mean(Rolling):
    """Rolling Mean"""

    def __init__(self, N):
        super(Mean, self).__init__(N, "mean")

class Sum(Rolling):
    """Rolling Sum"""
    def __init__(self, N):
        super(Sum, self).__init__(N, "sum")

class Std(Rolling):
    """Rolling Std"""

    def __init__(self, N):
        super(Std, self).__init__(N, "std")


class Var(Rolling):
    """Rolling Variance"""

    def __init__(self, N):
        super(Var, self).__init__(N, "var")


class Skew(Rolling):
    """Rolling Skewness"""

    def __init__(self, N):
        super(Skew, self).__init__(N, "skew")


class Kurt(Rolling):
    """Rolling Kurtosis"""

    def __init__(self, N):
        if N != 0 and N < 4:
            raise ValueError("The rolling window size of Kurtosis operation should >= 5")
        super(Kurt, self).__init__(N, "kurt")


class Max(Rolling):
    """Rolling Max"""

    def __init__(self, N):
        super(Max, self).__init__(N, "max")


class IdxMax(Rolling):
    """Rolling Max Index"""

    def __init__(self, N):
        super(IdxMax, self).__init__(N, "idxmax")

    def __call__(self, series: pd.Series):
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        return series


class Min(Rolling):
    """Rolling Min"""

    def __init__(self, N):
        super(Min, self).__init__(N, "min")


class IdxMin(Rolling):
    """Rolling Min Index"""

    def __init__(self, N):
        super(IdxMin, self).__init__(N, "idxmin")

    def __call__(self, series):
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        return series


class Quantile(Rolling):
    """Rolling Quantile"""

    def __init__(self, N, qscore):
        super(Quantile, self).__init__(N, "quantile")
        self.qscore = qscore

    def __call__(self, series: pd.Series):
        if self.N == 0:
            series = series.expanding(min_periods=1).quantile(self.qscore)
        else:
            series = series.rolling(self.N, min_periods=1).quantile(self.qscore)
        return series


class Med(Rolling):
    """Rolling Median"""

    def __init__(self, N):
        super(Med, self).__init__(N, "median")


class Mad(Rolling):
    """Rolling Mean Absolute Deviation"""

    def __init__(self, N):
        super(Mad, self).__init__(N, "mad")

    def __call__(self, series: pd.Series):
        def mad(x):
            x1 = x[~np.isnan(x)]
            return np.mean(np.abs(x1 - x1.mean()))

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(mad, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(mad, raw=True)
        return series


class Rank(Rolling):
    """Rolling Rank (Percentile)"""

    def __init__(self, N):
        super(Rank, self).__init__(N, "rank")

    # for compatiblity of python 3.7, which doesn't support pandas 1.4.0+ which implements Rolling.rank
    def __call__(self, series: pd.Series):
        rolling_or_expending = series.expanding(min_periods=1) if self.N == 0 else series.rolling(self.N, min_periods=1)
        if hasattr(rolling_or_expending, "rank"):
            return rolling_or_expending.rank(pct=True)

        def rank(x):
            if np.isnan(x[-1]):
                return np.nan
            x1 = x[~np.isnan(x)]
            if x1.shape[0] == 0:
                return np.nan
            return percentileofscore(x1, x1[-1]) / 100

        return rolling_or_expending.apply(rank, raw=True)


class Count(Rolling):
    """Rolling Count"""

    def __init__(self, N):
        super(Count, self).__init__(N, "count")


class Delta(Rolling):
    """Rolling Delta"""

    def __init__(self, N):
        super(Delta, self).__init__(N, "delta")

    def __call__(self, series: pd.Series):
        if self.N == 0:
            series = series - series.iloc[0]
        else:
            series = series - series.shift(self.N)
        return series


# TODO:
# support pair-wise rolling like `Slope(A, B, N)`
class Slope(Rolling):
    """Rolling Slope"""

    def __init__(self, N):
        super(Slope, self).__init__(N, "slope")

    def __call__(self, series: pd.Series):
        if self.N == 0:
            series = pd.Series(expanding_slope(series.values), index=series.index)
        else:
            series = pd.Series(rolling_slope(series.values, self.N), index=series.index)
        return series


class Rsquare(Rolling):
    """Rolling R-value Square"""

    def __init__(self, N):
        super(Rsquare, self).__init__(N, "rsquare")

    def __call__(self, _series: pd.Series):
        if self.N == 0:
            series = pd.Series(expanding_rsquare(_series.values), index=_series.index)
        else:
            series = pd.Series(rolling_rsquare(_series.values, self.N), index=_series.index)
            series.loc[np.isclose(_series.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)] = np.nan
        return series


class Resi(Rolling):
    """Rolling Regression Residuals"""

    def __init__(self, N):
        super(Resi, self).__init__(N, "resi")

    def __call__(self, series: pd.Series):
        if self.N == 0:
            series = pd.Series(expanding_resi(series.values), index=series.index)
        else:
            series = pd.Series(rolling_resi(series.values, self.N), index=series.index)
        return series


class WMA(Rolling):
    """Rolling WMA"""

    def __init__(self, N):
        super(WMA, self).__init__(N, "wma")

    def __call__(self, series: pd.Series):
        def weighted_mean(x):
            w = np.arange(len(x)) + 1
            w = w / w.sum()
            return np.nanmean(w * x)

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(weighted_mean, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(weighted_mean, raw=True)
        return series


class EMA(Rolling):
    """Rolling Exponential Mean (EMA)"""

    def __init__(self, N):
        super(EMA, self).__init__(N, "ema")

    def __call__(self, series: pd.Series):
        def exp_weighted_mean(x):
            a = 1 - 2 / (1 + len(x))
            w = a ** np.arange(len(x))[::-1]
            w /= w.sum()
            return np.nansum(w * x)

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(exp_weighted_mean, raw=True)
        elif 0 < self.N < 1:
            series = series.ewm(alpha=self.N, min_periods=1).mean()
        else:
            series = series.ewm(span=self.N, min_periods=1).mean()
        return series


class PairRolling(abc.ABC):
    """Pair Rolling Operator"""

    def __init__(self, N, func):
        self.N = N
        self.func = func

    def __call__(self, series_left: pd.Series, series_right: pd.Series):
        if self.N == 0:
            series = getattr(series_left.expanding(min_periods=1), self.func)(series_right)
        else:
            series = getattr(series_left.rolling(self.N, min_periods=1), self.func)(series_right)
        return series


class Corr(PairRolling):
    """Rolling Correlation"""

    def __init__(self, N):
        super(Corr, self).__init__(N, "corr")

    def __call__(self, series_left, series_right):
        res: pd.Series = super(Corr, self).__call__(series_left, series_right)
        res.loc[
            np.isclose(series_left.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
            | np.isclose(series_right.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
            ] = np.nan
        return res
