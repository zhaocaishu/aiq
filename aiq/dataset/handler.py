import abc

import numpy as np
import pandas as pd
import talib as ta

from aiq.ops import Greater, Less, Ref, Mean, Std, Rsquare, Resi, Slope, Skew, Max, Min, Quantile, Rank, IdxMax, IdxMin, \
    Corr, Log, Sum, Abs


class DataHandler(abc.ABC):
    def __init__(self):
        pass

    def fetch(self, df: pd.DataFrame = None) -> pd.DataFrame:
        pass


class Alpha158(DataHandler):
    def __init__(self, test_mode=False):
        self.test_mode = test_mode

        self._feature_names = None
        self._label_name = None

    def fetch(self, df: pd.DataFrame = None) -> pd.DataFrame:
        open = df['Open']
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # kbar
        features = [(close - open) / open,
                    (high - low) / open,
                    (close - open) / (high - low + 1e-12),
                    (high - Greater()(open, close)) / open,
                    (high - Greater()(open, close)) / (high - low + 1e-12),
                    (Less()(open, close) - low) / open,
                    (Less()(open, close) - low) / (high - low + 1e-12),
                    (2 * close - high - low) / open,
                    (2 * close - high - low) / (high - low + 1e-12)]
        names = ['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2']

        # price
        for field in ['Open', 'High', 'Low', 'Close']:
            for d in range(5):
                if d != 0:
                    features.append(Ref(d)(df[field]) / close)
                else:
                    features.append(df[field] / close)
                names.append(field.upper() + str(d))

        # volume
        for d in range(5):
            field = 'Volume'
            feature = field.upper() + str(d)
            if d != 0:
                features.append(Ref(d)(df[field]) / (volume + 1e-12))
            else:
                features.append(df[field] / (volume + 1e-12))
            names.append(feature)

        # rolling
        windows = [5, 10, 20, 30, 60]
        include = None
        exclude = []

        def use(x):
            return x not in exclude and (include is None or x in include)

        if use("ROC"):
            # https://www.investopedia.com/terms/r/rateofchange.asp
            # Rate of change, the price change in the past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Ref(d)(close) / close)
                names.append('ROC%d' % d)

        if use("MA"):
            # https://www.investopedia.com/ask/answers/071414/whats-difference-between-moving-average-and-weighted-moving-average.asp
            # Simple Moving Average, the simple moving average in the past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Mean(d)(close) / close)
                names.append('MA%d' % d)

        if use("STD"):
            # The standard diviation of close price for the past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Std(d)(close) / close)
                names.append('STD%d' % d)

        if use("BETA"):
            # The rate of close price change in the past d days, divided by latest close price to remove unit
            # For example, price increase 10 dollar per day in the past d days, then Slope will be 10.
            for d in windows:
                features.append(Slope(d)(close) / close)
                names.append('BETA%d' % d)

        if use("RSQR"):
            # The R-sqaure value of linear regression for the past d days, represent the trend linear
            for d in windows:
                features.append(Rsquare(d)(close))
                names.append('RSQR%d' % d)

        if use("RESI"):
            # The redisdual for linear regression for the past d days, represent the trend linearity for past d days.
            for d in windows:
                features.append(Resi(d)(close) / close)
                names.append('RESI%d' % d)

        if use("MAX"):
            # The max price for past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Max(d)(high) / close)
                names.append('MAX%d' % d)

        if use("LOW"):
            # The low price for past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Min(d)(low) / close)
                names.append('MIN%d' % d)

        if use("QTLU"):
            # The 80% quantile of past d day's close price, divided by latest close price to remove unit
            # Used with MIN and MAX
            for d in windows:
                features.append(Quantile(d, 0.8)(close) / close)
                names.append('QTLU%d' % d)

        if use("QTLD"):
            # The 20% quantile of past d day's close price, divided by latest close price to remove unit
            for d in windows:
                features.append(Quantile(d, 0.2)(close) / close)
                names.append('QTLD%d' % d)

        if use("RANK"):
            # Get the percentile of current close price in past d day's close price.
            # Represent the current price level comparing to past N days, add additional information to moving average.
            for d in windows:
                features.append(Rank(d)(close))
                names.append('RANK%d' % d)

        if use("RSV"):
            # Represent the price position between upper and lower resistent price for past d days.
            for d in windows:
                features.append((close - Min(d)(low)) / (Max(d)(high) - Min(d)(low) + 1e-12))
                names.append('RSV%d' % d)

        if use("IMAX"):
            # The number of days between current date and previous highest price date.
            # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
            # The indicator measures the time between highs and the time between lows over a time period.
            # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
            for d in windows:
                features.append(IdxMax(d)(high) / d)
                names.append('IMAX%d' % d)

        if use("IMIN"):
            # The number of days between current date and previous lowest price date.
            # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
            # The indicator measures the time between highs and the time between lows over a time period.
            # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
            for d in windows:
                features.append(IdxMin(d)(low) / d)
                names.append('IMIN%d' % d)

        if use("IMXD"):
            # The time period between previous lowest-price date occur after highest price date.
            # Large value suggest downward momemtum.
            for d in windows:
                features.append((IdxMax(d)(high) - IdxMin(d)(low)) / d)
                names.append('IMXD%d' % d)

        if use("CORR"):
            # The correlation between absolute close price and log scaled trading volume
            for d in windows:
                features.append(Corr(d)(close, Log()(volume + 1)))
                names.append('CORR%d' % d)

        if use("CORD"):
            # The correlation between price change ratio and volume change ratio
            for d in windows:
                features.append(Corr(d)(close / Ref(1)(close), Log()(volume / Ref(1)(volume) + 1)))
                names.append('CORD%d' % d)

        if use("CNTP"):
            # The percentage of days in past d days that price go up.
            for d in windows:
                features.append(Mean(d)(close > Ref(1)(close)))
                names.append('CNTP%d' % d)

        if use("CNTN"):
            # The percentage of days in past d days that price go down.
            for d in windows:
                features.append(Mean(d)(close < Ref(1)(close)))
                names.append('CNTN%d' % d)

        if use("CNTD"):
            # The diff between past up day and past down day
            for d in windows:
                features.append(Mean(d)(close > Ref(1)(close)) - Mean(d)(close < Ref(1)(close)))
                names.append('CNTD%d' % d)

        if use("SUMP"):
            # The total gain / the absolute total price changed
            # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
            for d in windows:
                features.append(
                    Sum(d)(Greater()(close - Ref(1)(close), 0)) / (Sum(d)(Abs()(close - Ref(1)(close))) + 1e-12))
                names.append('SUMP%d' % d)

        if use("SUMN"):
            # The total lose / the absolute total price changed
            # Can be derived from SUMP by SUMN = 1 - SUMP
            # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
            for d in windows:
                features.append(
                    Sum(d)(Greater()(Ref(1)(close) - close, 0)) / (Sum(d)(Abs()(close - Ref(1)(close))) + 1e-12))
                names.append('SUMN%d' % d)

        if use("SUMD"):
            # The diff ratio between total gain and total lose
            # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
            for d in windows:
                features.append(
                    (Sum(d)(Greater()(close - Ref(1)(close), 0)) - Sum(d)(Greater()(Ref(1)(close) - close, 0))) / (
                            Sum(d)(Abs()(close - Ref(1)(close))) + 1e-12))
                names.append('SUMD%d' % d)

        if use("VMA"):
            # Simple Volume Moving average: https://www.barchart.com/education/technical-indicators/volume_moving_average
            for d in windows:
                features.append(Mean(d)(volume) / (volume + 1e-12))
                names.append('VMA%d' % d)

        if use("VSTD"):
            # The standard deviation for volume in past d days.
            for d in windows:
                features.append(Std(d)(volume) / (volume + 1e-12))
                names.append('VSTD%d' % d)

        if use("WVMA"):
            # The volume weighted price change volatility
            for d in windows:
                features.append(Std(d)(Abs()(close / Ref(1)(close) - 1) * volume) / (
                        Mean(d)(Abs()(close / Ref(1)(close) - 1) * volume) + 1e-12))
                names.append('WVMA%d' % d)

        if use("VSUMP"):
            # The total volume increase / the absolute total volume changed
            for d in windows:
                features.append(
                    Sum(d)(Greater()(volume - Ref(1)(volume), 0)) / (Sum(d)(Abs()(volume - Ref(1)(volume))) + 1e-12))
                names.append('VSUMP%d' % d)

        if use("VSUMN"):
            # The total volume increase / the absolute total volume changed
            for d in windows:
                features.append(
                    Sum(d)(Greater()(Ref(1)(volume) - volume, 0)) / (Sum(d)(Abs()(volume - Ref(1)(volume))) + 1e-12))
                names.append('VSUMN%d' % d)

        if use("VSUMD"):
            # The diff ratio between total volume increase and total volume decrease
            # RSI indicator for volume
            for d in windows:
                features.append(
                    (Sum(d)(Greater()(volume - Ref(1)(volume), 0)) - Sum(d)(Greater()(Ref(1)(volume) - volume, 0))) / (
                                Sum(d)(Abs()(volume - Ref(1)(volume))) + 1e-12))
                names.append('VSUMD%d' % d)

        self._feature_names = names
        self._label_name = 'LABEL'
        df = pd.concat([df, pd.concat([features[i].rename(names[i]) for i in range(len(names))], axis=1)])

        if not self.test_mode:
            # regression target
            df[self._label_name] = Ref(-2)(close) / Ref(-1)(close) - 1
            df = df.dropna(subset=[self._label_name])

        return df

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def label_name(self):
        return self._label_name
