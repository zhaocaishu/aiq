import abc

import numpy as np
import pandas as pd

from aiq.ops import Greater, Less, Ref, Mean, Std, Rsquare, Resi, Slope, Skew, Max, Min, Quantile, Rank, IdxMax, \
    IdxMin, Corr, Log, Sum, Abs, Cov, CSRank


class DataHandler(abc.ABC):
    def __init__(self):
        pass

    def fetch(self, df: pd.DataFrame = None) -> pd.DataFrame:
        pass


class Alpha158(DataHandler):
    def __init__(self, test_mode=False):
        self.test_mode = test_mode

        self.feature_names_ = None
        self.label_name_ = None

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
                    (high - Greater(open, close)) / open,
                    (high - Greater(open, close)) / (high - low + 1e-12),
                    (Less(open, close) - low) / open,
                    (Less(open, close) - low) / (high - low + 1e-12),
                    (2 * close - high - low) / open,
                    (2 * close - high - low) / (high - low + 1e-12)]
        names = ['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2']

        # price
        for field in ['Open', 'High', 'Low', 'Close']:
            for d in range(5):
                if d != 0:
                    features.append(Ref(df[field], d) / close)
                else:
                    features.append(df[field] / close)
                names.append(field.upper() + str(d))

        # volume
        for d in range(5):
            if d != 0:
                features.append(Ref(volume, d) / (volume + 1e-12))
            else:
                features.append(volume / (volume + 1e-12))
            names.append('VOLUME%d' % d)

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
                features.append(Ref(close, d) / close)
                names.append('ROC%d' % d)

        if use("MA"):
            # https://www.investopedia.com/ask/answers/071414/whats-difference-between-moving-average-and-weighted-moving-average.asp
            # Simple Moving Average, the simple moving average in the past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Mean(close, d) / close)
                names.append('MA%d' % d)

        if use("STD"):
            # The standard diviation of close price for the past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Std(close, d) / close)
                names.append('STD%d' % d)

        if use("BETA"):
            # The rate of close price change in the past d days, divided by latest close price to remove unit
            # For example, price increase 10 dollar per day in the past d days, then Slope will be 10.
            for d in windows:
                features.append(Slope(close, d) / close)
                names.append('BETA%d' % d)

        if use("RSQR"):
            # The R-sqaure value of linear regression for the past d days, represent the trend linear
            for d in windows:
                features.append(Rsquare(close, d))
                names.append('RSQR%d' % d)

        if use("RESI"):
            # The redisdual for linear regression for the past d days, represent the trend linearity for past d days.
            for d in windows:
                features.append(Resi(close, d) / close)
                names.append('RESI%d' % d)

        if use("MAX"):
            # The max price for past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Max(high, d) / close)
                names.append('MAX%d' % d)

        if use("LOW"):
            # The low price for past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Min(low, d) / close)
                names.append('MIN%d' % d)

        if use("QTLU"):
            # The 80% quantile of past d day's close price, divided by latest close price to remove unit
            # Used with MIN and MAX
            for d in windows:
                features.append(Quantile(close, d, 0.8) / close)
                names.append('QTLU%d' % d)

        if use("QTLD"):
            # The 20% quantile of past d day's close price, divided by latest close price to remove unit
            for d in windows:
                features.append(Quantile(close, d, 0.2) / close)
                names.append('QTLD%d' % d)

        if use("RANK"):
            # Get the percentile of current close price in past d day's close price.
            # Represent the current price level comparing to past N days, add additional information to moving average.
            for d in windows:
                features.append(Rank(close, d))
                names.append('RANK%d' % d)

        if use("RSV"):
            # Represent the price position between upper and lower resistent price for past d days.
            for d in windows:
                features.append((close - Min(low, d)) / (Max(high, d) - Min(low, d) + 1e-12))
                names.append('RSV%d' % d)

        if use("IMAX"):
            # The number of days between current date and previous highest price date.
            # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
            # The indicator measures the time between highs and the time between lows over a time period.
            # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
            for d in windows:
                features.append(IdxMax(high, d) / d)
                names.append('IMAX%d' % d)

        if use("IMIN"):
            # The number of days between current date and previous lowest price date.
            # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
            # The indicator measures the time between highs and the time between lows over a time period.
            # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
            for d in windows:
                features.append(IdxMin(low, d) / d)
                names.append('IMIN%d' % d)

        if use("IMXD"):
            # The time period between previous lowest-price date occur after highest price date.
            # Large value suggest downward momemtum.
            for d in windows:
                features.append((IdxMax(high, d) - IdxMin(low, d)) / d)
                names.append('IMXD%d' % d)

        if use("CORR"):
            # The correlation between absolute close price and log scaled trading volume
            for d in windows:
                features.append(Corr(close, Log(volume + 1), d))
                names.append('CORR%d' % d)

        if use("CORD"):
            # The correlation between price change ratio and volume change ratio
            for d in windows:
                features.append(Corr(close / Ref(close, 1), Log(volume / Ref(volume, 1) + 1), d))
                names.append('CORD%d' % d)

        if use("CNTP"):
            # The percentage of days in past d days that price go up.
            for d in windows:
                features.append(Mean(close > Ref(close, 1), d))
                names.append('CNTP%d' % d)

        if use("CNTN"):
            # The percentage of days in past d days that price go down.
            for d in windows:
                features.append(Mean(close < Ref(close, 1), d))
                names.append('CNTN%d' % d)

        if use("CNTD"):
            # The diff between past up day and past down day
            for d in windows:
                features.append(Mean(close > Ref(close, 1), d) - Mean(close < Ref(close, 1), d))
                names.append('CNTD%d' % d)

        if use("SUMP"):
            # The total gain / the absolute total price changed
            # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
            for d in windows:
                features.append(
                    Sum(Greater(close - Ref(close, 1), 0), d) / (Sum(Abs(close - Ref(close, 1)), d) + 1e-12))
                names.append('SUMP%d' % d)

        if use("SUMN"):
            # The total lose / the absolute total price changed
            # Can be derived from SUMP by SUMN = 1 - SUMP
            # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
            for d in windows:
                features.append(
                    Sum(Greater(Ref(close, 1) - close, 0), d) / (Sum(Abs(close - Ref(close, 1)), d) + 1e-12))
                names.append('SUMN%d' % d)

        if use("SUMD"):
            # The diff ratio between total gain and total lose
            # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
            for d in windows:
                features.append(
                    (Sum(Greater(close - Ref(close, 1), 0), d) - Sum(Greater(Ref(close, 1) - close, 0), d)) / (
                            Sum(Abs(close - Ref(close, 1)), d) + 1e-12))
                names.append('SUMD%d' % d)

        if use("VMA"):
            # Simple Volume Moving average: https://www.barchart.com/education/technical-indicators/volume_moving_average
            for d in windows:
                features.append(Mean(volume, d) / (volume + 1e-12))
                names.append('VMA%d' % d)

        if use("VSTD"):
            # The standard deviation for volume in past d days.
            for d in windows:
                features.append(Std(volume, d) / (volume + 1e-12))
                names.append('VSTD%d' % d)

        if use("WVMA"):
            # The volume weighted price change volatility
            for d in windows:
                features.append(Std(Abs(close / Ref(close, 1) - 1) * volume, d) / (
                        Mean(Abs(close / Ref(close, 1) - 1) * volume, d) + 1e-12))
                names.append('WVMA%d' % d)

        if use("VSUMP"):
            # The total volume increase / the absolute total volume changed
            for d in windows:
                features.append(
                    Sum(Greater(volume - Ref(volume, 1), 0), d) / (Sum(Abs(volume - Ref(volume, 1)), d) + 1e-12))
                names.append('VSUMP%d' % d)

        if use("VSUMN"):
            # The total volume increase / the absolute total volume changed
            for d in windows:
                features.append(
                    Sum(Greater(Ref(volume, 1) - volume, 0), d) / (Sum(Abs(volume - Ref(volume, 1)), d) + 1e-12))
                names.append('VSUMN%d' % d)

        if use("VSUMD"):
            # The diff ratio between total volume increase and total volume decrease
            # RSI indicator for volume
            for d in windows:
                features.append(
                    (Sum(Greater(volume - Ref(volume, 1), 0), d) - Sum(Greater(Ref(volume, 1) - volume, 0), d)) / (
                            Sum(Abs(volume - Ref(volume, 1)), d) + 1e-12))
                names.append('VSUMD%d' % d)

        if use("VALUE"):
            features.append(1.0 / df['Pb'])
            features.append(1.0 / df['Pe_ttm'])
            features.append(Mean(df['Dv_ttm'], 252))
            names.append('BPLF')
            names.append('EPTTM')
            names.append('DVTTM')

        if use("TURN"):
            features.append(Mean(df['Turnover_rate_f'], 21))
            features.append(Std(df['Turnover_rate_f'], 21))
            features.append(Mean(df['Turnover_rate_f'], 21) / Mean(df['Turnover_rate_f'], 504))
            names.append('TURN1M')
            names.append('STDTURN1M')
            names.append('BIASTURN1M')

        # features
        self.feature_names_ = names.copy()

        # labels
        if not self.test_mode:
            # regression target
            features.append(Ref(close, -2) / Ref(close, -1) - 1)
            self.label_name_ = 'LABEL'
            names.append(self.label_name_)

        # concat all features and labels
        df = pd.concat(
            [df, pd.concat([features[i].rename(names[i]) for i in range(len(names))], axis=1).astype('float32')],
            axis=1)

        # remove nan labels
        if not self.test_mode:
            df = df.dropna(subset=[self.label_name_])

        return df

    @property
    def feature_names(self):
        return self.feature_names_

    @property
    def label_name(self):
        return self.label_name_


class Alpha101(DataHandler):
    def __init__(self, test_mode=False):
        self.test_mode = test_mode

    def fetch(self, df: pd.DataFrame = None) -> pd.DataFrame:
        open = df['Open']
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        df['VWAP_CSRANK'] = CSRank((high + low) / 2.0)
        df['CLOSE_CSRANK'] = CSRank(close)
        df['VOLUME_CSRANK'] = CSRank(volume)
        df['OPEN_CSRANK'] = CSRank(open)
        df['HIGH_CSRANK'] = CSRank(high)

        def ts_func_lv1(x):
            x['OVRANKCORR10'] = -1.0 * Corr(x['OPEN_CSRANK'], x['VOLUME_CSRANK'], 10)
            x['CVRANKCOV5'] = Cov(x['CLOSE_CSRANK'], x['VOLUME_CSRANK'], 5)
            x['HVRANKCORR3'] = Corr(x['HIGH_CSRANK'], x['VOLUME_CSRANK'], 5)
            x['HVRANKCORR5'] = -1.0 * Corr(x['High'], x['VOLUME_CSRANK'], 3)
            x['HVRANKCOV5'] = Cov(x['HIGH_CSRANK'], x['VOLUME_CSRANK'], 5)
            x['WVRANKCORR5'] = Corr(x['HIGH_CSRANK'], x['VOLUME_CSRANK'], 5)
            return x

        df = df.groupby('Symbol', group_keys=False).apply(ts_func_lv1)
        df['CVRANKCOV5'] = -1.0 * CSRank(df['CVRANKCOV5'])
        df['HVRANKCORR3'] = CSRank(df['HVRANKCORR3'])
        df['HVRANKCOV5'] = -1.0 * CSRank(df['HVRANKCOV5'])
        df['WVRANKCORR5'] = CSRank(df['WVRANKCORR5'])

        def ts_func_lv2(x):
            x['HVRANKCORR3'] = -1.0 * Sum(x['HVRANKCORR3'], 3)
            x['WVRANKCORR5'] = -1.0 * Max(x['WVRANKCORR5'], 5)
            x['CHLRANKCORR12'] = (x['Close'] - Min(x['Low'], 12)) / (Max(x['High'], 12) - Min(x['Low'], 12) + 1e-12)
            return x

        df = df.groupby('Symbol', group_keys=False).apply(ts_func_lv2)
        df['CHLRANKCORR12'] = CSRank(df['CHLRANKCORR12'])

        def ts_func_lv3(x):
            x['CHLRANKCORR12'] = -1.0 * Corr(x['CHLRANKCORR12'], x['VOLUME_CSRANK'], 6)
            return x

        df = df.groupby('Symbol', group_keys=False).apply(ts_func_lv3)

        if not self.test_mode:
            df['LABELRANK'] = CSRank(df['LABEL'])

        return df

    @property
    def feature_names(self):
        return ['OVRANKCORR10', 'CVRANKCOV5', 'HVRANKCORR3', 'HVRANKCORR5', 'HVRANKCOV5',
                'WVRANKCORR5', 'CHLRANKCORR12']

    @property
    def label_name(self):
        return 'LABELRANK'
