import abc
from typing import List

import pandas as pd
import numpy as np

from aiq.ops import (
    Greater,
    Less,
    Ref,
    Mean,
    Std,
    Rsquare,
    Resi,
    Slope,
    Skew,
    Max,
    Min,
    Quantile,
    Rank,
    IdxMax,
    IdxMin,
    Corr,
    Log,
    Sum,
    Abs,
    Cov,
    CSRank,
)
from aiq.utils.module import init_instance_by_config


class DataHandler(abc.ABC):
    def __init__(self, processors: List = []):
        pass

    def process(self, df: pd.DataFrame = None, mode: str = "train") -> pd.DataFrame:
        pass


class Alpha158(DataHandler):
    def __init__(self, processors: List = []):
        self.feature_names_ = None
        self.label_name_ = None
        self.processors = [init_instance_by_config(proc) for proc in processors]

    def extract_feature_labels(self, df: pd.DataFrame = None, mode: str = "train"):
        # adjusted prices
        adjusted_factor = df["Adj_factor"]
        df["Open"] = df["Open"] * adjusted_factor
        df["Close"] = df["Close"] * adjusted_factor
        df["High"] = df["High"] * adjusted_factor
        df["Low"] = df["Low"] * adjusted_factor

        open = df["Open"]
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # kbar
        features = [
            (close - open) / open,
            (high - low) / open,
            (close - open) / (high - low + 1e-12),
            (high - Greater(open, close)) / open,
            (high - Greater(open, close)) / (high - low + 1e-12),
            (Less(open, close) - low) / open,
            (Less(open, close) - low) / (high - low + 1e-12),
            (2 * close - high - low) / open,
            (2 * close - high - low) / (high - low + 1e-12),
        ]
        feature_names = [
            "KMID",
            "KLEN",
            "KMID2",
            "KUP",
            "KUP2",
            "KLOW",
            "KLOW2",
            "KSFT",
            "KSFT2",
        ]

        # price
        for field in ["Open", "High", "Low", "Close"]:
            for d in range(5):
                if d != 0:
                    features.append(Ref(df[field], d) / close)
                else:
                    features.append(df[field] / close)
                feature_names.append(field.upper() + str(d))

        # volume
        for d in range(5):
            if d != 0:
                features.append(Ref(volume, d) / (volume + 1e-12))
            else:
                features.append(volume / (volume + 1e-12))
            feature_names.append("VOLUME%d" % d)

        # rolling
        windows = [5, 10, 20, 30, 60]
        include = None
        exclude = ["VALUE", "ILLIQ", "TURN"]

        def use(x):
            return x not in exclude and (include is None or x in include)

        if use("ROC"):
            # https://www.investopedia.com/terms/r/rateofchange.asp
            # Rate of change, the price change in the past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Ref(close, d) / close)
                feature_names.append("ROC%d" % d)

        if use("MA"):
            # https://www.investopedia.com/ask/answers/071414/whats-difference-between-moving-average-and-weighted-moving-average.asp
            # Simple Moving Average, the simple moving average in the past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Mean(close, d) / close)
                feature_names.append("MA%d" % d)

        if use("STD"):
            # The standard diviation of close price for the past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Std(close, d) / close)
                feature_names.append("STD%d" % d)

        if use("BETA"):
            # The rate of close price change in the past d days, divided by latest close price to remove unit
            # For example, price increase 10 dollar per day in the past d days, then Slope will be 10.
            for d in windows:
                features.append(Slope(close, d) / close)
                feature_names.append("BETA%d" % d)

        if use("RSQR"):
            # The R-sqaure value of linear regression for the past d days, represent the trend linear
            for d in windows:
                features.append(Rsquare(close, d))
                feature_names.append("RSQR%d" % d)

        if use("RESI"):
            # The redisdual for linear regression for the past d days, represent the trend linearity for past d days.
            for d in windows:
                features.append(Resi(close, d) / close)
                feature_names.append("RESI%d" % d)

        if use("MAX"):
            # The max price for past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Max(high, d) / close)
                feature_names.append("MAX%d" % d)

        if use("LOW"):
            # The low price for past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Min(low, d) / close)
                feature_names.append("MIN%d" % d)

        if use("QTLU"):
            # The 80% quantile of past d day's close price, divided by latest close price to remove unit
            # Used with MIN and MAX
            for d in windows:
                features.append(Quantile(close, d, 0.8) / close)
                feature_names.append("QTLU%d" % d)

        if use("QTLD"):
            # The 20% quantile of past d day's close price, divided by latest close price to remove unit
            for d in windows:
                features.append(Quantile(close, d, 0.2) / close)
                feature_names.append("QTLD%d" % d)

        if use("RANK"):
            # Get the percentile of current close price in past d day's close price.
            # Represent the current price level comparing to past N days, add additional information to moving average.
            for d in windows:
                features.append(Rank(close, d))
                feature_names.append("RANK%d" % d)

        if use("RSV"):
            # Represent the price position between upper and lower resistent price for past d days.
            for d in windows:
                features.append(
                    (close - Min(low, d)) / (Max(high, d) - Min(low, d) + 1e-12)
                )
                feature_names.append("RSV%d" % d)

        if use("IMAX"):
            # The number of days between current date and previous highest price date.
            # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
            # The indicator measures the time between highs and the time between lows over a time period.
            # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
            for d in windows:
                features.append(IdxMax(high, d) / d)
                feature_names.append("IMAX%d" % d)

        if use("IMIN"):
            # The number of days between current date and previous lowest price date.
            # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
            # The indicator measures the time between highs and the time between lows over a time period.
            # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
            for d in windows:
                features.append(IdxMin(low, d) / d)
                feature_names.append("IMIN%d" % d)

        if use("IMXD"):
            # The time period between previous lowest-price date occur after highest price date.
            # Large value suggest downward momemtum.
            for d in windows:
                features.append((IdxMax(high, d) - IdxMin(low, d)) / d)
                feature_names.append("IMXD%d" % d)

        if use("CORR"):
            # The correlation between absolute close price and log scaled trading volume
            for d in windows:
                features.append(Corr(close, Log(volume + 1), d))
                feature_names.append("CORR%d" % d)

        if use("CORD"):
            # The correlation between price change ratio and volume change ratio
            for d in windows:
                features.append(
                    Corr(close / Ref(close, 1), Log(volume / Ref(volume, 1) + 1), d)
                )
                feature_names.append("CORD%d" % d)

        if use("CNTP"):
            # The percentage of days in past d days that price go up.
            for d in windows:
                features.append(Mean(close > Ref(close, 1), d))
                feature_names.append("CNTP%d" % d)

        if use("CNTN"):
            # The percentage of days in past d days that price go down.
            for d in windows:
                features.append(Mean(close < Ref(close, 1), d))
                feature_names.append("CNTN%d" % d)

        if use("CNTD"):
            # The diff between past up day and past down day
            for d in windows:
                features.append(
                    Mean(close > Ref(close, 1), d) - Mean(close < Ref(close, 1), d)
                )
                feature_names.append("CNTD%d" % d)

        if use("SUMP"):
            # The total gain / the absolute total price changed
            # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
            for d in windows:
                features.append(
                    Sum(Greater(close - Ref(close, 1), 0), d)
                    / (Sum(Abs(close - Ref(close, 1)), d) + 1e-12)
                )
                feature_names.append("SUMP%d" % d)

        if use("SUMN"):
            # The total lose / the absolute total price changed
            # Can be derived from SUMP by SUMN = 1 - SUMP
            # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
            for d in windows:
                features.append(
                    Sum(Greater(Ref(close, 1) - close, 0), d)
                    / (Sum(Abs(close - Ref(close, 1)), d) + 1e-12)
                )
                feature_names.append("SUMN%d" % d)

        if use("SUMD"):
            # The diff ratio between total gain and total lose
            # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
            for d in windows:
                features.append(
                    (
                        Sum(Greater(close - Ref(close, 1), 0), d)
                        - Sum(Greater(Ref(close, 1) - close, 0), d)
                    )
                    / (Sum(Abs(close - Ref(close, 1)), d) + 1e-12)
                )
                feature_names.append("SUMD%d" % d)

        if use("VMA"):
            # Simple Volume Moving average: https://www.barchart.com/education/technical-indicators/volume_moving_average
            for d in windows:
                features.append(Mean(volume, d) / (volume + 1e-12))
                feature_names.append("VMA%d" % d)

        if use("VSTD"):
            # The standard deviation for volume in past d days.
            for d in windows:
                features.append(Std(volume, d) / (volume + 1e-12))
                feature_names.append("VSTD%d" % d)

        if use("WVMA"):
            # The volume weighted price change volatility
            for d in windows:
                features.append(
                    Std(Abs(close / Ref(close, 1) - 1) * volume, d)
                    / (Mean(Abs(close / Ref(close, 1) - 1) * volume, d) + 1e-12)
                )
                feature_names.append("WVMA%d" % d)

        if use("VSUMP"):
            # The total volume increase / the absolute total volume changed
            for d in windows:
                features.append(
                    Sum(Greater(volume - Ref(volume, 1), 0), d)
                    / (Sum(Abs(volume - Ref(volume, 1)), d) + 1e-12)
                )
                feature_names.append("VSUMP%d" % d)

        if use("VSUMN"):
            # The total volume increase / the absolute total volume changed
            for d in windows:
                features.append(
                    Sum(Greater(Ref(volume, 1) - volume, 0), d)
                    / (Sum(Abs(volume - Ref(volume, 1)), d) + 1e-12)
                )
                feature_names.append("VSUMN%d" % d)

        if use("VSUMD"):
            # The diff ratio between total volume increase and total volume decrease
            # RSI indicator for volume
            for d in windows:
                features.append(
                    (
                        Sum(Greater(volume - Ref(volume, 1), 0), d)
                        - Sum(Greater(Ref(volume, 1) - volume, 0), d)
                    )
                    / (Sum(Abs(volume - Ref(volume, 1)), d) + 1e-12)
                )
                feature_names.append("VSUMD%d" % d)

        # concat features
        self.feature_names_ = feature_names.copy()
        features_df = pd.concat(
            [
                df[["Date", "Instrument"]],
                pd.concat(
                    [
                        features[i].rename(feature_names[i])
                        for i in range(len(feature_names))
                    ],
                    axis=1,
                ).astype("float32"),
            ],
            axis=1,
        )

        # labels
        if mode in ["train", "valid"]:
            self.label_name_ = "LABEL"
            label = (Ref(close, -5) / Ref(close, -1) - 1).rename(self.label_name_)
        else:
            self.label_name_ = None
            label = None

        # if there is a label, concatenate it with the features
        if label is not None:
            features_labels = pd.concat([features_df, label], axis=1)
        else:
            features_labels = features_df

        return features_labels

    def process(
        self, dfs: List[pd.DataFrame] = [], mode: str = "train"
    ) -> pd.DataFrame:
        # extract feature from data
        feature_labels = [self.extract_feature_labels(df, mode) for df in dfs]

        # concat features and set multi-index
        df = pd.concat(feature_labels, ignore_index=True).set_index(
            ["Date", "Instrument"]
        )
        df.sort_index(inplace=True)

        # data preprocessor
        column_tuples = [
            ("feature", feature_name) for feature_name in self.feature_names_
        ]
        if self.label_name_:
            column_tuples.append(("label", self.label_name_))
        df.columns = pd.MultiIndex.from_tuples(column_tuples)

        if mode == "train":
            for processor in self.processors:
                processor.fit(df)
                df = processor(df)
        else:
            for processor in self.processors:
                df = processor(df)

        df.columns = df.columns.droplevel()

        return df

    @property
    def feature_names(self):
        return self.feature_names_

    @property
    def label_name(self):
        return self.label_name_
