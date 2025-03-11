from typing import List

import pandas as pd

from aiq.ops import (
    Greater,
    Less,
    Ref,
    Mean,
    Std,
    Rsquare,
    Resi,
    Slope,
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
)
from aiq.utils.module import init_instance_by_config


class DataHandler:
    def __init__(self, processors: List = None):
        pass

    def process(self, df: pd.DataFrame = None, mode: str = "train") -> pd.DataFrame:
        pass


class Alpha158(DataHandler):
    def __init__(self, processors: List = None):
        self._feature_names = None
        self._label_names = None
        self.processors = [init_instance_by_config(proc) for proc in processors]

    def extract_features(self, df: pd.DataFrame = None, mode: str = "train"):
        # adjusted prices
        adjusted_factor = df["Adj_factor"]
        open = df["Open"] * adjusted_factor
        close = df["Close"] * adjusted_factor
        high = df["High"] * adjusted_factor
        low = df["Low"] * adjusted_factor
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
        for field, price in zip(
            ["Open", "High", "Low", "Close"], [open, high, low, close]
        ):
            for d in range(5):
                if d != 0:
                    features.append(Ref(price, d) / close)
                else:
                    features.append(price / close)
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
        self._feature_names = feature_names.copy()
        feature_df = pd.concat(
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

        return feature_df

    def extract_labels(self, df: pd.DataFrame = None, mode: str = "train"):
        adjusted_factor = df["Adj_factor"]
        close = df["Close"] * adjusted_factor

        # labels
        if mode in ["train", "valid"]:
            self._label_names = ["RETN_5D"]
            labels = [Ref(close, -5) / Ref(close, -1) - 1]
            label_df = pd.concat(
                [
                    labels[i].rename(self._label_names[i])
                    for i in range(len(self._label_names))
                ],
                axis=1,
            ).astype("float32")
        else:
            self._label_names = None
            label_df = None

        return label_df

    def process_feature_labels(
        self, dfs: List[pd.DataFrame] = None, mode: str = "train"
    ) -> pd.DataFrame:
        # extract feature and label from data
        feature_label_dfs = [
            (
                pd.concat([self.extract_features(df), self.extract_labels(df)], axis=1)
                if (label_df := self.extract_labels(df)) is not None
                else self.extract_features(df)
            )
            for df in dfs
        ]

        feature_label_df = pd.concat(feature_label_dfs, ignore_index=True).set_index(
            ["Date", "Instrument"]
        )
        feature_label_df.sort_index(inplace=True)

        # data preprocessing
        column_tuples = [
            ("feature", feature_name) for feature_name in self._feature_names
        ]
        if self._label_names:
            column_tuples.extend(
                [("label", label_name) for label_name in self._label_names]
            )
        feature_label_df.columns = pd.MultiIndex.from_tuples(column_tuples)

        if mode == "train":
            for processor in self.processors:
                processor.fit(feature_label_df)
                feature_label_df = processor(feature_label_df)
        else:
            for processor in self.processors:
                if processor.is_for_infer():
                    feature_label_df = processor(feature_label_df)

        feature_label_df.columns = feature_label_df.columns.droplevel()
        return feature_label_df

    def process(
        self, dfs: List[pd.DataFrame] = None, mode: str = "train"
    ) -> pd.DataFrame:
        feature_label_df = self.process_feature_labels(dfs, mode=mode)
        return feature_label_df

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def label_names(self):
        return self._label_names


class MarketAlpha158(Alpha158):
    def __init__(
        self,
        processors: List = None,
        market_processors: List = None,
    ):
        self._feature_names = None
        self._label_names = None
        self.processors = [init_instance_by_config(proc) for proc in processors]

        # handle market information
        self._market_feature_names = None
        self.market_processors = [
            init_instance_by_config(proc) for proc in market_processors
        ]

    def extract_market_features(self, df: pd.DataFrame = None):
        # prices
        close = df["Close"]
        amount = df["AMount"]

        # features
        features = [
            close / Ref(close, 1) - 1,
            Mean(close / Ref(close, 1) - 1, 5),
            Std(close / Ref(close, 1) - 1, 5),
            Mean(amount, 5) / amount,
            Std(amount, 5) / amount,
            Mean(amount / Ref(amount, 1) - 1, 10),
            Std(amount / Ref(amount, 1) - 1, 10),
            Mean(amount, 10) / amount,
            Std(amount, 10) / amount,
            Mean(close / Ref(close, 1) - 1, 20),
            Std(close / Ref(close, 1) - 1, 20),
            Mean(amount, 20) / amount,
            Std(amount, 20) / amount,
            Mean(close / Ref(close, 1) - 1, 30),
            Std(close / Ref(close, 1) - 1, 30),
            Mean(amount, 30) / amount,
            Std(amount, 30) / amount,
            Mean(close / Ref(close, 1) - 1, 60),
            Std(close / Ref(close, 1) - 1, 60),
            Mean(amount, 60) / amount,
            Std(amount, 60) / amount,
        ]
        feature_names = [
            "MKT_RETURN_1D",
            "MKT_RETURN_MEAN_5D",
            "MKT_RETURN_STD_5D",
            "MKT_AMOUNT_MEAN_5D",
            "MKT_AMOUNT_STD_5D",
            "MKT_RETURN_MEAN_10D",
            "MKT_RETURN_STD_10D",
            "MKT_AMOUNT_MEAN_10D",
            "MKT_AMOUNT_STD_10D",
            "MKT_RETURN_MEAN_20D",
            "MKT_RETURN_STD_20D",
            "MKT_AMOUNT_MEAN_20D",
            "MKT_AMOUNT_STD_20D",
            "MKT_RETURN_MEAN_30D",
            "MKT_RETURN_STD_30D",
            "MKT_AMOUNT_MEAN_30D",
            "MKT_AMOUNT_STD_30D",
            "MKT_RETURN_MEAN_60D",
            "MKT_RETURN_STD_60D",
            "MKT_AMOUNT_MEAN_60D",
            "MKT_AMOUNT_STD_60D",
        ]

        # concat features
        self._market_feature_names = feature_names.copy()
        feature_df = pd.concat(
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

        return feature_df

    def process_market_features(
        self,
        dfs: List[pd.DataFrame] = None,
        df_names: List[str] = None,
        mode: str = "train",
    ) -> pd.DataFrame:
        feature_dfs = [self.extract_market_features(df) for df in dfs]

        # extract and rename features for different markets, then merge them into a new DataFrame
        feature_df = pd.concat(
            [
                feature_df.rename(
                    columns={
                        feature_name: f"{df_name}_{feature_name}"
                        for feature_name in self._market_feature_names
                    }
                )
                .drop(columns=["Instrument"])
                .set_index("Date")
                for feature_df, df_name in zip(feature_dfs, df_names)
            ],
            axis=1,
            join="inner",
        )

        self._feature_names.extend(feature_df.columns.tolist())

        column_tuples = [
            ("feature", feature_name) for feature_name in feature_df.columns.tolist()
        ]
        feature_df.columns = pd.MultiIndex.from_tuples(column_tuples)

        # data preprocessing
        if mode == "train":
            for processor in self.market_processors:
                processor.fit(feature_df)
                feature_df = processor(feature_df)
        else:
            for processor in self.market_processors:
                if processor.is_for_infer():
                    feature_df = processor(feature_df)

        feature_df.columns = feature_df.columns.droplevel()
        feature_df = feature_df.reset_index()

        return feature_df

    def extract_labels(self, df: pd.DataFrame = None, mode: str = "train"):
        adjusted_factor = df["Adj_factor"]
        close = df["Close"] * adjusted_factor

        # labels
        if mode in ["train", "valid"]:
            self._label_names = ["RETN_1D", "RETN_2D", "RETN_3D", "RETN_4D", "RETN_5D"]
            labels = [
                Ref(close, -1) / close - 1,
                Ref(close, -2) / close - 1,
                Ref(close, -3) / close - 1,
                Ref(close, -4) / close - 1,
                Ref(close, -5) / close - 1,
            ]
            label_df = pd.concat(
                [
                    labels[i].rename(self._label_names[i])
                    for i in range(len(self._label_names))
                ],
                axis=1,
            ).astype("float32")
        else:
            self._label_names = None
            label_df = None

        return label_df

    def process(
        self,
        dfs: List[pd.DataFrame] = None,
        market_dfs: List[pd.DataFrame] = None,
        market_names: List[str] = None,
        mode: str = "train",
    ) -> pd.DataFrame:
        # instrument feature and label
        feature_label_df = self.process_feature_labels(dfs, mode=mode)
        feature_label_df = feature_label_df.reset_index()

        # market information
        market_feature_df = self.process_market_features(
            market_dfs, market_names, mode=mode
        )
        market_feature_df = market_feature_df.reset_index()

        # merge market information
        feature_label_df = pd.merge(
            feature_label_df,
            market_feature_df,
            on="Date",
            how="inner",
        )
        feature_label_df = feature_label_df.set_index(["Date", "Instrument"])
        feature_label_df.sort_index(inplace=True)

        return feature_label_df
