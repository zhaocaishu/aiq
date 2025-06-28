from typing import List
import pickle

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
    EMA
)
from aiq.utils.module import init_instance_by_config

from .loader import DataLoader
from .processor import Processor


class DataHandler:
    def __init__(
        self,
        data_dir,
        instruments,
        start_time=None,
        end_time=None,
        fit_start_time=None,
        fit_end_time=None,
        processors=None,
    ):
        self.data_dir = data_dir
        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.processors = [init_instance_by_config(proc) for proc in processors]

    def setup_data(self, mode="train") -> pd.DataFrame:
        raise NotImplementedError

    def load(self, filepath: str):
        try:
            with open(filepath, "rb") as f:
                loaded_components = pickle.load(f)
                self.processors = loaded_components.get("processors", [])
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
        except Exception as e:
            print(f"Error loading processing components from {filepath}: {e}")

    def save(self, filepath: str):
        components_to_save = {"processors": self.processors}
        try:
            with open(filepath, "wb") as f:
                pickle.dump(components_to_save, f)
            print(f"Processing components successfully saved to {filepath}")
        except Exception as e:
            print(f"Error saving processing components to {filepath}: {e}")


class Alpha158(DataHandler):
    def __init__(
        self,
        data_dir,
        instruments,
        start_time=None,
        end_time=None,
        fit_start_time=None,
        fit_end_time=None,
        processors=None,
        benchmark=None,
    ):
        super().__init__(
            data_dir,
            instruments,
            start_time,
            end_time,
            fit_start_time,
            fit_end_time,
            processors,
        )
        self._feature_names = None
        self._label_names = None

        # Load benchmark data
        if benchmark is not None:
            self.benchmark_df = DataLoader.load_market_features(
                self.data_dir,
                market=benchmark,
                start_time=self.start_time,
                end_time=self.end_time,
            )
            self.benchmark_df = self.benchmark_df.rename(
                columns={"Close": "Bench_Close"}
            )
        else:
            self.benchmark_df = None

    def extract_instrument_features(self, df):
        # fundamental data
        ind_class_l1 = df["Ind_class_l1"]
        ind_class_l2 = df["Ind_class_l2"]
        mkt_class = df["Mkt_class"]
        ep = 1.0 / df["Pe_ttm"]
        bp = 1.0 / df["Pb"]
        mkt_cap = np.log(df["Circ_mv"])

        # adjusted prices
        adjusted_factor = df["Adj_factor"]
        open = df["Open"] * adjusted_factor
        close = df["Close"] * adjusted_factor
        high = df["High"] * adjusted_factor
        low = df["Low"] * adjusted_factor
        volume = df["Volume"] * adjusted_factor
        turn = df["Turnover_rate_f"]

        # kbar
        features = [
            mkt_class,
            ind_class_l1,
            ind_class_l2,
            mkt_cap,
            ep,
            bp,
            (close - open) / open,
            (high - low) / open,
            (close - open) / ((high - low) + 0.001),
            (high - Greater(open, close)) / open,
            (high - Greater(open, close)) / ((high - low) + 0.001),
            (Less(open, close) - low) / open,
            (Less(open, close) - low) / ((high - low) + 0.001),
            (2 * close - high - low) / open,
            (2 * close - high - low) / ((high - low) + 0.001),
            Log(open / Ref(close, 1)),
        ]
        feature_names = [
            "MKT_CLS",
            "IND_CLS_L1",
            "IND_CLS_L2",
            "MKT_CAP",
            "EP",
            "BP",
            "KMID",
            "KLEN",
            "KMID2",
            "KUP",
            "KUP2",
            "KLOW",
            "KLOW2",
            "KSFT",
            "KSFT2",
            "KOC",
        ]

        # price
        for field, price in zip(["Open", "High", "Low"], [open, high, low]):
            for d in range(1):
                features.append(Ref(price, d) / close)
                feature_names.append(field.upper() + str(d))

        # rolling
        windows = [5, 10, 20, 30, 60]
        include = None
        exclude = ["SUMN", "SUMD", "CNTN", "CNTD", "VSUMN", "VSUMD"]

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
                features.append(EMA(close, d) / close)
                feature_names.append("MA%d" % d)

        if use("STD"):
            # The standard diviation of close price for the past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Std(close, d) / close)
                feature_names.append("STD%d" % d)

        if use("SLOPE"):
            # The rate of close price change in the past d days, divided by latest close price to remove unit
            # For example, price increase 10 dollar per day in the past d days, then Slope will be 10.
            for d in windows:
                features.append(Slope(close, d) / close)
                feature_names.append("SLOPE%d" % d)

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

        if use("MIN"):
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
                features.append((close - Min(low, d)) / (Max(high, d) - Min(low, d)))
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
                    / Sum(Abs(close - Ref(close, 1)), d)
                )
                feature_names.append("SUMP%d" % d)

        if use("SUMN"):
            # The total lose / the absolute total price changed
            # Can be derived from SUMP by SUMN = 1 - SUMP
            # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
            for d in windows:
                features.append(
                    Sum(Greater(Ref(close, 1) - close, 0), d)
                    / Sum(Abs(close - Ref(close, 1)), d)
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
                    / Sum(Abs(close - Ref(close, 1)), d)
                )
                feature_names.append("SUMD%d" % d)

        if use("VMA"):
            # Simple Volume Moving average: https://www.barchart.com/education/technical-indicators/volume_moving_average
            for d in windows:
                features.append(EMA(volume, d) / volume)
                feature_names.append("VMA%d" % d)

        if use("VSTD"):
            # The standard deviation for volume in past d days.
            for d in windows:
                features.append(Std(volume, d) / volume)
                feature_names.append("VSTD%d" % d)

        if use("WVMA"):
            # The volume weighted price change volatility
            for d in windows:
                features.append(
                    Std((close / Ref(close, 1) - 1) * Log(1 + volume), d)
                    / Mean((close / Ref(close, 1) - 1) * Log(1 + volume), d)
                )
                feature_names.append("WVMA%d" % d)

        if use("VSUMP"):
            # The total volume increase / the absolute total volume changed
            for d in windows:
                features.append(
                    Sum(Greater(volume - Ref(volume, 1), 0), d)
                    / Sum(Abs(volume - Ref(volume, 1)), d)
                )
                feature_names.append("VSUMP%d" % d)

        if use("VSUMN"):
            # The total volume increase / the absolute total volume changed
            for d in windows:
                features.append(
                    Sum(Greater(Ref(volume, 1) - volume, 0), d)
                    / Sum(Abs(volume - Ref(volume, 1)), d)
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
                    / Sum(Abs(volume - Ref(volume, 1)), d)
                )
                feature_names.append("VSUMD%d" % d)

        if use("TURN"):
            for d in windows:
                features.append(EMA(turn, d))
                features.append(Std(turn, d))
                feature_names.append("TURN_MEAN_%dD" % d)
                feature_names.append("TURN_STD_%dD" % d)

        # concat features
        self._feature_names = feature_names.copy()
        feature_df = pd.concat(
            [
                df[["Instrument", "Date"]],
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

        # 根据List_date过滤上市前3个月的数据
        if "List_date" in df.columns:
            list_date = pd.to_datetime(df["List_date"].iloc[0])
            min_date = (list_date + pd.DateOffset(months=3)).strftime("%Y-%m-%d")
            feature_df = feature_df[feature_df["Date"] >= min_date]

        return feature_df

    def extract_instrument_labels(self, df):
        self._label_names = ["RETN_5D"]
        if self.benchmark_df is not None:
            merge_df = pd.merge(
                df,
                self.benchmark_df[["Date", "Bench_Close"]],
                on="Date",
                how="inner",
            )

            assert merge_df.shape[0] == df.shape[0]

            adjusted_factor = merge_df["Adj_factor"]
            close = merge_df["Close"] * adjusted_factor
            benchmark_close = merge_df["Bench_Close"]

            labels = [
                (Ref(close, -5) / Ref(close, -1))
                / (Ref(benchmark_close, -5) / Ref(benchmark_close, -1))
                - 1
            ]
        else:
            merge_df = df
            adjusted_factor = merge_df["Adj_factor"]
            close = merge_df["Close"] * adjusted_factor

            labels = [Ref(close, -5) / Ref(close, -1) - 1]

        label_df = pd.concat(
            [
                merge_df[["Instrument", "Date"]],
                pd.concat(
                    [
                        labels[i].rename(self._label_names[i])
                        for i in range(len(self._label_names))
                    ],
                    axis=1,
                ).astype("float32"),
            ],
            axis=1,
        )

        return label_df

    def process(
        self,
        df: pd.DataFrame = None,
        feature_names: List[str] = [],
        label_names: List[str] = [],
        processors: List[Processor] = [],
        mode: str = "train",
    ):
        column_tuples = [("feature", feature_name) for feature_name in feature_names]
        if label_names:
            column_tuples.extend([("label", label_name) for label_name in label_names])
        df.columns = pd.MultiIndex.from_tuples(column_tuples)

        fit_df = df.loc[self.fit_start_time : self.fit_end_time]
        for proc in processors:
            if mode == "train" and hasattr(proc, "fit"):
                proc.fit(fit_df)
            df = proc(df)

        df.columns = df.columns.droplevel()
        return df

    def setup_data(self, mode="train") -> pd.DataFrame:
        # Load data
        df = DataLoader.load_instruments_features(
            self.data_dir, self.instruments, self.start_time, self.end_time
        )

        # Extract feature and label from data
        feature_df = df.groupby("Instrument", group_keys=False).apply(
            self.extract_instrument_features
        )
        label_df = df.groupby("Instrument", group_keys=False).apply(
            lambda group: self.extract_instrument_labels(group)
        )
        feature_label_df = pd.merge(
            feature_df, label_df, on=["Date", "Instrument"], how="inner"
        )
        feature_label_df = feature_label_df.set_index(
            ["Date", "Instrument"]
        ).sort_index()

        # Instrument-level feature processing
        feature_label_df = self.process(
            df=feature_label_df,
            feature_names=self._feature_names,
            label_names=self._label_names,
            processors=self.processors,
            mode=mode,
        )

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
        data_dir,
        instruments,
        start_time=None,
        end_time=None,
        fit_start_time=None,
        fit_end_time=None,
        processors=None,
        market_processors=None,
        benchmark=None,
    ):
        super().__init__(
            data_dir,
            instruments,
            start_time,
            end_time,
            fit_start_time,
            fit_end_time,
            processors,
            benchmark,
        )

        self.market_processors = [
            init_instance_by_config(proc) for proc in market_processors
        ]
        self._market_feature_names = None

    def extract_market_features(self, df: pd.DataFrame = None):
        close = df["Close"]
        amount = df["AMount"]

        # Define window sizes and compute features systematically
        returns = close / Ref(close, 1) - 1
        features = [returns]
        feature_names = ["MKT_RETURN_1D"]

        windows = [5, 10, 20, 30, 60]
        for window in windows:
            features.extend(
                [
                    Mean(returns, window),
                    Std(returns, window),
                    Mean(amount, window) / amount,
                    Std(amount, window) / amount,
                ]
            )
            feature_names.extend(
                [
                    f"MKT_RETURN_MEAN_{window}D",
                    f"MKT_RETURN_STD_{window}D",
                    f"MKT_AMOUNT_MEAN_{window}D",
                    f"MKT_AMOUNT_STD_{window}D",
                ]
            )

        # Concat features
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

    def setup_data(self, mode="train") -> pd.DataFrame:
        # Load market data
        market_df = DataLoader.load_markets_features(
            self.data_dir,
            ["000903.SH", "000300.SH", "000905.SH"],
            self.start_time,
            self.end_time,
        )

        # Instrument-level feature and label extraction
        feature_label_df = super().setup_data(mode=mode)
        feature_label_df = feature_label_df.reset_index()

        # Market-level feature extraction
        market_feature_df = pd.concat(
            [
                self.extract_market_features(
                    market_df[market_df["Instrument"] == market_name]
                )
                .rename(
                    columns={
                        feature_name: f"{market_name}_{feature_name}"
                        for feature_name in self._market_feature_names
                    }
                )
                .drop(columns=["Instrument"])
                .set_index("Date")
                for market_name in market_df["Instrument"].unique()
            ],
            axis=1,
            join="inner",
        )

        self._market_feature_names = market_feature_df.columns.tolist()
        self._feature_names.extend(self._market_feature_names)

        # Market-level feature processing
        market_feature_df = self.process(
            df=market_feature_df,
            feature_names=self._market_feature_names,
            processors=self.market_processors,
            mode=mode,
        )
        market_feature_df = market_feature_df.reset_index()

        # Merge with instrument features
        market_feature_label_df = (
            pd.merge(
                feature_label_df,
                market_feature_df,
                on="Date",
                how="inner",
            )
            .set_index(["Date", "Instrument"])
            .sort_index()
        )

        assert (
            feature_label_df.shape[0] == market_feature_label_df.shape[0]
        ), "Mismatch in row counts after merging."

        return market_feature_label_df

    def load(self, filepath: str):
        try:
            with open(filepath, "rb") as f:
                loaded_components = pickle.load(f)
                self.processors = loaded_components.get("processors", [])
                self.market_processors = loaded_components.get("market_processors", [])
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
        except Exception as e:
            print(f"Error loading processing components from {filepath}: {e}")

    def save(self, filepath: str):
        components_to_save = {
            "processors": self.processors,
            "market_processors": self.market_processors,
        }
        try:
            with open(filepath, "wb") as f:
                pickle.dump(components_to_save, f)
            print(f"Processing components successfully saved to {filepath}")
        except Exception as e:
            print(f"Error saving processing components to {filepath}: {e}")
