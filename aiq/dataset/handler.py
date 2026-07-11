from typing import List, Union
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
    EMA,
)
from aiq.utils.module import init_instance_by_config

from .loader import DataLoader
from .processor import Processor


class DataHandler:
    def __init__(
        self,
        data_dir: str,
        instruments: Union[str, List[str]],
        start_time: str = "",
        end_time: str = "",
        fit_start_time: str = "",
        fit_end_time: str = "",
        processors: List[Processor] = [],
        label_price: str = "close",
    ):
        self.data_dir = data_dir
        if isinstance(instruments, str):
            df = DataLoader.load_instruments(
                self.data_dir, instruments, start_time, end_time
            )
            self.instruments = df["Instrument"].unique().tolist()
        else:
            self.instruments = instruments
        self.calendar = DataLoader.load_calendar(
            self.data_dir,
            start_time=start_time,
            end_time=end_time,
        )
        self.start_time = start_time
        self.end_time = end_time
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.processors = [init_instance_by_config(proc) for proc in processors]
        self.label_price = label_price

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
        data_dir: str,
        instruments: Union[str, List[str]],
        start_time: str = "",
        end_time: str = "",
        fit_start_time: str = "",
        fit_end_time: str = "",
        processors: List[Processor] = [],
        label_price: str = "close",
    ):
        super().__init__(
            data_dir,
            instruments,
            start_time,
            end_time,
            fit_start_time,
            fit_end_time,
            processors,
            label_price,
        )
        self.feature_names = []
        self.label_names = ["RET_5D"]

    def _get_calendar_index(self) -> pd.DatetimeIndex:
        """Return a normalized, unique and sorted trading calendar."""
        calendar = self.calendar

        if isinstance(calendar, pd.DataFrame):
            if "Date" not in calendar.columns:
                raise ValueError("Trading calendar DataFrame must contain 'Date'.")
            calendar = calendar["Date"]

        calendar_index = pd.DatetimeIndex(pd.to_datetime(calendar, errors="coerce"))
        calendar_index = calendar_index[~calendar_index.isna()]
        calendar_index = calendar_index.normalize().unique().sort_values()

        if len(calendar_index) == 0:
            raise ValueError("Trading calendar is empty.")

        return calendar_index

    def _align_instrument_to_calendar(
        self,
        df: pd.DataFrame,
        calendar: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Align one instrument to the market trading calendar.

        A completely missing row between the first available quotation and the
        handler end date is treated as a suspension day. On such days:

        - OHLC and pre-close are set to the latest available close;
        - volume, amount, turnover and money-flow fields are set to zero;
        - adjustment factor, fundamentals and classifications are forward-filled;
        - ``Is_suspended`` is set to 1.

        Existing rows with partially missing fields are not classified as
        suspensions, so ordinary data-quality problems are not silently hidden.
        """
        if df.empty:
            return df

        instrument = df["Instrument"].iloc[0]
        aligned = df.copy()
        aligned["Date"] = pd.to_datetime(
            aligned["Date"], errors="coerce"
        ).dt.normalize()
        aligned = aligned.dropna(subset=["Date"])
        aligned = (
            aligned.sort_values("Date")
            .drop_duplicates(subset=["Date"], keep="last")
            .set_index("Date")
        )

        if aligned.empty:
            return df.iloc[0:0].copy()

        # Do not create observations before the stock has its first valid quote.
        # The dynamic universe filter is responsible for excluding delisted or
        # otherwise ineligible stocks at the sample endpoint.
        first_date = aligned.index.min()
        end_date = aligned.index.max()
        instrument_calendar = calendar[
            (calendar >= first_date) & (calendar <= end_date)
        ]

        original_dates = aligned.index
        aligned = aligned.reindex(instrument_calendar)
        suspension_mask = ~aligned.index.isin(original_dates)

        aligned["Instrument"] = instrument
        aligned["Is_suspended"] = suspension_mask.astype(np.int8)

        # Use the last observable close as the unchanged reference price during
        # suspension. This preserves a fixed market-calendar horizon.
        previous_close = aligned["Close"].ffill()
        for column in ["Open", "High", "Low", "Close", "Pre_Close"]:
            if column in aligned.columns:
                aligned.loc[suspension_mask, column] = previous_close.loc[
                    suspension_mask
                ]

        # A suspension day has no actual transaction.
        zero_fill_columns = [
            "Change",
            "Pct_Chg",
            "Volume",
            "AMount",
            "Turnover_rate",
            "Turnover_rate_f",
            "Volume_ratio",
            "Mfd_inflow_vol_ratio",
            "Mfd_large_amount_ratio",
        ]
        for column in zero_fill_columns:
            if column in aligned.columns:
                aligned.loc[suspension_mask, column] = 0.0

        # Only use information already known at that time. Backward filling is
        # intentionally avoided to prevent future information leakage.
        forward_fill_columns = [
            "Adj_factor",
            "Pe",
            "Pe_ttm",
            "Pb",
            "Ps",
            "Ps_ttm",
            "Dv_ratio",
            "Dv_ttm",
            "Total_share",
            "Float_share",
            "Free_share",
            "Total_mv",
            "Circ_mv",
            "Ind_class_l1",
            "Ind_class_l2",
            "List_date",
        ]
        existing_forward_fill_columns = [
            column for column in forward_fill_columns if column in aligned.columns
        ]
        if existing_forward_fill_columns:
            forward_filled = aligned[existing_forward_fill_columns].ffill()
            aligned.loc[
                suspension_mask,
                existing_forward_fill_columns,
            ] = forward_filled.loc[
                suspension_mask,
                existing_forward_fill_columns,
            ]

        aligned.index.name = "Date"
        aligned = aligned.reset_index()
        aligned["Date"] = aligned["Date"].dt.strftime("%Y-%m-%d")
        return aligned

    def preprocess_instrument_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align all instruments to the same market trading calendar."""
        if df.empty:
            return df

        required_columns = {"Date", "Instrument", "Close"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                "Missing required columns before calendar alignment: "
                f"{sorted(missing_columns)}"
            )

        calendar = self._get_calendar_index()
        aligned_frames = [
            self._align_instrument_to_calendar(group, calendar)
            for _, group in df.groupby("Instrument", sort=False)
        ]
        aligned_frames = [frame for frame in aligned_frames if not frame.empty]

        if not aligned_frames:
            return pd.DataFrame()

        return (
            pd.concat(aligned_frames, ignore_index=True)
            .sort_values(["Instrument", "Date"])
            .reset_index(drop=True)
        )

    def extract_instrument_features(self, df):
        # fundamental data
        ind_class_l1 = df["Ind_class_l1"]
        ind_class_l2 = df["Ind_class_l2"]
        cap = np.log(df["Circ_mv"])
        ep = (1.0 / df["Pe_ttm"].replace(0, np.nan)).fillna(0)
        bp = (1.0 / df["Pb"].replace(0, np.nan)).fillna(0)
        sp = (1.0 / df["Ps_ttm"].replace(0, np.nan)).fillna(0)

        # volume & amount
        volume = df["Volume"] * 100  # 股
        amount = df["AMount"] * 1000  # 元
        vwap = amount / (volume + 1e-12)

        # A suspension has no real VWAP. For feature continuity only, use the
        # unchanged close. Is_suspended remains available while constructing
        # features and labels; endpoint tradability is filtered separately.
        if "Is_suspended" in df.columns:
            vwap = vwap.where(df["Is_suspended"] == 0, df["Close"])

        # adjusted prices
        adj_factor = df["Adj_factor"]
        open = df["Open"] * adj_factor
        close = df["Close"] * adj_factor
        high = df["High"] * adj_factor
        low = df["Low"] * adj_factor
        vwap = vwap * adj_factor

        # turnover rate
        turn = df["Turnover_rate_f"]

        # moneyflow
        mfd_inflow_vol_ratio = df["Mfd_inflow_vol_ratio"]
        mfd_large_amount_ratio = df["Mfd_large_amount_ratio"]

        # kbar
        features = [
            ind_class_l1,
            ind_class_l2,
            cap,
            ep,
            bp,
            sp,
            Log(volume + 1.0),
            (high - low) / open,
            (close - open) / open,
            (close - open) / ((high - low) + 1e-12),
            (high - Greater(open, close)) / open,
            (high - Greater(open, close)) / ((high - low) + 1e-12),
            (Less(open, close) - low) / open,
            (Less(open, close) - low) / ((high - low) + 1e-12),
            (2 * close - high - low) / open,
            (2 * close - high - low) / ((high - low) + 1e-12),
            open / close,
            high / close,
            low / close,
            vwap / close,
            mfd_inflow_vol_ratio,
            mfd_large_amount_ratio,
        ]
        feature_names = [
            "IND_CLS_L1",
            "IND_CLS_L2",
            "FUND_CAP",
            "FUND_EP",
            "FUND_BP",
            "FUND_SP",
            "TS_VOLUME",
            "TS_KLEN",
            "TS_KMID1",
            "TS_KMID2",
            "TS_KUP1",
            "TS_KUP2",
            "TS_KLOW1",
            "TS_KLOW2",
            "TS_KSFT1",
            "TS_KSFT2",
            "TS_OPEN0",
            "TS_HIGH0",
            "TS_LOW0",
            "TS_VWAP0",
            "TS_MFD_INFLOW_VOL_RATIO",
            "TS_MFD_LARGE_AMT_RATIO",
        ]

        # rolling
        windows = [5, 10, 20, 30, 60]
        include = None
        exclude = ["CS_SUMN", "CS_SUMD", "CS_CNTN", "CS_CNTD", "CS_VSUMN", "CS_VSUMD"]

        def use(x):
            return x not in exclude and (include is None or x in include)

        if use("CS_ROC"):
            # https://www.investopedia.com/terms/r/rateofchange.asp
            # Rate of change, the price change in the past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Ref(close, d) / close)
                feature_names.append("CS_ROC%d" % d)

        if use("CS_MA"):
            # https://www.investopedia.com/ask/answers/071414/whats-difference-between-moving-average-and-weighted-moving-average.asp
            # Simple Moving Average, the simple moving average in the past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Mean(close, d) / close)
                feature_names.append("CS_MA%d" % d)

        if use("CS_STD"):
            # The standard diviation of close price for the past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Std(close, d) / close)
                feature_names.append("CS_STD%d" % d)

        if use("CS_BETA"):
            # The rate of close price change in the past d days, divided by latest close price to remove unit
            # For example, price increase 10 dollar per day in the past d days, then Slope will be 10.
            for d in windows:
                features.append(Slope(close, d) / close)
                feature_names.append("CS_BETA%d" % d)

        if use("CS_RESI"):
            # The redisdual for linear regression for the past d days, represent the trend linearity for past d days.
            for d in windows:
                features.append(Resi(close, d) / close)
                feature_names.append("CS_RESI%d" % d)

        if use("CS_RSQR"):
            # The R-sqaure value of linear regression for the past d days, represent the trend linear
            for d in windows:
                features.append(Rsquare(close, d))
                feature_names.append("CS_RSQR%d" % d)

        if use("CS_MAX"):
            # The max price for past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Max(high, d) / close)
                feature_names.append("CS_MAX%d" % d)

        if use("CS_MIN"):
            # The low price for past d days, divided by latest close price to remove unit
            for d in windows:
                features.append(Min(low, d) / close)
                feature_names.append("CS_MIN%d" % d)

        if use("CS_QTLU"):
            # The 80% quantile of past d day's close price, divided by latest close price to remove unit
            # Used with MIN and MAX
            for d in windows:
                features.append(Quantile(close, d, 0.8) / close)
                feature_names.append("CS_QTLU%d" % d)

        if use("CS_QTLD"):
            # The 20% quantile of past d day's close price, divided by latest close price to remove unit
            for d in windows:
                features.append(Quantile(close, d, 0.2) / close)
                feature_names.append("CS_QTLD%d" % d)

        if use("CS_RANK"):
            # Get the percentile of current close price in past d day's close price.
            # Represent the current price level comparing to past N days, add additional information to moving average.
            for d in windows:
                features.append(Rank(close, d))
                feature_names.append("CS_RANK%d" % d)

        if use("CS_RSV"):
            # Represent the price position between upper and lower resistent price for past d days.
            for d in windows:
                features.append(
                    (close - Min(low, d)) / (Max(high, d) - Min(low, d) + 1e-12)
                )
                feature_names.append("CS_RSV%d" % d)

        if use("CS_IMAX"):
            # The number of days between current date and previous highest price date.
            # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
            # The indicator measures the time between highs and the time between lows over a time period.
            # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
            for d in windows:
                features.append(IdxMax(high, d) / d)
                feature_names.append("CS_IMAX%d" % d)

        if use("CS_IMIN"):
            # The number of days between current date and previous lowest price date.
            # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
            # The indicator measures the time between highs and the time between lows over a time period.
            # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
            for d in windows:
                features.append(IdxMin(low, d) / d)
                feature_names.append("CS_IMIN%d" % d)

        if use("CS_IMXD"):
            # The time period between previous lowest-price date occur after highest price date.
            # Large value suggest downward momemtum.
            for d in windows:
                features.append((IdxMax(high, d) - IdxMin(low, d)) / d)
                feature_names.append("CS_IMXD%d" % d)

        if use("CS_CORR"):
            # The correlation between absolute close price and log scaled trading volume
            for d in windows:
                features.append(Corr(close, Log(volume + 1), d))
                feature_names.append("CS_CORR%d" % d)

        if use("CS_CORD"):
            # The correlation between price change ratio and volume change ratio
            for d in windows:
                features.append(
                    Corr(close / Ref(close, 1), Log(volume / Ref(volume, 1) + 1), d)
                )
                feature_names.append("CS_CORD%d" % d)

        if use("CS_CNTP"):
            # The percentage of days in past d days that price go up.
            for d in windows:
                features.append(Mean(close > Ref(close, 1), d))
                feature_names.append("CS_CNTP%d" % d)

        if use("CS_CNTN"):
            # The percentage of days in past d days that price go down.
            for d in windows:
                features.append(Mean(close < Ref(close, 1), d))
                feature_names.append("CS_CNTN%d" % d)

        if use("CS_CNTD"):
            # The diff between past up day and past down day
            for d in windows:
                features.append(
                    Mean(close > Ref(close, 1), d) - Mean(close < Ref(close, 1), d)
                )
                feature_names.append("CS_CNTD%d" % d)

        if use("CS_SUMP"):
            # The total gain / the absolute total price changed
            # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
            for d in windows:
                features.append(
                    Sum(Greater(close - Ref(close, 1), 0), d)
                    / (Sum(Abs(close - Ref(close, 1)), d) + 1e-12)
                )
                feature_names.append("CS_SUMP%d" % d)

        if use("CS_SUMN"):
            # The total lose / the absolute total price changed
            # Can be derived from SUMP by SUMN = 1 - SUMP
            # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
            for d in windows:
                features.append(
                    Sum(Greater(Ref(close, 1) - close, 0), d)
                    / (Sum(Abs(close - Ref(close, 1)), d) + 1e-12)
                )
                feature_names.append("CS_SUMN%d" % d)

        if use("CS_SUMD"):
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
                feature_names.append("CS_SUMD%d" % d)

        if use("CS_VMA"):
            # Simple Volume Moving average: https://www.barchart.com/education/technical-indicators/volume_moving_average
            for d in windows:
                features.append(Mean(volume, d) / (volume + 1e-12))
                feature_names.append("CS_VMA%d" % d)

        if use("CS_VSTD"):
            # The standard deviation for volume in past d days.
            for d in windows:
                features.append(Std(volume, d) / (volume + 1e-12))
                feature_names.append("CS_VSTD%d" % d)

        if use("CS_WVMA"):
            # The volume weighted price change volatility
            for d in windows:
                features.append(
                    Std(Abs(close / Ref(close, 1) - 1) * volume, d)
                    / (Mean(Abs(close / Ref(close, 1) - 1) * volume, d) + 1e-12)
                )
                feature_names.append("CS_WVMA%d" % d)

        if use("CS_VSUMP"):
            # The total volume increase / the absolute total volume changed
            for d in windows:
                features.append(
                    Sum(Greater(volume - Ref(volume, 1), 0), d)
                    / (Sum(Abs(volume - Ref(volume, 1)), d) + 1e-12)
                )
                feature_names.append("CS_VSUMP%d" % d)

        if use("CS_VSUMN"):
            # The total volume increase / the absolute total volume changed
            for d in windows:
                features.append(
                    Sum(Greater(Ref(volume, 1) - volume, 0), d)
                    / (Sum(Abs(volume - Ref(volume, 1)), d) + 1e-12)
                )
                feature_names.append("CS_VSUMN%d" % d)

        if use("CS_VSUMD"):
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
                feature_names.append("CS_VSUMD%d" % d)

        if use("CS_TURN_MA"):
            for d in windows:
                features.append(EMA(turn, d))
                feature_names.append("CS_TURN_MA%d" % d)

        if use("CS_TURN_STD"):
            for d in windows:
                features.append(Std(turn, d))
                feature_names.append("CS_TURN_STD%d" % d)

        # feature names
        self.feature_names = feature_names

        # concat features
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
            min_date = list_date + pd.DateOffset(months=3)
            feature_df = feature_df[pd.to_datetime(feature_df["Date"]) >= min_date]

        return feature_df

    def extract_instrument_labels(self, df):
        adj_factor = df["Adj_factor"]

        if self.label_price == "close":
            price = df["Close"] * adj_factor
        elif self.label_price == "vwap":
            # Volume: 手 -> 股; AMount: 千元 -> 元，与特征侧 TS_VWAP0 保持一致
            volume = df["Volume"] * 100
            amount = df["AMount"] * 1000
            vwap = amount / (volume + 1e-12)
            if "Is_suspended" in df.columns:
                # This synthetic price only keeps the calendar-aligned series
                # continuous. Tradability filters should still exclude samples
                # whose entry day is suspended.
                vwap = vwap.where(df["Is_suspended"] == 0, df["Close"])
            price = vwap * adj_factor
        else:
            raise ValueError("label_price must be one of {'close', 'vwap'}")

        # Forward return from t+1 to t+5
        ret_5d = Ref(price, -5) / Ref(price, -1) - 1
        labels = [ret_5d]

        return df[["Instrument", "Date"]].assign(
            **{
                name: label.astype("float32")
                for name, label in zip(self.label_names, labels)
            }
        )

    def process(
        self,
        df: pd.DataFrame,
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
            # 判断是否在当前模式下启用该处理器
            if mode == "train" or proc.is_for_infer():
                df = proc(df)

        df.columns = df.columns.droplevel()
        return df

    def setup_data(self, mode="train") -> pd.DataFrame:
        # Load data
        df = DataLoader.load_instruments_features(
            self.data_dir, self.instruments, self.start_time, self.end_time
        )

        # Align each stock to the common market calendar before calculating any
        # rolling feature or forward label. Missing whole rows are treated as
        # suspension days and filled according to market semantics.
        df = self.preprocess_instrument_data(df)

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
            feature_names=self.feature_names,
            label_names=self.label_names,
            processors=self.processors,
            mode=mode,
        ).astype("float32")

        return feature_label_df


class MarketAlpha158(Alpha158):
    def __init__(
        self,
        data_dir: str,
        instruments: Union[str, List[str]],
        start_time: str = "",
        end_time: str = "",
        fit_start_time: str = "",
        fit_end_time: str = "",
        processors: List[Processor] = [],
        market_names: List[str] = [],
        market_processors: List[Processor] = [],
        label_price: str = "close",
    ):
        super().__init__(
            data_dir,
            instruments,
            start_time,
            end_time,
            fit_start_time,
            fit_end_time,
            processors,
            label_price,
        )

        self.market_names = market_names
        self.market_processors = [
            init_instance_by_config(proc) for proc in market_processors
        ]

    def extract_market_features(self, df: pd.DataFrame):
        close = df["Close"]
        amount = df["AMount"]

        # Define window sizes and compute features systematically
        returns = close / Ref(close, 1) - 1
        features = [returns]
        feature_names = ["RETURN_1D"]

        windows = [5, 10, 20, 30, 60]
        for window in windows:
            features.extend(
                [
                    Mean(returns, window),
                    Std(returns, window),
                    amount / Mean(amount, window),
                    Std(amount, window) / Mean(amount, window),
                ]
            )
            feature_names.extend(
                [
                    f"RETURN_MEAN_{window}D",
                    f"RETURN_STD_{window}D",
                    f"AMOUNT_MEAN_{window}D",
                    f"AMOUNT_STD_{window}D",
                ]
            )

        # Concat features
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
        feature_df = feature_df.set_index(["Date", "Instrument"]).sort_index()

        return feature_df

    def setup_data(self, mode="train") -> pd.DataFrame:
        # Load instrument-level features and labels
        feature_label_df = super().setup_data(mode=mode)
        feature_label_df = feature_label_df.reset_index()

        # Load raw market data
        market_df = DataLoader.load_markets_features(
            self.data_dir,
            self.market_names,
            self.start_time,
            self.end_time,
        )

        # Extract market-level features
        market_feature_df = pd.concat(
            [
                self.extract_market_features(
                    market_df[market_df["Instrument"] == market_name]
                )
                .add_prefix(f"MKT_{market_name}_")
                .droplevel("Instrument")
                for market_name in self.market_names
            ],
            axis=1,
            join="inner",
        )

        market_feature_names = market_feature_df.columns.tolist()
        self.feature_names.extend(market_feature_names)

        # Process market features (Normalization, etc.)
        market_feature_df = self.process(
            df=market_feature_df,
            feature_names=market_feature_names,
            processors=self.market_processors,
            mode=mode,
        ).astype("float32")
        market_feature_df = market_feature_df.reset_index()

        # Merge instrument features with market features
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

        # Validation
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
