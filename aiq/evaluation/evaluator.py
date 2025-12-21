import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from aiq.dataset.loader import DataLoader
from aiq.ops import Ref


class Evaluator:
    """A class for evaluating prediction models with IC, ICIR, Hit Rate and Precision@K metrics."""

    def __init__(
        self,
        data_dir=None,
        start_time="",
        end_time="",
        benchmark="000905.SH",
        pred_col="PRED_RET_5D",
        label_col="RET_5D",
        top_k=30,
        min_samples=50,
    ):
        self.data_dir = data_dir
        self.start_time = start_time
        self.end_time = end_time
        self.benchmark = benchmark
        self.pred_col = pred_col
        self.label_col = label_col
        self.top_k = top_k
        self.min_samples = min_samples

    def _validate_columns(self, df, extra_cols=None):
        """Check if DataFrame contains required columns."""
        required_cols = {self.pred_col, self.label_col} | set(extra_cols or [])
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

    def _extract_instrument_returns(self, df):
        close = df["Close"]
        if "Adj_factor" in df.columns:
            close = close * df["Adj_factor"]

        returns = Ref(close, -5) / Ref(close, -1) - 1

        return pd.concat(
            [df[["Instrument", "Date"]], returns.rename("RET_5D")],
            axis=1,
        )

    def _setup_data(self, pred_df):
        # Load instruments
        instruments_df = DataLoader.load_instruments(
            self.data_dir, self.benchmark, self.start_time, self.end_time
        )[["Instrument", "Date"]]

        # Load features
        instruments = instruments_df["Instrument"].unique().tolist()
        features_df = DataLoader.load_instruments_features(
            self.data_dir, instruments, self.start_time, self.end_time
        )

        # Calculate returns per instrument and drop NaNs
        returns_df = (
            features_df.groupby("Instrument", group_keys=False)
            .apply(self._extract_instrument_returns)
            .dropna(subset=["RET_5D"])
        )

        # Merge with instruments and predictions
        merged_df = returns_df.merge(
            instruments_df, on=["Instrument", "Date"], how="inner"
        ).merge(pred_df, on=["Instrument", "Date"], how="inner")

        # Load benchmakr features
        benchmark_features_df = DataLoader.load_instruments_features(
            self.data_dir, [self.benchmark], self.start_time, self.end_time
        )
        benchmark_returns = (
            benchmark_features_df.groupby("Instrument", group_keys=False)
            .apply(self._extract_instrument_returns)
            .dropna(subset=["RET_5D"])
            .rename(columns={"RET_5D": "BENCH_RET_5D"})
        )[["Date", "BENCH_RET_5D"]]

        # Merge with benchmark returns
        merged_df = merged_df.merge(benchmark_returns, on="Date", how="inner")
        merged_df["EXCESS_RET_5D"] = merged_df["RET_5D"] - merged_df["BENCH_RET_5D"]

        return merged_df

    def _compute_ic(self, group):
        """Calculate Spearman correlation coefficient (IC) for a group."""
        if len(group) < self.min_samples:
            return np.nan
        return group[self.pred_col].corr(group[self.label_col], method="spearman")

    def _compute_hit_rate(self, group):
        """Calculate Top-K and Bottom-K hit rates for a group."""
        self._validate_columns(group, extra_cols=["Instrument"])
        if len(group) < self.min_samples:
            return {f"HR@Top{self.top_k}": np.nan, f"HR@Bottom{self.top_k}": np.nan}

        top_pred = set(group.nlargest(self.top_k, self.pred_col)["Instrument"])
        top_label = set(group.nlargest(self.top_k, self.label_col)["Instrument"])
        bottom_pred = set(group.nsmallest(self.top_k, self.pred_col)["Instrument"])
        bottom_label = set(group.nsmallest(self.top_k, self.label_col)["Instrument"])

        return {
            f"HR@Top{self.top_k}": len(top_pred & top_label) / self.top_k,
            f"HR@Bottom{self.top_k}": len(bottom_pred & bottom_label) / self.top_k,
        }

    def _compute_precision_at_k(self, group):
        """Compute Precision@K — proportion of correctly predicted positive samples among top-K predictions."""
        # Validate required columns
        self._validate_columns(group, extra_cols=["Instrument", "EXCESS_RET_5D"])

        # Drop invalid rows
        group = group.dropna(subset=[self.pred_col, "EXCESS_RET_5D"])
        if len(group) < self.min_samples:
            return {f"Precision@{self.top_k}": np.nan}

        # Select top-K predictions
        top_pred = group.nlargest(self.top_k, self.pred_col, keep="all")

        # Compute Precision@K (fraction of true positives)
        precision_at_k = np.mean(top_pred["EXCESS_RET_5D"].to_numpy() > 0)

        # Compute benchmark precision@k
        benchmark_precision_at_k = np.mean(group["BENCH_RET_5D"].to_numpy() > 0)

        return {
            f"Precision@{self.top_k}": precision_at_k,
            f"BenchmarkPrecision@{self.top_k}": benchmark_precision_at_k,
        }

    def _compute_portfolio_arr(
        self, pred_df, trading_days_per_year=252, holding_period=5
    ):
        """
        Calculates and returns only the Annualized Rate of Return (ARR)
        based on the Top N predicted returns daily.
        """

        # Select the Top N stocks for each date based on PRED_RET_5D
        top_stocks = (
            pred_df.sort_values(["Date", self.pred_col], ascending=[True, False])
            .groupby("Date")
            .head(self.top_k)
        )

        # Calculate the daily average excess return of the Top N portfolio
        daily_avg_ret = top_stocks.groupby("Date")["EXCESS_RET_5D"].mean()

        # Drop any NaN values to avoid calculation errors
        daily_avg_ret = daily_avg_ret.dropna()
        n = len(daily_avg_ret)

        if n == 0:
            return 0.0

        # Calculate the total cumulative growth factor over the entire dataset
        total_growth = (1 + daily_avg_ret).prod()

        # Convert total growth to an average periodic growth rate
        geo_mean_periodic_ret = total_growth ** (1 / n) - 1

        # Apply the compounding formula for annualization
        ann_factor = trading_days_per_year / holding_period
        arr = (1 + geo_mean_periodic_ret) ** ann_factor - 1

        return arr

    def evaluate(self, pred_df, groupby_col="Date"):
        """Evaluate model performance with IC, ICIR, and Hit Rate metrics."""
        df = self._setup_data(pred_df)

        self._validate_columns(df, extra_cols=[groupby_col, "Instrument", "RET_5D"])

        # Calculate daily IC and ICIR
        daily_ic = df.groupby(groupby_col).apply(self._compute_ic).dropna()
        ic_mean = daily_ic.mean()
        icir = ic_mean / daily_ic.std() if daily_ic.std() != 0 else np.nan

        # Calculate daily hit rates
        daily_hr = pd.DataFrame(
            df.groupby(groupby_col).apply(self._compute_hit_rate).tolist()
        )
        hr_mean = daily_hr.mean().to_dict()

        # Calculate daily precision@k
        daily_precision_k = pd.DataFrame(
            df.groupby(groupby_col).apply(self._compute_precision_at_k).tolist()
        )
        precision_k_mean = daily_precision_k.mean().to_dict()

        # Calculate portfolio ARR
        portfolio_arr = self._compute_portfolio_arr(df)

        # Combine results
        results = {
            "IC": ic_mean,
            "ICIR": icir,
            **hr_mean,
            **precision_k_mean,
            "ARR": portfolio_arr,
        }
        return pd.DataFrame([results]).to_markdown(index=False, floatfmt=".4f")
