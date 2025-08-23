import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from aiq.dataset.loader import DataLoader
from aiq.ops import Ref


class Evaluator:
    """A class for evaluating prediction models with IC, ICIR, and Hit Rate metrics."""

    def __init__(
        self,
        data_dir=None,
        start_time="",
        end_time="",
        benchmark="000905.SH",
        pred_col="PRED",
        label_col="LABEL",
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
        close = df["Close"] * df["Adj_factor"]
        returns = Ref(close, -5) / Ref(close, -1) - 1

        return pd.concat(
            [df[["Instrument", "Date"]], returns.rename("Return")],
            axis=1,
        )

    def _setup_data(self, pred_df):
        # Load available instruments within the date range
        instruments_df = DataLoader.load_instruments(
            self.data_dir, self.benchmark, self.start_time, self.end_time
        )[["Instrument", "Date"]]

        # Load instrument features
        instruments = instruments_df["Instrument"].unique().tolist()
        features_df = DataLoader.load_instruments_features(
            self.data_dir, instruments, self.start_time, self.end_time
        )

        # Calculate returns per instrument and drop NaNs
        returns_df = (
            features_df.groupby("Instrument", group_keys=False)
            .apply(self._extract_instrument_returns)
            .dropna(subset=["Return"])
        )

        # Merge with instruments and predictions
        merged_df = returns_df.merge(
            instruments_df, on=["Instrument", "Date"], how="inner"
        ).merge(pred_df, on=["Instrument", "Date"], how="inner")

        return merged_df

    def _compute_ic(self, group):
        """Calculate Spearman correlation coefficient (IC) for a group."""
        return (
            spearmanr(group[self.pred_col], group[self.label_col])[0]
            if len(group) >= self.min_samples
            else np.nan
        )

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

    def _compute_win_rate(self, group):
        """Calculate Top-K win rate (proportion of stocks with positive returns)."""
        self._validate_columns(group, extra_cols=["Instrument", "Return"])
        if len(group) < self.min_samples:
            return {f"WR@Top{self.top_k}": np.nan}

        top_pred = group.nlargest(self.top_k, self.pred_col)
        positive_returns = (top_pred["Return"] > 0).sum()

        return {f"WR@Top{self.top_k}": positive_returns / self.top_k}

    def _compute_topk_return(self, group):
        """Calculate Top-K portfolio return (average return of top K predicted stocks)."""
        self._validate_columns(group, extra_cols=["Instrument", "Return"])
        if len(group) < self.min_samples:
            return {f"RET@Top{self.top_k}": np.nan}

        top_pred = group.nlargest(self.top_k, self.pred_col)
        avg_return = top_pred["Return"].mean()

        return {f"RET@Top{self.top_k}": avg_return}

    def evaluate(self, df, groupby_col="Date"):
        """Evaluate model performance with IC, ICIR, and Hit Rate metrics."""
        df = self._setup_data(df)

        self._validate_columns(df, extra_cols=[groupby_col, "Instrument", "Return"])

        # Calculate daily IC and ICIR
        daily_ic = df.groupby(groupby_col).apply(self._compute_ic).dropna()
        ic_mean = daily_ic.mean()
        icir = ic_mean / daily_ic.std() if daily_ic.std() != 0 else np.nan

        # Calculate daily hit rates
        daily_hr = pd.DataFrame(
            df.groupby(groupby_col).apply(self._compute_hit_rate).tolist()
        )
        hr_mean = daily_hr.mean().to_dict()

        # Calculate daily win rates
        daily_wr = pd.DataFrame(
            df.groupby(groupby_col).apply(self._compute_win_rate).tolist()
        )
        wr_mean = daily_wr.mean().to_dict()

        # Calculate daily top-k returns
        daily_ret = pd.DataFrame(
            df.groupby(groupby_col).apply(self._compute_topk_return).tolist()
        )
        ret_mean = daily_ret.mean().to_dict()

        # Combine results
        results = {"IC": ic_mean, "ICIR": icir, **hr_mean, **wr_mean, **ret_mean}
        return pd.DataFrame([results]).to_markdown(index=False, floatfmt=".4f")
