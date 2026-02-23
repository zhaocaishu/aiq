import numpy as np
import pandas as pd

from aiq.dataset.loader import DataLoader
from aiq.ops import Ref


class Evaluator:
    """A class for evaluating prediction models with IC, ICIR, Hit Rate and Precision@K metrics."""

    def __init__(
        self,
        data_dir,
        start_time,
        end_time,
        benchmark="000905.SH",
        date_col="Date",
        pred_col="PRED_RET_5D",
        label_col="RET_5D",
        top_k=30,
        min_samples=50,
    ):
        self.data_dir = data_dir
        self.start_time = start_time
        self.end_time = end_time
        self.benchmark = benchmark
        self.date_col = date_col
        self.pred_col = pred_col
        self.label_col = label_col
        self.top_k = top_k
        self.min_samples = min_samples

    def _validate_columns(self, df, required_cols=None):
        """Check if DataFrame contains required columns."""
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

    def _extract_instrument_returns(self, df):
        if "Adj_factor" in df.columns:
            adj_close = df["Close"] * df["Adj_factor"]
        else:
            adj_close = df["Close"]

        ret_1d = Ref(adj_close, -2) / Ref(adj_close, -1) - 1
        ret_5d = Ref(adj_close, -5) / Ref(adj_close, -1) - 1

        return pd.concat(
            [
                df[["Date", "Instrument"]],
                ret_1d.rename("RET_1D"),
                ret_5d.rename("RET_5D"),
            ],
            axis=1,
        )

    def _setup_data(self, pred_df):
        # Load and process instrument returns
        instruments = (
            DataLoader.load_instruments(
                self.data_dir, self.benchmark, self.start_time, self.end_time
            )["Instrument"]
            .unique()
            .tolist()
        )
        instrument_features = DataLoader.load_instruments_features(
            self.data_dir, instruments, self.start_time, self.end_time
        )
        instrument_returns = (
            instrument_features.groupby("Instrument", group_keys=False)
            .apply(self._extract_instrument_returns)
            .dropna(subset=["RET_5D"])
        )

        # Load and process benchmark returns
        benchmark_features = DataLoader.load_markets_features(
            self.data_dir, [self.benchmark], self.start_time, self.end_time
        )
        benchmark_returns = (
            self._extract_instrument_returns(benchmark_features)
            .dropna(subset=["RET_5D"])
            .rename(
                columns={
                    "RET_1D": "BENCH_RET_1D",
                    "RET_5D": "BENCH_RET_5D",
                }
            )[["Date", "BENCH_RET_1D", "BENCH_RET_5D"]]
        )

        # Multi-stage merge to align actual, predicted, and benchmark data
        merged_df = instrument_returns.merge(
            pred_df[["Date", "Instrument", "PRED_RET_5D"]],
            on=["Date", "Instrument"],
            how="inner",
        ).merge(benchmark_returns, on="Date", how="inner")

        return merged_df

    def _compute_ic(self, group):
        """Calculate Spearman correlation coefficient (IC) for a group."""
        if len(group) < self.min_samples:
            return np.nan

        return group[self.pred_col].corr(group[self.label_col], method="spearman")

    def _compute_hit_rate(self, group):
        """Calculate Top-K and Bottom-K hit rates for a group."""
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
        if len(group) < self.min_samples:
            return {f"Precision@{self.top_k}": np.nan}

        # Select the top-K records based on prediction scores
        top_pred = group.nlargest(self.top_k, self.pred_col, keep="first")

        # Compute Precision@K by checking where predicted return beats the benchmark
        true_positives = (top_pred[self.label_col] > top_pred["BENCH_RET_5D"]).sum()
        precision_at_k = true_positives / self.top_k

        return {f"Precision@{self.top_k}": precision_at_k}

    def _compute_portfolio_metrics(self, df, trading_days=252, n_drop=5):
        df = df.sort_values([self.date_col, self.pred_col], ascending=[True, False])

        current_holdings = set()
        daily_excess_rets = []

        for _, daily_data in df.groupby(self.date_col):
            if len(daily_data) < self.top_k:
                continue

            if not current_holdings:
                # Initial entry
                current_holdings = set(daily_data.head(self.top_k)["Instrument"])
            else:
                # Sell: N instruments in holding with lowest predicted scores
                in_hold = daily_data[daily_data["Instrument"].isin(current_holdings)]
                sell_list = in_hold.nsmallest(n_drop, self.pred_col)[
                    "Instrument"
                ].tolist()

                # Buy: N instruments NOT in holding with highest predicted scores
                not_in_hold = daily_data[
                    ~daily_data["Instrument"].isin(current_holdings)
                ]
                buy_list = not_in_hold.head(n_drop)["Instrument"].tolist()

                current_holdings = (current_holdings - set(sell_list)) | set(buy_list)

            # Calculate daily equal-weighted excess return
            port_ret = daily_data[daily_data["Instrument"].isin(current_holdings)][
                "RET_1D"
            ].mean()
            bench_ret = daily_data["BENCH_RET_1D"].iloc[0]
            daily_excess_rets.append(port_ret - bench_ret)

        rets = pd.Series(daily_excess_rets).dropna()
        if rets.empty:
            return {}

        # Risk and Return Analytics
        nav = (1 + rets).cumprod()
        arr = nav.iloc[-1] ** (trading_days / len(rets)) - 1
        vol = rets.std() * np.sqrt(trading_days)
        sharpe = (
            (rets.mean() / rets.std() * np.sqrt(trading_days)) if rets.std() != 0 else 0
        )
        mdd = (nav / nav.cummax() - 1).min()

        return {"ARR": arr, "Sharpe": sharpe, "MaxDrawdown": mdd, "AnnVol": vol}

    def evaluate(self, pred_df):
        """Evaluate model performance with IC, ICIR, and Hit Rate metrics."""
        df = self._setup_data(pred_df)

        self._validate_columns(
            df,
            required_cols=[
                self.date_col,
                self.label_col,
                self.pred_col,
                "Instrument",
                "RET_1D",
                "BENCH_RET_1D",
                "BENCH_RET_5D",
            ],
        )

        # Calculate daily IC and ICIR
        daily_ic = df.groupby(self.date_col).apply(self._compute_ic).dropna()
        ic = daily_ic.mean()
        icir = ic / daily_ic.std() if daily_ic.std() != 0 else np.nan

        # Calculate daily hit rates
        daily_hr = pd.DataFrame(
            df.groupby(self.date_col).apply(self._compute_hit_rate).tolist()
        )
        hr = daily_hr.mean().to_dict()

        # Calculate daily precision@k
        daily_precision_k = pd.DataFrame(
            df.groupby(self.date_col).apply(self._compute_precision_at_k).tolist()
        )
        precision_k = daily_precision_k.mean().to_dict()

        # Calculate portfolio ARR
        portfolio_metrics = self._compute_portfolio_metrics(df)

        # Combine results
        results = {"IC": ic, "ICIR": icir, **hr, **precision_k, **portfolio_metrics}
        return pd.DataFrame([results]).to_markdown(index=False, floatfmt=".4f")
