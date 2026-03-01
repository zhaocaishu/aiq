import numpy as np
import pandas as pd

from aiq.dataset.loader import DataLoader
from aiq.ops import Ref


class Evaluator:
    """
    Offline evaluation framework for A-share multi-asset prediction models.

    Metrics:
        - IC / ICIR
        - HitRate@K
        - Precision@K
        - Long-only TopK Portfolio (ARR / Sharpe / MDD / Vol)
    """

    def __init__(
        self,
        data_dir: str,
        start_time: str,
        end_time: str,
        benchmark: str = "000905.SH",
        date_col: str = "Date",
        instrument_col: str = "Instrument",
        pred_col: str = "PRED_RET_5D",
        label_col: str = "RET_5D",
        top_k: int = 30,
    ):
        self.data_dir = data_dir
        self.start_time = start_time
        self.end_time = end_time
        self.benchmark = benchmark

        self.date_col = date_col
        self.instrument_col = instrument_col
        self.pred_col = pred_col
        self.label_col = label_col
        self.top_k = top_k

    def _compute_forward_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute forward 1D / 5D returns."""

        adj_close = (
            df["Close"] * df["Adj_factor"]
            if "Adj_factor" in df.columns
            else df["Close"]
        )

        ret_1d = Ref(adj_close, -2) / Ref(adj_close, -1) - 1
        ret_5d = Ref(adj_close, -5) / Ref(adj_close, -1) - 1

        return pd.DataFrame(
            {
                self.date_col: df[self.date_col],
                self.instrument_col: df[self.instrument_col],
                self.label_col: ret_5d,
                "RET_1D": ret_1d,
            }
        )

    def _prepare_dataset(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Align prediction / label / benchmark returns."""

        # Load instruments
        instruments = (
            DataLoader.load_instruments(
                self.data_dir, self.benchmark, self.start_time, self.end_time
            )[self.instrument_col]
            .unique()
            .tolist()
        )

        inst_features = DataLoader.load_instruments_features(
            self.data_dir, instruments, self.start_time, self.end_time
        )

        inst_ret = (
            inst_features.groupby(self.instrument_col, group_keys=False)
            .apply(self._compute_forward_returns)
            .dropna()
        )

        # Benchmark
        bench_features = DataLoader.load_markets_features(
            self.data_dir, [self.benchmark], self.start_time, self.end_time
        )

        bench_ret = self._compute_forward_returns(bench_features).rename(
            columns={
                self.label_col: "BENCH_RET_5D",
            }
        )[[self.date_col, "BENCH_RET_5D"]]

        df = (
            inst_ret.merge(
                pred_df[[self.date_col, self.instrument_col, self.pred_col]],
                on=[self.date_col, self.instrument_col],
                how="inner",
            )
            .merge(bench_ret, on=self.date_col, how="inner")
            .dropna()
        )

        return df

    def _compute_daily_ic(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby(self.date_col).apply(
            lambda x: x[self.pred_col].corr(x[self.label_col], method="spearman")
        )

    def _compute_hit_rate(self, group: pd.DataFrame) -> dict:
        k = self.top_k

        top_pred = set(group.nlargest(k, self.pred_col)[self.instrument_col])
        top_true = set(group.nlargest(k, self.label_col)[self.instrument_col])

        bottom_pred = set(group.nsmallest(k, self.pred_col)[self.instrument_col])
        bottom_true = set(group.nsmallest(k, self.label_col)[self.instrument_col])

        return {
            f"HR@Top{k}": len(top_pred & top_true) / k,
            f"HR@Bottom{k}": len(bottom_pred & bottom_true) / k,
        }

    def _compute_precision_at_k(self, group: pd.DataFrame) -> dict:
        k = self.top_k
        top_pred = group.nlargest(k, self.pred_col)

        precision = (top_pred[self.label_col] > top_pred["BENCH_RET_5D"]).mean()

        return {f"Precision@{k}": precision}

    def _run_topk_dropout_portfolio(
        self,
        df: pd.DataFrame,
        initial_capital: float = 1_000_000,
        trading_days: int = 252,
        n_drop: int = 5,
        commission: float = 0.0003,
        stamp_tax: float = 0.001,
    ) -> dict:

        df = df.sort_values([self.date_col, self.pred_col], ascending=[True, False])
        dates = sorted(df[self.date_col].unique())

        capital = initial_capital
        nav_series = []
        daily_returns = []
        turnover_series = []

        prev_holdings = set()
        target_holdings = set()

        for date in dates:

            daily = df[df[self.date_col] == date]
            if len(daily) < self.top_k:
                continue

            # Realize return from previous holdings (T+1 -> T+2)
            if prev_holdings:
                held = daily[daily[self.instrument_col].isin(prev_holdings)]
                daily_ret = held["RET_1D"].mean() if not held.empty else 0.0

                capital *= 1.0 + daily_ret

                daily_returns.append(daily_ret)
                nav_series.append(capital)

            # Generate next-day target portfolio using today's signal
            if not prev_holdings:
                target_holdings = set(daily.head(self.top_k)[self.instrument_col])
            else:
                in_hold = daily[daily[self.instrument_col].isin(prev_holdings)]

                sell = set(
                    in_hold.nsmallest(n_drop, self.pred_col)[self.instrument_col]
                )

                buy = set(
                    daily[~daily[self.instrument_col].isin(prev_holdings)].head(n_drop)[
                        self.instrument_col
                    ]
                )

                target_holdings = (prev_holdings - sell) | buy

            # Execute rebalance at T+1 close and apply transaction cost
            if prev_holdings:
                sell_count = len(prev_holdings - target_holdings)
                buy_count = len(target_holdings - prev_holdings)

                turnover = (sell_count + buy_count) / self.top_k
                turnover_series.append(turnover)

                cost_rate = (
                    sell_count * (commission + stamp_tax) + buy_count * commission
                ) / self.top_k

                capital *= 1.0 - cost_rate

            prev_holdings = target_holdings.copy()

        if not daily_returns:
            return {}

        rets = np.array(daily_returns)
        nav = np.array(nav_series)

        ann_return = (nav[-1] / initial_capital) ** (trading_days / len(rets)) - 1.0

        ann_vol = rets.std() * np.sqrt(trading_days)

        sharpe = (
            rets.mean() / rets.std() * np.sqrt(trading_days)
            if rets.std() > 0
            else np.nan
        )

        max_drawdown = np.min(nav / np.maximum.accumulate(nav) - 1.0)

        avg_turnover = np.mean(turnover_series) if turnover_series else 0.0

        return {
            "ARR": ann_return,
            "Sharpe": sharpe,
            "MaxDrawdown": max_drawdown,
            "AnnVol": ann_vol,
            "AvgTurnover": avg_turnover,
        }

    def evaluate(self, pred_df: pd.DataFrame) -> pd.DataFrame:

        df = self._prepare_dataset(pred_df)

        # IC
        daily_ic = self._compute_daily_ic(df).dropna()
        ic = daily_ic.mean()
        icir = ic / daily_ic.std() if daily_ic.std() != 0 else np.nan

        # Hit Rate
        daily_hr = pd.DataFrame(
            df.groupby(self.date_col).apply(self._compute_hit_rate).tolist()
        )
        hr_stats = daily_hr.mean().to_dict()

        # Precision
        daily_prec = pd.DataFrame(
            df.groupby(self.date_col).apply(self._compute_precision_at_k).tolist()
        )
        prec_stats = daily_prec.mean().to_dict()

        # Portfolio
        portfolio_stats = self._run_topk_dropout_portfolio(df)

        results = {
            "IC": ic,
            "ICIR": icir,
            **hr_stats,
            **prec_stats,
            **portfolio_stats,
        }

        return pd.DataFrame([results]).to_markdown(index=False, floatfmt=".4f")
