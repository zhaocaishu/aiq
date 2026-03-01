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
        up_limit_col: str = "Up_limit",
        down_limit_col: str = "Down_limit",
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
        self.up_limit_col = up_limit_col
        self.down_limit_col = down_limit_col
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

        # Build core data dictionary
        data = {
            self.date_col: df[self.date_col],
            self.instrument_col: df[self.instrument_col],
            self.label_col: ret_5d,
            "Price": df["Close"],
            "RET_1D": ret_1d,
        }

        # Optional add trading limits if available
        for col in [self.up_limit_col, self.down_limit_col]:
            if col in df.columns:
                data[col] = df[col]

        return pd.DataFrame(data)

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

        cash = initial_capital
        positions = {}  # {instrument: market_value}
        nav_series = []
        daily_returns = []
        turnover_series = []

        prev_total_value = initial_capital

        for date in dates:
            daily = df[df[self.date_col] == date]

            # Apply natural position returns (T+1 realized PnL)
            if positions:
                total_value = cash

                for inst in list(positions.keys()):
                    row = daily[daily[self.instrument_col] == inst]
                    if row.empty:
                        continue

                    ret = row["RET_1D"].values[0]
                    positions[inst] *= 1.0 + ret

                    total_value += positions[inst]

                # Capital-weighted daily portfolio return
                daily_ret = total_value / prev_total_value - 1.0
                daily_returns.append(daily_ret)
                nav_series.append(total_value)

                prev_total_value = total_value
            else:
                # No position, NAV unchanged
                nav_series.append(prev_total_value)
                daily_returns.append(0.0)

            # Rebalance logic (executed at close)
            if len(daily) < self.top_k:
                continue

            current_holdings = set(positions.keys())

            # Initial portfolio construction
            if not current_holdings:
                buy_candidates = daily[
                    daily["Price"] < daily[self.up_limit_col]  # not limit-up
                ].head(self.top_k)

                if buy_candidates.empty:
                    continue

                cash_per_stock = cash / len(buy_candidates)

                for _, row in buy_candidates.iterrows():
                    cost = cash_per_stock * commission
                    invest = cash_per_stock - cost
                    positions[row[self.instrument_col]] = invest
                    cash -= cash_per_stock

                turnover_series.append(1.0)
                continue

            # Sell phase (drop worst n holdings)
            hold_df = daily[daily[self.instrument_col].isin(current_holdings)]
            drop_candidates = set(
                hold_df.nsmallest(n_drop, self.pred_col)[self.instrument_col]
            )

            sell_count = 0

            for inst in drop_candidates:
                row = daily[daily[self.instrument_col] == inst]
                if row.empty:
                    continue

                price = row["Price"].values[0]
                down_limit = row[self.down_limit_col].values[0]

                # Only sell if not limit-down
                if price > down_limit:
                    value = positions.pop(inst)

                    # Commission + stamp tax (sell side)
                    cost = value * (commission + stamp_tax)
                    cash += value - cost
                    sell_count += 1

            #  Buy phase (fill vacancies) ----
            not_hold = daily[~daily[self.instrument_col].isin(positions.keys())]

            buy_list = []
            for _, row in not_hold.iterrows():
                if len(buy_list) >= sell_count:
                    break

                # Only buy if not limit-up
                if row["Price"] < row[self.up_limit_col]:
                    buy_list.append(row[self.instrument_col])

            if buy_list:
                cash_per_stock = cash / len(buy_list)

                for inst in buy_list:
                    cost = cash_per_stock * commission
                    invest = cash_per_stock - cost
                    positions[inst] = invest
                    cash -= cash_per_stock

            # Portfolio turnover ratio
            turnover = (sell_count + len(buy_list)) / self.top_k
            turnover_series.append(turnover)

        # Performance statistics
        rets = np.array(daily_returns)
        nav = np.array(nav_series)

        # Annualized return
        ann_ret = (nav[-1] / initial_capital) ** (trading_days / len(nav)) - 1.0

        # Annualized Sharpe ratio
        sharpe = (
            rets.mean() / rets.std() * np.sqrt(trading_days) if rets.std() > 0 else 0
        )

        # Maximum drawdown
        mdd = np.min(nav / np.maximum.accumulate(nav) - 1.0)

        return {
            "ARR": ann_ret,
            "Sharpe": sharpe,
            "MaxDrawdown": mdd,
            "AnnVol": rets.std() * np.sqrt(trading_days),
            "AvgTurnover": np.mean(turnover_series) if turnover_series else 0,
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
