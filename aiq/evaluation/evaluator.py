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

        ret_1d = adj_close / Ref(adj_close, 1) - 1
        ret_5d = Ref(adj_close, -5) / Ref(adj_close, -1) - 1

        # Build core data dictionary
        data = {
            self.date_col: df[self.date_col],
            self.instrument_col: df[self.instrument_col],
            self.label_col: ret_5d,
            "Price": df["Close"],
            "Return": ret_1d
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
        cash = initial_capital
        positions = {}  # market value of holdings after previous close
        nav_series = []  # daily net asset value
        daily_returns = []  # daily portfolio return
        turnover_series = []  # daily turnover rate

        prev_total_value = initial_capital

        # Sort by instrument + date for proper shift
        df = df.sort_values([self.instrument_col, self.date_col])

        # Use previous day's prediction for today's trading decision
        df["Score"] = df.groupby(self.instrument_col)[self.pred_col].shift(1)
        df = df.dropna(subset=["Score"])

        dates = sorted(df[self.date_col].unique())

        for i, date in enumerate(dates):
            daily = df[df[self.date_col] == date]
            if daily.empty:
                continue

            # Convert daily data to a dictionary for fast lookup
            daily_dict = daily.set_index(self.instrument_col).to_dict(orient="index")

            # ---------- 1. Calculate daily return based on previous holdings ----------
            if positions:
                total_value = cash
                new_positions = {}
                for inst, value in positions.items():
                    if inst in daily_dict:
                        # Update market value using daily return
                        ret = daily_dict[inst]["Return"]
                        new_value = value * (1.0 + ret)
                        new_positions[inst] = new_value
                        total_value += new_value
                    else:
                        # Stock missing for the day, keep value unchanged
                        new_positions[inst] = value
                        total_value += value

                # Daily return and NAV
                daily_ret = total_value / prev_total_value - 1.0
                daily_returns.append(daily_ret)
                nav_series.append(total_value)
                prev_total_value = total_value

                # Update positions to post‑return market values (for next rebalance)
                positions = new_positions
            else:
                # No holdings, daily return is zero
                daily_returns.append(0.0)
                nav_series.append(prev_total_value)

            # ---------- 2. Rebalance decision (using yesterday's predictions, update positions for current day) ----------
            if len(daily) < self.top_k:
                # Not enough tradable stocks today to form target portfolio, skip rebalance
                continue

            current_holdings = set(positions.keys())

            # 2.1 Initial build
            if not current_holdings:
                # Filter stocks that can be bought (not limit‑up)
                buy_candidates = []
                for inst, row in daily_dict.items():
                    up_limit = row[self.up_limit_col]
                    if row["Price"] < up_limit:
                        buy_candidates.append(inst)
                # Take top_k by descending prediction
                buy_candidates.sort(key=lambda x: daily_dict[x]["Score"], reverse=True)
                buy_candidates = buy_candidates[: self.top_k]

                if buy_candidates:
                    cash_per_stock = cash / len(buy_candidates)
                    for inst in buy_candidates:
                        cost = cash_per_stock * commission
                        invest = (
                            cash_per_stock - cost
                        )  # post‑purchase market value (net of commission)
                        positions[inst] = invest
                        cash -= cash_per_stock  # cash reduced by total expenditure (including commission)
                    turnover_series.append(1.0)
                continue

            # 2.2 Rebalance with existing holdings
            # Sell phase: select the n_drop holdings with the lowest predictions and not limit‑down
            hold_preds = [
                (inst, daily_dict[inst]["Score"])
                for inst in current_holdings
                if inst in daily_dict
            ]
            hold_preds.sort(
                key=lambda x: x[1]
            )  # ascending prediction, low scores sold first

            sell_count = 0
            for inst, pred in hold_preds:
                if sell_count >= n_drop:
                    break
                row = daily_dict[inst]
                if row is None:
                    continue
                price = row["Price"]
                down_limit = row[self.down_limit_col]
                if price > down_limit:  # can be sold
                    value = positions.pop(inst)  # remove from holdings
                    cost = value * (commission + stamp_tax)  # selling expenses
                    cash += value - cost
                    sell_count += 1

            # Buy phase: select highest‑prediction stocks not currently held, number equals actual sold count
            not_hold = [inst for inst in daily_dict if inst not in positions]
            not_hold.sort(key=lambda x: daily_dict[x]["Score"], reverse=True)

            buy_list = []
            for inst in not_hold:
                if len(buy_list) >= sell_count:
                    break
                row = daily_dict[inst]
                up_limit = row[self.up_limit_col]
                if row["Price"] < up_limit:  # can be bought
                    buy_list.append(inst)

            if buy_list:
                cash_per_stock = cash / len(buy_list)
                for inst in buy_list:
                    cost = cash_per_stock * commission
                    invest = cash_per_stock - cost
                    positions[inst] = invest
                    cash -= cash_per_stock

            # Turnover rate
            turnover = (sell_count + len(buy_list)) / self.top_k
            turnover_series.append(turnover)

        # ---------- 3. Calculate performance metrics ----------
        rets = np.array(daily_returns)
        nav = np.array(nav_series)

        if len(rets) == 0:
            return {}

        # Annualised return
        ann_ret = (nav[-1] / initial_capital) ** (trading_days / len(nav)) - 1.0

        # Sharpe ratio
        if rets.std() > 0:
            sharpe = rets.mean() / rets.std() * np.sqrt(trading_days)
        else:
            sharpe = 0.0

        # Maximum drawdown
        mdd = np.min(nav / np.maximum.accumulate(nav) - 1.0)

        # Annualised volatility
        ann_vol = rets.std() * np.sqrt(trading_days)

        # Average turnover
        avg_turnover = np.mean(turnover_series) if turnover_series else 0.0

        return {
            "ARR": ann_ret,
            "Sharpe": sharpe,
            "MaxDrawdown": mdd,
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
