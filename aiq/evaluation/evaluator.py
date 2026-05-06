import numpy as np
import pandas as pd
import logging

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
        logger: logging.Logger = None,
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

        # 统一日志句柄：外部传入优先，否则新建
        self.logger = logger or logging.getLogger(__name__)

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
            "Return": ret_1d,
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

    # ═══════════════════════════════════════════════════════════════════════════════
    # 日志格式化工具（全部通过 self.logger.info 输出）
    # ═══════════════════════════════════════════════════════════════════════════════
    def _print_header(self, title: str, width: int = 72):
        """打印带标题的分隔线"""
        self.logger.info(f"{'═' * width}")
        self.logger.info(f"  {title}")
        self.logger.info(f"{'═' * width}")

    def _print_sub_header(self, title: str, width: int = 72):
        """打印子标题"""
        self.logger.info(f"  ┌{'─' * (width - 4)}┐")
        self.logger.info(f"  │ {title:<{width - 6}}│")
        self.logger.info(f"  └{'─' * (width - 4)}┘")

    def _print_kv(self, key: str, value, indent: int = 2, width: int = 20):
        """打印键值对，自动对齐"""
        self.logger.info(f"{' ' * indent}▸ {key:<{width}} {value}")

    def _print_table_header(self, *cols, widths=None, aligns=None, indent: int = 4):
        """
        打印表格表头。
        aligns: 每列的对齐方式，'<' 左对齐，'>' 右对齐，'^' 居中。
                默认第一列左对齐（文本），其余右对齐（数值）。
        """
        if widths is None:
            widths = [max(len(str(c)) + 4, 12) for c in cols]
        if aligns is None:
            aligns = ["<"] + [">"] * (len(cols) - 1)
        line = " ".join(f"{c:{a}{w}}" for c, a, w in zip(cols, aligns, widths))
        sep = " ".join("─" * w for w in widths)
        self.logger.info(f"{' ' * indent}{line}")
        self.logger.info(f"{' ' * indent}{sep}")
        return widths, aligns

    def _print_table_row(self, *vals, widths, aligns, indent: int = 4):
        """打印表格行，对齐方式与表头保持一致"""
        line = " ".join(f"{v:{a}{w}}" for v, a, w in zip(vals, aligns, widths))
        self.logger.info(f"{' ' * indent}{line}")

    # ═══════════════════════════════════════════════════════════════════════════════

    def _run_topk_dropout_portfolio(
        self,
        df: pd.DataFrame,
        initial_capital: float = 1_000_000,
        trading_days: int = 252,
        n_drop: int = 5,
        commission: float = 0.0003,
        stamp_tax: float = 0.001,
        min_commission: float = 5,
    ) -> dict:
        cash = initial_capital
        positions = {}
        nav_series = []
        daily_returns = []
        turnover_series = []

        prev_total_value = initial_capital

        # ═══════════════════════════════════════════════════════════════════════════
        # 策略启动横幅
        # ═══════════════════════════════════════════════════════════════════════════
        self._print_header("【策略启动】", width=72)
        self._print_kv("初始资金", f"{initial_capital:>15,.2f}", width=18)
        self._print_kv("持仓上限", f"{self.top_k:>15} 只", width=18)
        self._print_kv("每期调出", f"{n_drop:>15} 只", width=18)
        self._print_kv("佣金费率", f"{commission:>15.4%}", width=18)
        self._print_kv("印花税率", f"{stamp_tax:>15.4%}", width=18)
        self.logger.info(f"{'═' * 72}")

        df = df.sort_values([self.instrument_col, self.date_col])
        df["Score"] = df.groupby(self.instrument_col)[self.pred_col].shift(1)
        df = df.dropna(subset=["Score"])

        trading_dates = sorted(df[self.date_col].unique())

        for i, date in enumerate(trading_dates):
            daily = df[df[self.date_col] == date]
            if daily.empty:
                continue

            daily_dict = daily.set_index(self.instrument_col).to_dict(orient="index")

            # ═══════════════════════════════════════════════════════════════════════
            # 1. 盘前状态（市值重估）
            # ═══════════════════════════════════════════════════════════════════════
            if positions:
                total_value = cash
                new_positions = {}
                for inst, value in positions.items():
                    if inst in daily_dict:
                        ret = daily_dict[inst]["Return"]
                        new_value = value * (1.0 + ret)
                        new_positions[inst] = new_value
                        total_value += new_value
                    else:
                        new_positions[inst] = value
                        total_value += value

                daily_ret = total_value / prev_total_value - 1.0
                daily_returns.append(daily_ret)
                nav_series.append(total_value)
                prev_total_value = total_value
                positions = new_positions

                # ── 盘前状态面板 ──
                self._print_sub_header(f"{date}  盘前状态", width=72)
                self._print_kv("现金", f"{cash:>14,.2f}", width=12)
                self._print_kv(
                    "持仓市值",
                    f"{sum(positions.values()):>14,.2f}  ({len(positions)} 只)",
                    width=12,
                )
                self._print_kv("总资产", f"{total_value:>14,.2f}", width=12)
                self._print_kv("日收益率", f"{daily_ret:>+14.4%}", width=12)
            else:
                daily_returns.append(0.0)
                nav_series.append(prev_total_value)
                self._print_sub_header(f"{date}  盘前状态（空仓）", width=72)
                self._print_kv("现金", f"{cash:>14,.2f}", width=12)
                self._print_kv("总资产", f"{prev_total_value:>14,.2f}", width=12)

            # ═══════════════════════════════════════════════════════════════════════
            # 2. 调仓决策
            # ═══════════════════════════════════════════════════════════════════════
            if len(daily) < self.top_k:
                self.logger.info(f"  ⚠  可交易股票不足 {self.top_k} 只，跳过调仓")
                continue

            current_holdings = set(positions.keys())

            # ── 2.1 初始建仓 ──
            if not current_holdings:
                buy_candidates = []
                for inst, row in daily_dict.items():
                    up_limit = row[self.up_limit_col]
                    if row["Price"] < up_limit:
                        buy_candidates.append(inst)
                buy_candidates.sort(key=lambda x: daily_dict[x]["Score"], reverse=True)
                buy_candidates = buy_candidates[: self.top_k]

                if buy_candidates:
                    cash_per_stock = cash / len(buy_candidates)

                    self.logger.info(f"  ▶ 初始建仓  买入 {len(buy_candidates)} 只")
                    w, a = self._print_table_header(
                        "代码",
                        "价格",
                        "分配现金",
                        "佣金",
                        "净买入市值",
                        widths=[14, 10, 14, 10, 14],
                        indent=4,
                    )
                    for inst in buy_candidates:
                        cost = max(cash_per_stock * commission, min_commission)
                        invest = cash_per_stock - cost
                        positions[inst] = invest
                        cash -= cash_per_stock
                        price = daily_dict[inst]["Price"]
                        self._print_table_row(
                            inst,
                            f"{price:.2f}",
                            f"{cash_per_stock:,.2f}",
                            f"{cost:,.2f}",
                            f"{invest:,.2f}",
                            widths=w,
                            aligns=a,
                            indent=4,
                        )
                    turnover_series.append(1.0)

                    # 建仓后汇总
                    total_after = cash + sum(positions.values())
                    self.logger.info(
                        f"  ▸ 建仓后  现金 {cash:>12,.2f}  |  持仓 {sum(positions.values()):>12,.2f}  |  总资产 {total_after:>12,.2f}"
                    )
                continue

            # ── 2.2 再平衡：卖出 ──
            hold_preds = [
                (inst, daily_dict[inst]["Score"])
                for inst in current_holdings
                if inst in daily_dict
            ]
            hold_preds.sort(key=lambda x: x[1])

            sell_count = 0
            sold_list = []
            for inst, _ in hold_preds:
                if sell_count >= n_drop:
                    break
                row = daily_dict[inst]
                if row is None:
                    continue
                price = row["Price"]
                down_limit = row[self.down_limit_col]
                if price > down_limit:  # can sell
                    value = positions.pop(inst)
                    comm_cost = max(value * commission, min_commission)
                    tax_cost = value * stamp_tax
                    cost = comm_cost + tax_cost
                    cash += value - cost
                    sold_list.append((inst, value, cost, value - cost))
                    sell_count += 1

            if sold_list:
                self.logger.info(f"  ▶ 卖出 {sell_count} 只")
                w, a = self._print_table_header(
                    "代码",
                    "卖出前市值",
                    "费用合计",
                    "净回笼",
                    widths=[14, 14, 12, 14],
                    indent=4,
                )
                for inst, val, cost, net in sold_list:
                    self._print_table_row(
                        inst,
                        f"{val:,.2f}",
                        f"{cost:,.2f}",
                        f"{net:,.2f}",
                        widths=w,
                        aligns=a,
                        indent=4,
                    )

            # ── 2.3 再平衡：买入 ──
            not_hold = [inst for inst in daily_dict if inst not in positions]
            not_hold.sort(key=lambda x: daily_dict[x]["Score"], reverse=True)

            buy_list = []
            for inst in not_hold:
                if len(buy_list) >= sell_count:
                    break
                row = daily_dict[inst]
                up_limit = row[self.up_limit_col]
                if row["Price"] < up_limit:
                    buy_list.append(inst)

            if buy_list:
                cash_per_stock = cash / len(buy_list)
                self.logger.info(f"  ▶ 买入 {len(buy_list)} 只")
                w, a = self._print_table_header(
                    "代码",
                    "价格",
                    "分配现金",
                    "佣金",
                    "净买入市值",
                    widths=[14, 10, 14, 10, 14],
                    indent=4,
                )
                for inst in buy_list:
                    cost = max(cash_per_stock * commission, min_commission)
                    invest = cash_per_stock - cost
                    positions[inst] = invest
                    cash -= cash_per_stock
                    price = daily_dict[inst]["Price"]
                    self._print_table_row(
                        inst,
                        f"{price:.2f}",
                        f"{cash_per_stock:,.2f}",
                        f"{cost:,.2f}",
                        f"{invest:,.2f}",
                        widths=w,
                        aligns=a,
                        indent=4,
                    )

            # ── 调仓后汇总 ──
            total_after = cash + sum(positions.values())
            self.logger.info(
                f"  ▸ 调仓后  现金 {cash:>12,.2f}  |  持仓 {sum(positions.values()):>12,.2f}  |  总资产 {total_after:>12,.2f}"
            )

            turnover = (sell_count + len(buy_list)) / self.top_k
            turnover_series.append(turnover)

        # ═══════════════════════════════════════════════════════════════════════════
        # 3. 策略结束汇总
        # ═══════════════════════════════════════════════════════════════════════════
        rets = np.array(daily_returns)
        nav = np.array(nav_series)

        if len(rets) == 0:
            return {}

        ann_ret = (nav[-1] / initial_capital) ** (trading_days / len(nav)) - 1.0
        sharpe = (
            rets.mean() / rets.std() * np.sqrt(trading_days) if rets.std() > 0 else 0.0
        )
        mdd = np.min(nav / np.maximum.accumulate(nav) - 1.0)
        ann_vol = rets.std() * np.sqrt(trading_days)
        avg_turnover = np.mean(turnover_series) if turnover_series else 0.0

        self._print_header("【策略结束】", width=72)
        self._print_kv("最终总资产", f"{nav[-1]:>15,.2f}", width=16)
        self._print_kv("年化收益", f"{ann_ret:>+15.4%}", width=16)
        self._print_kv("夏普比率", f"{sharpe:>15.4f}", width=16)
        self._print_kv("最大回撤", f"{mdd:>+15.4%}", width=16)
        self._print_kv("年化波动", f"{ann_vol:>15.4%}", width=16)
        self._print_kv("平均换手", f"{avg_turnover:>15.4%}", width=16)
        self.logger.info(f"{'═' * 72}")

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

        self.logger.info(f"{'═' * 72}")
        # ═══════════════════════════════════════════════════════════════════════════

        results = {
            "IC": ic,
            "ICIR": icir,
            **hr_stats,
            **prec_stats,
            **portfolio_stats,
        }

        return pd.DataFrame([results]).to_markdown(index=False, floatfmt=".4f")
