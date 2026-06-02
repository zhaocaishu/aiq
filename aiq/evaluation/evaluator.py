import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Tuple

from aiq.dataset.loader import DataLoader
from aiq.ops import Ref
from aiq.utils.exchange import Exchange
from aiq.utils.decision import Order


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

    def _compute_bench_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute benchmark 5D returns."""
        ret_5d = Ref(df["Close"], -5) / Ref(df["Close"], -1) - 1
        data = {
            self.date_col: df[self.date_col],
            self.instrument_col: df[self.instrument_col],
            self.label_col: ret_5d,
        }

        return pd.DataFrame(data)

    def _prepare_dataset(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Align prediction / label / benchmark returns."""

        # Instruments
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
        inst_features = inst_features[
            [
                self.date_col,
                self.instrument_col,
                self.up_limit_col,
                self.down_limit_col,
                "Close",
            ]
        ]

        # Benchmark
        extended_end_time = (
            pd.to_datetime(self.end_time) + pd.Timedelta(days=10)
        ).strftime("%Y-%m-%d")
        bench_features = DataLoader.load_markets_features(
            self.data_dir, [self.benchmark], self.start_time, extended_end_time
        )

        bench_ret = self._compute_bench_returns(bench_features).rename(
            columns={
                self.label_col: "BENCH_RET_5D",
            }
        )[[self.date_col, "BENCH_RET_5D"]]

        df = inst_features.merge(
            pred_df[
                [self.date_col, self.instrument_col, self.pred_col, self.label_col]
            ],
            on=[self.date_col, self.instrument_col],
            how="inner",
        ).merge(bench_ret, on=self.date_col, how="inner")

        assert set(df[self.date_col]) == set(
            pred_df[self.date_col]
        ), f"{self.date_col} mismatch"
        assert set(df[self.instrument_col]) == set(
            pred_df[self.instrument_col]
        ), f"{self.instrument_col} mismatch"
        assert not pred_df[self.pred_col].isna().any(), f"{self.pred_col} contains NaN"

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

    def _print_header(self, title: str, width: int = 72) -> None:
        """打印带标题的粗分隔线"""
        self.logger.info("═" * width)
        self.logger.info(f"  {title}")
        self.logger.info("═" * width)

    def _print_sub_header(self, title: str, width: int = 72) -> None:
        """打印子标题外框"""
        self.logger.info(f"  ┌{'─' * (width - 4)}┐")
        self.logger.info(f"  │ {title:<{width - 6}}│")
        self.logger.info(f"  └{'─' * (width - 4)}┘")

    def _print_kv(self, key: str, value: Any, indent: int = 2, width: int = 20) -> None:
        """打印对齐的键值对"""
        self.logger.info(f"{' ' * indent}▸ {key:<{width}} {value}")

    def _print_table_header(
        self,
        *cols: str,
        widths: Optional[List[int]] = None,
        aligns: Optional[List[str]] = None,
        indent: int = 4,
    ) -> Tuple[List[int], List[str]]:
        """打印结构化表格的表头"""
        if widths is None:
            widths = [max(len(str(c)) + 4, 12) for c in cols]
        if aligns is None:
            # 默认：第一列左对齐（文本），其余列右对齐（数值）
            aligns = ["<"] + [">"] * (len(cols) - 1)

        line = " ".join(f"{c:{a}{w}}" for c, a, w in zip(cols, aligns, widths))
        sep = " ".join("─" * w for w in widths)

        self.logger.info(f"{' ' * indent}{line}")
        self.logger.info(f"{' ' * indent}{sep}")
        return widths, aligns

    def _print_table_row(
        self, *vals: Any, widths: List[int], aligns: List[str], indent: int = 4
    ) -> None:
        """打印表格数据行，保持与表头严格对齐"""
        line = " ".join(f"{v:{a}{w}}" for v, a, w in zip(vals, aligns, widths))
        self.logger.info(f"{' ' * indent}{line}")

    def _log_daily_holdings(
        self,
        date: Any,
        positions: Dict[str, int],
        daily_dict: Dict[str, Dict[str, Any]],
        prev_prices: Optional[Dict[str, float]] = None,
    ) -> None:
        """输出每日持仓明细表"""
        self.logger.info(f"📅 交易日：{date}")

        # 1. 数据对齐与准备
        rows = []
        total_mv = 0.0
        prev_prices = prev_prices or {}

        for inst, shares in positions.items():
            if inst in daily_dict:
                price = daily_dict[inst]["Close"]
                suspended = False
            else:
                price = prev_prices.get(inst)
                suspended = True

            mv = shares * price if price is not None else 0.0
            total_mv += mv
            rows.append((inst, shares, price, mv, suspended))

        # 按市值降序排列
        rows.sort(key=lambda x: x[3], reverse=True)

        # 2. 布局宽度定义
        col_widths = {
            "code": 12,
            "shares": 10,
            "price": 10,
            "mv": 14,
            "wt": 8,
            "note": 6,
        }
        total_width = sum(col_widths.values()) + len(col_widths) * 3 - 1
        sep = "  " + "─" * total_width

        # 3. 打印表头
        header = (
            f"  {'股票代码':<{col_widths['code']}}   "
            f"{'股数':>{col_widths['shares']}}   "
            f"{'收盘价':>{col_widths['price']}}   "
            f"{'持仓市值':>{col_widths['mv']}}   "
            f"{'权重%':>{col_widths['wt']}}   "
            f"{'备注':<{col_widths['note']}}"
        )
        self.logger.info(header)
        self.logger.info(sep)

        # 4. 打印数据行
        for inst, shares, price, mv, suspended in rows:
            shares_str = f"{shares:,}"
            price_str = f"{price:.2f}" if price is not None else "—"
            mv_str = f"{mv:,.2f}" if mv > 0 else "0.00"
            weight = (mv / total_mv * 100) if total_mv > 0 else 0.0
            note = "停牌" if suspended else ""

            line = (
                f"  {inst:<{col_widths['code']}}   "
                f"{shares_str:>{col_widths['shares']}}   "
                f"{price_str:>{col_widths['price']}}   "
                f"{mv_str:>{col_widths['mv']}}   "
                f"{weight:>{col_widths['wt']}.2f}   "
                f"{note:<{col_widths['note']}}"
            )
            self.logger.info(line)

        # 5. 打印总计栏
        self.logger.info(sep)
        sum_shares = f"{len(positions):,}"
        sum_mv = f"{total_mv:,.2f}"
        total_line = (
            f"  {'合计':<{col_widths['code']}}   "
            f"{sum_shares:>{col_widths['shares']}}   "
            f"{'':>{col_widths['price']}}   "
            f"{sum_mv:>{col_widths['mv']}}   "
            f"{'100.00':>{col_widths['wt']}}   "
            f"{'':<{col_widths['note']}}"
        )
        self.logger.info(total_line)
        self.logger.info("")

    # ═══════════════════════════════════════════════════════════════════════════════
    # 核心回测引擎
    # ═══════════════════════════════════════════════════════════════════════════════

    def _run_topk_dropout_portfolio(
        self,
        df: pd.DataFrame,
        initial_capital: float = 1_000_000,
        trading_days: int = 252,
        n_drop: int = 5,
        hold_thresh: int = 1,
        commission: float = 0.0003,
        stamp_tax: float = 0.001,
        min_commission: float = 5,
        risk_degree: float = 0.95,
        method_sell: str = "bottom",
        method_buy: str = "top",
        forbid_all_trade_at_limit: bool = False,
    ) -> Dict[str, float]:
        """TopK-Dropout 回测核心引擎（A股适配版）"""
        # 状态记账变量初始化
        cash = initial_capital
        positions: Dict[str, int] = {}  # stock_id -> 持仓股数
        hold_start: Dict[str, Any] = {}  # stock_id -> 首次买入日期
        prev_prices: Dict[str, float] = {}  # 记录上一次有效收盘价以处理停牌

        # 绩效跟踪序列
        nav_series: List[float] = []
        daily_returns: List[float] = []
        turnover_series: List[float] = []

        exchange = Exchange()  # 外部映射的交易所状态判定实例

        # ── 1. 策略启动横幅 ──
        self._print_header("【策略启动】", width=72)
        self._print_kv("初始资金", f"{initial_capital:>15,.2f}", width=18)
        self._print_kv("持仓上限", f"{self.top_k:>15} 只", width=18)
        self._print_kv("每期调出", f"{n_drop:>15} 只", width=18)
        self._print_kv("最小持有", f"{hold_thresh:>15} 日", width=18)
        self._print_kv("风险仓位", f"{risk_degree:>15.2%}", width=18)
        self._print_kv("佣金费率", f"{commission:>15.4%}", width=18)
        self._print_kv("印花税率", f"{stamp_tax:>15.4%}", width=18)
        limit_mode = (
            "禁止双向交易" if forbid_all_trade_at_limit else "允许卖涨停/买跌停"
        )
        self._print_kv("涨跌停模式", limit_mode, width=18)
        self.logger.info("═" * 72)

        # ── 2. 数据预处理（T-1日预测截面对齐） ──
        df = df.sort_values([self.instrument_col, self.date_col])
        df["Score"] = df.groupby(self.instrument_col)[self.pred_col].shift(1)
        df = df.dropna(subset=["Score"])
        trading_dates = sorted(df[self.date_col].unique())

        # ── 3. 历史日历主循环 ──
        for i, date in enumerate(trading_dates):
            daily = df[df[self.date_col] == date]
            if daily.empty:
                continue

            daily_dict = daily.set_index(self.instrument_col).to_dict(orient="index")

            # 内置可交易性复用校验闭包
            def _is_tradable(inst_id: str, direction: Optional[int] = None) -> bool:
                if inst_id not in daily_dict:
                    return False
                row_data = daily_dict[inst_id]
                return exchange.is_stock_tradable(
                    stock_id=inst_id,
                    price=row_data["Close"],
                    up_limit=row_data[self.up_limit_col],
                    down_limit=row_data[self.down_limit_col],
                    daily_dict=daily_dict,
                    direction=direction,
                )

            current_holdings = list(positions.keys())

            # ═════════════════════════════════════════════════════════════════
            # 日交易市值统计（量化标准换手率计算基础）
            # ═════════════════════════════════════════════════════════════════
            daily_buy_value = 0.0
            daily_sell_value = 0.0

            # ── 3.1 初始无仓位建仓 ──
            if not current_holdings:
                buy_candidates = [
                    inst
                    for inst in daily_dict
                    if _is_tradable(
                        inst, None if forbid_all_trade_at_limit else Order.BUY
                    )
                ]

                if method_buy == "top":
                    buy_candidates.sort(
                        key=lambda x: daily_dict[x]["Score"], reverse=True
                    )
                else:
                    raise NotImplementedError(
                        f"不支持的建仓模式: method_buy={method_buy}"
                    )

                buy_candidates = buy_candidates[: self.top_k]

                if buy_candidates:
                    invest_cash = cash * risk_degree
                    cash_per_stock = invest_cash / len(buy_candidates)

                    self.logger.info(f"  ▶ 初始建仓  买入 {len(buy_candidates)} 只")
                    w, a = self._print_table_header(
                        "代码",
                        "价格",
                        "买入股数",
                        "佣金",
                        "净买入市值",
                        widths=[14, 10, 12, 10, 14],
                        indent=4,
                    )

                    for inst in buy_candidates:
                        price = daily_dict[inst]["Close"]
                        shares = (
                            int((cash_per_stock / price) / 100) * 100
                        )  # 整手国内限制
                        if shares < 100:
                            continue

                        invest = shares * price
                        comm = max(invest * commission, min_commission)

                        # 现金流边界防透支保护
                        if cash < invest + comm:
                            max_shares = int(((cash - comm) / price) / 100) * 100
                            if max_shares < 100:
                                continue
                            shares = max_shares
                            invest = shares * price
                            comm = max(invest * commission, min_commission)

                        # ── 累计买入市值（换手率分子） ──
                        daily_buy_value += invest

                        cash -= invest + comm
                        positions[inst] = shares
                        hold_start[inst] = date

                        self._print_table_row(
                            inst,
                            f"{price:.2f}",
                            f"{shares}",
                            f"{comm:,.2f}",
                            f"{invest:,.2f}",
                            widths=w,
                            aligns=a,
                            indent=4,
                        )

                    # ── 日终持仓市值与换手率（统一清算逻辑） ──
                    position_value_after = 0.0
                    for inst, shares in positions.items():
                        if inst in daily_dict:
                            p = daily_dict[inst]["Close"]
                            position_value_after += shares * p
                            prev_prices[inst] = p
                        else:
                            position_value_after += shares * prev_prices.get(inst, 0.0)

                    turnover = (
                        (daily_buy_value / 2) / position_value_after
                        if position_value_after > 0
                        else 0.0
                    )
                    turnover_series.append(turnover)
                    nav_series.append(cash + position_value_after)
                    self._log_daily_holdings(date, positions, daily_dict)
                else:
                    turnover_series.append(0.0)
                    nav_series.append(cash)
                continue

            # ── 3.2 动态再平衡信号计算（TopK-Dropout 核心算法） ──

            # 1. 当前持仓（含停牌）按分数降序排序，停牌股 Score 视为 -inf 排到末尾
            last_scores = sorted(
                [
                    (
                        inst,
                        (
                            daily_dict[inst]["Score"]
                            if inst in daily_dict
                            else float("-inf")
                        ),
                    )
                    for inst in current_holdings
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            last_pool = [x[0] for x in last_scores]

            # 2. 筛选外部非持仓可买入的标的候选池
            not_hold = [inst for inst in daily_dict if inst not in positions]
            tradable_not_hold = [
                inst
                for inst in not_hold
                if _is_tradable(inst, None if forbid_all_trade_at_limit else Order.BUY)
            ]

            if method_buy == "top":
                tradable_not_hold.sort(
                    key=lambda x: daily_dict[x]["Score"], reverse=True
                )
            else:
                raise NotImplementedError(f"不支持的买入模式: method_buy={method_buy}")

            n_today_need = n_drop + self.top_k - len(last_pool)
            today_pool = tradable_not_hold[: max(0, n_today_need)]

            # 3. 合并新老池重新排序，防止出现卖出高分买入低分的反向交易
            comb_pool = last_pool + today_pool
            comb_scores = sorted(
                [
                    (
                        inst,
                        (
                            daily_dict[inst]["Score"]
                            if inst in daily_dict
                            else float("-inf")
                        ),
                    )
                    for inst in comb_pool
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            comb = [x[0] for x in comb_scores]

            # 4. 判定掉队标的 (Dropout Candidates)
            if method_sell == "bottom":
                bottom_n = comb[-n_drop:] if n_drop > 0 else []
                sell_candidates = [inst for inst in last_pool if inst in bottom_n]
            else:
                raise NotImplementedError(
                    f"不支持的卖出模式: method_sell={method_sell}"
                )

            # 5. 锁定期 (Hold Threshold) 约束检查
            sell_queue = []
            for inst in sell_candidates:
                if inst not in hold_start:
                    continue
                hold_days = trading_dates.index(date) - trading_dates.index(
                    hold_start[inst]
                )
                if hold_days < hold_thresh:
                    continue
                if not _is_tradable(
                    inst, None if forbid_all_trade_at_limit else Order.SELL
                ):
                    continue
                sell_queue.append(inst)

            # 6. 计算买入对冲队列
            n_buy_need = len(sell_queue) + self.top_k - len(last_pool)
            buy_candidates = today_pool[: max(0, n_buy_need)]
            buy_queue = [
                inst
                for inst in buy_candidates
                if inst not in positions
                and _is_tradable(inst, None if forbid_all_trade_at_limit else Order.BUY)
            ]

            # ── 3.3 交易执行：卖出控制 ──
            sell_count = 0
            sold_list = []
            for inst in sell_queue:
                if inst not in positions:
                    continue
                price = daily_dict[inst]["Close"]
                shares = positions.pop(inst)
                sell_value = shares * price

                # ── 累计卖出市值（换手率分子） ──
                daily_sell_value += sell_value

                comm = max(sell_value * commission, min_commission)
                tax = sell_value * stamp_tax
                total_cost = comm + tax

                cash += sell_value - total_cost
                sell_count += 1
                sold_list.append(
                    (inst, sell_value, total_cost, sell_value - total_cost)
                )
                hold_start.pop(inst, None)

            if sold_list:
                self.logger.info(f"  ▶ 卖出 {sell_count} 只")
                w, a = self._print_table_header(
                    "代码",
                    "卖出市值",
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

            # ── 3.4 交易执行：买入控制 ──
            if buy_queue:
                # 动态穿透估算当前包含停牌价的总资产
                current_total_asset = cash + sum(
                    (
                        positions[s] * daily_dict[s]["Close"]
                        if s in daily_dict
                        else positions[s] * prev_prices.get(s, 0.0)
                    )
                    for s in positions
                )

                # 依据目标风险模型分配单标的等权重额度
                target_cash_per_stock = (current_total_asset * risk_degree) / self.top_k
                cash_per_stock = (
                    min(target_cash_per_stock, cash / len(buy_queue))
                    if buy_queue
                    else 0.0
                )

                self.logger.info(f"  ▶ 买入 {len(buy_queue)} 只")
                w, a = self._print_table_header(
                    "代码",
                    "价格",
                    "买入股数",
                    "佣金",
                    "净买入市值",
                    widths=[14, 10, 12, 10, 14],
                    indent=4,
                )

                for inst in buy_queue:
                    price = daily_dict[inst]["Close"]
                    shares = int((cash_per_stock / price) / 100) * 100
                    if shares < 100:
                        continue

                    invest = shares * price
                    comm = max(invest * commission, min_commission)

                    if cash < invest + comm:
                        max_shares = int(((cash - comm) / price) / 100) * 100
                        if max_shares < 100:
                            continue
                        shares = max_shares
                        invest = shares * price
                        comm = max(invest * commission, min_commission)

                    # ── 累计买入市值（换手率分子） ──
                    daily_buy_value += invest

                    cash -= invest + comm
                    positions[inst] = shares
                    hold_start[inst] = date

                    self._print_table_row(
                        inst,
                        f"{price:.2f}",
                        f"{shares}",
                        f"{comm:,.2f}",
                        f"{invest:,.2f}",
                        widths=w,
                        aligns=a,
                        indent=4,
                    )

            # ── 3.5 调仓结束日终资产清算 ──
            position_value_after = 0.0
            for inst, shares in positions.items():
                if inst in daily_dict:
                    p = daily_dict[inst]["Close"]
                    position_value_after += shares * p
                    prev_prices[inst] = p
                else:
                    # 遭遇停牌：沿用其历史最近一次可得的收盘价记账
                    position_value_after += shares * prev_prices.get(inst, 0.0)

            total_after = cash + position_value_after

            if i > 0:
                daily_returns.append(total_after / nav_series[-1] - 1.0)

            nav_series.append(total_after)

            # ═════════════════════════════════════════════════════════════════
            # 单边换手率
            # ═════════════════════════════════════════════════════════════════
            total_trade_value = daily_buy_value + daily_sell_value
            turnover = (
                (total_trade_value / 2) / position_value_after
                if position_value_after > 0
                else 0.0
            )
            turnover_series.append(turnover)

            self._log_daily_holdings(date, positions, daily_dict, prev_prices)

        # ── 4. 回测绩效汇总与输出 ──
        rets = np.array(daily_returns)
        nav = np.array(nav_series)

        if len(rets) == 0:
            return {}

        ann_ret = (nav[-1] / initial_capital) ** (trading_days / len(nav)) - 1.0
        sharpe = (
            (rets.mean() / rets.std() * np.sqrt(trading_days))
            if rets.std() > 0
            else 0.0
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
        self._print_kv("平均换手(单边)", f"{avg_turnover:>15.4%}", width=16)
        self.logger.info("═" * 72)

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
        icir = (ic / daily_ic.std()) * np.sqrt(252 / 5) if daily_ic.std() != 0 else np.nan

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
