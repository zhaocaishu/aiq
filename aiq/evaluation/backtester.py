import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from aiq.utils.exchange import Exchange
from aiq.utils.decision import Order


class BaseBacktester(ABC):
    """
    Abstract Base Class for A-share multi-asset backtesting.
    Provides shared utilities like cross-sectional logging, table formatting,
    and standardized accounting infrastructure.
    """

    def __init__(
        self,
        top_k: int,
        date_col: str = "Date",
        instrument_col: str = "Instrument",
        up_limit_col: str = "Up_limit",
        down_limit_col: str = "Down_limit",
        pred_col: str = "PRED_RET_5D",
        label_col: str = "RET_5D",
        logger: Optional[logging.Logger] = None,
    ):
        self.top_k = top_k
        self.date_col = date_col
        self.instrument_col = instrument_col
        self.up_limit_col = up_limit_col
        self.down_limit_col = down_limit_col
        self.pred_col = pred_col
        self.label_col = label_col
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def run(self, df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Execute the backtest simulation loop. Must be implemented by subclasses."""
        pass

    # ═══════════════════════════════════════════════════════════════════════════════
    # 通用日志与高颜值表格格式化工具（子类直接复用）
    # ═══════════════════════════════════════════════════════════════════════════════

    def _print_header(self, title: str, width: int = 72) -> None:
        self.logger.info("═" * width)
        self.logger.info(f"  {title}")
        self.logger.info("═" * width)

    def _print_kv(self, key: str, value: Any, indent: int = 2, width: int = 20) -> None:
        self.logger.info(f"{' ' * indent}▸ {key:<{width}} {value}")

    def _print_table_header(
        self,
        *cols: str,
        widths: Optional[List[int]] = None,
        aligns: Optional[List[str]] = None,
        indent: int = 4,
    ) -> Tuple[List[int], List[str]]:
        if widths is None:
            widths = [max(len(str(c)) + 4, 12) for c in cols]
        if aligns is None:
            aligns = ["<"] + [">"] * (len(cols) - 1)

        line = " ".join(f"{c:{a}{w}}" for c, a, w in zip(cols, aligns, widths))
        sep = " ".join("─" * w for w in widths)

        self.logger.info(f"{' ' * indent}{line}")
        self.logger.info(f"{' ' * indent}{sep}")
        return widths, aligns

    def _print_table_row(
        self, *vals: Any, widths: List[int], aligns: List[str], indent: int = 4
    ) -> None:
        line = " ".join(f"{v:{a}{w}}" for v, a, w in zip(vals, aligns, widths))
        self.logger.info(f"{' ' * indent}{line}")

    def _log_daily_holdings(
        self,
        date: Any,
        positions: Dict[str, int],
        daily_dict: Dict[str, Dict[str, Any]],
        prev_prices: Optional[Dict[str, float]] = None,
    ) -> None:
        self.logger.info(f"📅 交易日：{date}")
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

        rows.sort(key=lambda x: x[3], reverse=True)

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


class TopKDropoutBacktester(BaseBacktester):
    """
    TopK-Dropout Strategy Backtester for A-shares.
    """

    def run(
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
        """Execute TopK-Dropout simulation loop based on prediction scores."""
        cash = initial_capital
        positions: Dict[str, int] = {}
        hold_start: Dict[str, Any] = {}
        prev_prices: Dict[str, float] = {}

        nav_series: List[float] = []
        daily_returns: List[float] = []
        turnover_series: List[float] = []

        exchange = Exchange()

        # ── 1. 策略启动日志横幅 ──
        self._print_header("【TopK-Dropout 策略启动】", width=72)
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

        # ── 2. 数据预处理 ──
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

            def _is_tradable(inst_id: str, direction: Optional[int] = None) -> bool:
                if inst_id not in daily_dict:
                    return False
                row_data = daily_dict[inst_id]

                is_yizi_limit_up = (row_data["High"] == row_data["Low"]) and (
                    row_data["Close"] >= row_data[self.up_limit_col] * 0.999
                )
                is_yizi_limit_down = (row_data["High"] == row_data["Low"]) and (
                    row_data["Close"] <= row_data[self.down_limit_col] * 1.001
                )

                if is_yizi_limit_up or is_yizi_limit_down:
                    return False

                return exchange.is_stock_tradable(
                    stock_id=inst_id,
                    price=row_data["VWAP"],
                    up_limit=row_data[self.up_limit_col],
                    down_limit=row_data[self.down_limit_col],
                    daily_dict=daily_dict,
                    direction=direction,
                )

            current_holdings = list(positions.keys())
            daily_buy_value = 0.0
            daily_sell_value = 0.0

            # ── 3.1 初始建仓 ──
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
                        price = daily_dict[inst]["VWAP"]
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

            # ── 3.2 再平衡信号计算 (TopK-Dropout 核心核心) ──
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

            if method_sell == "bottom":
                bottom_n = comb[-n_drop:] if n_drop > 0 else []
                sell_candidates = [inst for inst in last_pool if inst in bottom_n]
            else:
                raise NotImplementedError(
                    f"不支持的卖出模式: method_sell={method_sell}"
                )

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
                price = daily_dict[inst]["VWAP"]
                shares = positions.pop(inst)
                sell_value = shares * price

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
                current_total_asset = cash + sum(
                    (
                        positions[s] * daily_dict[s]["Close"]
                        if s in daily_dict
                        else positions[s] * prev_prices.get(s, 0.0)
                    )
                    for s in positions
                )
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
                    price = daily_dict[inst]["VWAP"]
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

            # ── 3.5 资产清算 ──
            position_value_after = 0.0
            for inst, shares in positions.items():
                if inst in daily_dict:
                    p = daily_dict[inst]["Close"]
                    position_value_after += shares * p
                    prev_prices[inst] = p
                else:
                    position_value_after += shares * prev_prices.get(inst, 0.0)

            total_after = cash + position_value_after
            if i > 0:
                daily_returns.append(total_after / nav_series[-1] - 1.0)

            nav_series.append(total_after)

            total_trade_value = daily_buy_value + daily_sell_value
            turnover = (
                (total_trade_value / 2) / position_value_after
                if position_value_after > 0
                else 0.0
            )
            turnover_series.append(turnover)

            self._log_daily_holdings(date, positions, daily_dict, prev_prices)

        # ── 4. 绩效指标生成 ──
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

        self._print_header("【TopK-Dropout 策略结束】", width=72)
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
