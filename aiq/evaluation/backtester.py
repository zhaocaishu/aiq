"""TopK-Dropout backtester with raw-price trading and adjusted-share valuation."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from aiq.utils.exchange import Exchange
from aiq.utils.decision import Order


@dataclass(frozen=True)
class Position:
    """一只股票的日终估值结果；权重使用小数比例。"""

    instrument: str
    adjusted_shares: float
    adjusted_close: float
    market_value: float
    weight: float
    price_fallback: bool


@dataclass(frozen=True)
class Portfolio:
    """单个交易日的完整日终记录。"""

    date: Any
    total_asset: float
    daily_return: float
    buy_value: float
    sell_value: float
    turnover: float
    cash: float
    position_value: float
    holdings: Tuple[Position, ...]


class BaseBacktester(ABC):
    """Optionally restrict purchases using a daily boolean universe column."""

    def __init__(
        self,
        top_k: int,
        date_col: str = "Date",
        instrument_col: str = "Instrument",
        up_limit_col: str = "Up_limit",
        down_limit_col: str = "Down_limit",
        adj_factor_col: str = "Adj_factor",
        score_col: str = "PRED_RET_5D",
        universe_col: Optional[str] = None,
        trading_calendar: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.top_k = top_k
        self.date_col = date_col
        self.instrument_col = instrument_col
        self.up_limit_col = up_limit_col
        self.down_limit_col = down_limit_col
        self.adj_factor_col = adj_factor_col
        self.score_col = score_col
        self.universe_col = universe_col
        self.trading_calendar = trading_calendar
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def run(self, df: pd.DataFrame, **kwargs) -> Dict[str, float]:
        pass

    @staticmethod
    def _valid_price(value: Any) -> bool:
        """Accept only finite, strictly positive numeric prices or factors."""
        try:
            return bool(np.isfinite(value) and value > 0)
        except (TypeError, ValueError):
            return False

    def _print_header(self, title: str, width: int = 72) -> None:
        self.logger.info("═" * width)
        self.logger.info(f"  {title}")
        self.logger.info("═" * width)

    def _print_kv(self, key: str, value: Any, indent: int = 2, width: int = 20) -> None:
        self.logger.info(f"{' ' * indent}▸ {key:<{width}} {value}")

    def _print_table_header(
        self,
        *cols: Any,
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

    def _log_daily_holdings(self, portfolio: Portfolio) -> None:
        """输出日终交易摘要和持仓快照"""
        self.logger.info(
            f"📅 {portfolio.date} | 总资产 {portfolio.total_asset:,.2f} | "
            f"日收益 {portfolio.daily_return:+.2%}"
        )
        self.logger.info(
            f"  买入 {portfolio.buy_value:,.2f} | 卖出 {portfolio.sell_value:,.2f} | "
            f"换手 {portfolio.turnover:.2%}"
        )
        self.logger.info(
            f"  持仓 {len(portfolio.holdings)} 只 | "
            f"持仓市值 {portfolio.position_value:,.2f} | 可用现金 {portfolio.cash:,.2f}"
        )
        if not portfolio.holdings:
            return

        widths, aligns = self._print_table_header(
            "股票代码",
            "复权份额",
            "后复权收盘价",
            "持仓市值",
            "持仓内权重",
            "备注",
            widths=[12, 14, 14, 14, 10, 10],
            aligns=["<", ">", ">", ">", ">", "<"],
            indent=2,
        )
        for holding in portfolio.holdings:
            self._print_table_row(
                holding.instrument,
                f"{holding.adjusted_shares:,.6f}",
                f"{holding.adjusted_close:.2f}",
                f"{holding.market_value:,.2f}",
                f"{holding.weight:.2%}",
                "价格回退" if holding.price_fallback else "",
                widths=widths,
                aligns=aligns,
                indent=2,
            )


class TopKDropoutBacktester(BaseBacktester):
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
        cash = initial_capital
        # positions stores synthetic adjusted shares: raw shares / entry adj_factor.
        # Raw prices still drive lot sizing, order values and exchange limit checks.
        positions: Dict[str, float] = {}
        hold_start: Dict[str, Any] = {}
        prev_prices: Dict[str, float] = {}

        # 初始资金作为基准净值，每个回测交易日记录一次收益。
        nav_series: List[float] = [initial_capital]
        daily_returns: List[float] = []
        turnover_series: List[float] = []
        exchange = Exchange()

        def _buy_shares(price: float, budget: float, available_cash: float) -> int:
            """按单票市值预算和含佣金的可用现金计算整手股数。"""
            buy_budget = min(
                budget,
                available_cash / (1.0 + commission),
                available_cash - min_commission,
            )
            # 修正整手临界点的浮点舍入，再按实际金额复核预算。
            lots = np.nextafter(buy_budget / (price * 100), np.inf)
            shares = max(0, int(lots)) * 100
            invest = shares * price
            if (
                invest > budget
                or invest + max(invest * commission, min_commission) > available_cash
            ):
                shares = max(0, shares - 100)
            return shares

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

        if self.trading_calendar is None:
            raise ValueError("必须提供完整的市场交易日历 trading_calendar")
        if df.empty:
            return {}

        df = df.copy()
        if self.adj_factor_col not in df.columns:
            raise ValueError(f"缺少复权因子列: {self.adj_factor_col}")
        df[self.adj_factor_col] = pd.to_numeric(
            df[self.adj_factor_col], errors="coerce"
        )
        raw_close = pd.to_numeric(df["Close"], errors="coerce")
        valid_close = raw_close.gt(0) & df[self.adj_factor_col].gt(0)
        df["_AdjClose"] = (raw_close * df[self.adj_factor_col]).where(valid_close)

        if self.universe_col is not None:
            if self.universe_col not in df.columns:
                raise ValueError(f"缺少每日股票池标记列: {self.universe_col}")
            if not pd.api.types.is_bool_dtype(df[self.universe_col]):
                raise ValueError(f"每日股票池标记列必须为布尔类型: {self.universe_col}")
            df[self.universe_col] = df[self.universe_col].fillna(False)
        df[self.date_col] = pd.to_datetime(df[self.date_col]).dt.strftime("%Y-%m-%d")
        full_calendar = sorted(
            set(pd.to_datetime(self.trading_calendar).strftime("%Y-%m-%d"))
        )
        date_index = {date: i for i, date in enumerate(full_calendar)}
        prev_date = dict(zip(full_calendar[1:], full_calendar[:-1]))
        scores = df.set_index([self.date_col, self.instrument_col])[self.score_col]
        keys = pd.MultiIndex.from_arrays(
            [df[self.date_col].map(prev_date), df[self.instrument_col]]
        )
        df["Score"] = scores.reindex(keys).to_numpy()

        # 日历可包含预加载历史，但只模拟输入数据覆盖的日期区间。
        start_date = df[self.date_col].min()
        end_date = df[self.date_col].max()
        calendar = [d for d in full_calendar if start_date <= d <= end_date]

        def _record_day(
            date: str,
            daily_dict: Dict[str, Dict[str, Any]],
            buy_value: float,
            sell_value: float,
            opening: bool = False,
        ) -> Portfolio:
            """完成日终估值、绩效统计，并生成只读的交易与持仓快照。"""
            position_value = 0.0
            valued_holdings = []
            for inst, adjusted_shares in positions.items():
                adjusted_close = daily_dict.get(inst, {}).get("_AdjClose")
                price_fallback = not self._valid_price(adjusted_close)
                if price_fallback:
                    adjusted_close = prev_prices.get(inst)
                    if not self._valid_price(adjusted_close):
                        raise ValueError(
                            f"{date}: 持仓 {inst} 缺少有效后复权收盘价及历史估值价格"
                        )
                else:
                    adjusted_close = float(adjusted_close)
                    prev_prices[inst] = adjusted_close

                market_value = adjusted_shares * adjusted_close
                position_value += market_value
                valued_holdings.append(
                    (
                        inst,
                        adjusted_shares,
                        adjusted_close,
                        market_value,
                        price_fallback,
                    )
                )

            valued_holdings.sort(key=lambda holding: holding[3], reverse=True)
            holdings = tuple(
                Position(
                    instrument=inst,
                    adjusted_shares=adjusted_shares,
                    adjusted_close=adjusted_close,
                    market_value=market_value,
                    weight=market_value / position_value if position_value > 0 else 0.0,
                    price_fallback=price_fallback,
                )
                for inst, adjusted_shares, adjusted_close, market_value, price_fallback in valued_holdings
            )

            total_asset = cash + position_value
            daily_return = total_asset / nav_series[-1] - 1.0
            daily_returns.append(daily_return)
            nav_series.append(total_asset)

            # 建仓日按买入金额计换手；其余日期沿用买卖金额均值口径。
            trade_value = buy_value if opening else (buy_value + sell_value) / 2
            turnover = trade_value / total_asset if total_asset > 0 else 0.0
            turnover_series.append(turnover)

            return Portfolio(
                date=date,
                total_asset=total_asset,
                daily_return=daily_return,
                buy_value=buy_value,
                sell_value=sell_value,
                turnover=turnover,
                cash=cash,
                position_value=position_value,
                holdings=holdings,
            )

        for date in calendar:
            i = date_index[date]
            daily = df[df[self.date_col] == date]
            if daily.empty:
                # 区间内整日行情缺失不能静默缩短统计期间。
                raise ValueError(f"{date}: 回测区间内缺少整日行情，请补齐数据")
            daily_dict = daily.set_index(self.instrument_col).to_dict(orient="index")

            def _is_tradable(inst_id: str, direction: Optional[int] = None) -> bool:
                # 成交、整手计算和涨跌停使用原始价格，复权因子用于份额换算。
                row = daily_dict.get(inst_id)
                if row is None or not self._valid_price(row.get("VWAP")):
                    return False
                if not self._valid_price(row.get(self.adj_factor_col)):
                    return False
                if (
                    direction == Order.BUY
                    and self.universe_col is not None
                    and not row[self.universe_col]
                ):
                    return False
                return exchange.is_stock_tradable(
                    stock_id=inst_id,
                    price_col="Close",
                    up_limit_col=self.up_limit_col,
                    down_limit_col=self.down_limit_col,
                    daily_dict=daily_dict,
                    direction=None if forbid_all_trade_at_limit else direction,
                )

            daily_buy_value = 0.0
            daily_sell_value = 0.0
            current_holdings = list(positions.keys())

            if not current_holdings:
                buy_candidates = [
                    inst
                    for inst in daily_dict
                    if np.isfinite(daily_dict[inst]["Score"])
                    and _is_tradable(inst, Order.BUY)
                ]
                if method_buy == "top":
                    buy_candidates.sort(
                        key=lambda x: daily_dict[x]["Score"], reverse=True
                    )
                else:
                    raise NotImplementedError(
                        f"不支持的建仓模式: method_buy={method_buy}"
                    )
                if buy_candidates:
                    invest_cash = cash * risk_degree
                    cash_per_stock = invest_cash / min(self.top_k, len(buy_candidates))
                    buy_queue = []
                    reserved_cash = 0.0
                    for inst in buy_candidates:
                        price = daily_dict[inst]["VWAP"]
                        lot_value = price * 100
                        lot_cost = lot_value + max(
                            lot_value * commission, min_commission
                        )
                        if (
                            lot_value > cash_per_stock
                            or lot_cost > cash - reserved_cash
                        ):
                            continue
                        buy_queue.append(inst)
                        reserved_cash += lot_cost
                        if len(buy_queue) == self.top_k:
                            break
                    self.logger.info(f"  ▶ 初始建仓  买入 {len(buy_queue)} 只")
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
                        price = daily_dict[inst].get("VWAP")
                        adj_factor = daily_dict[inst].get(self.adj_factor_col)
                        lot_value = price * 100
                        reserved_cash -= lot_value + max(
                            lot_value * commission, min_commission
                        )
                        shares = _buy_shares(
                            price, cash_per_stock, cash - reserved_cash
                        )
                        if shares < 100:
                            continue
                        invest = shares * price
                        comm = max(invest * commission, min_commission)
                        daily_buy_value += invest
                        cash -= invest + comm
                        positions[inst] = shares / float(adj_factor)
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

                # 无论建仓成功或继续空仓，都记录并展示完整的日终结果。
                portfolio = _record_day(
                    date, daily_dict, daily_buy_value, daily_sell_value, opening=True
                )
                self._log_daily_holdings(portfolio)
                continue

            score_map = {}
            for _inst, _row in daily_dict.items():
                _s = _row.get("Score", np.nan)
                score_map[_inst] = _s if np.isfinite(_s) else float("-inf")
            last_pool = sorted(
                current_holdings,
                key=lambda inst: score_map.get(inst, float("-inf")),
                reverse=True,
            )
            buy_pool = sorted(
                (
                    inst
                    for inst in daily_dict
                    if inst not in positions
                    and np.isfinite(daily_dict[inst]["Score"])
                    and _is_tradable(inst, Order.BUY)
                ),
                key=lambda inst: daily_dict[inst]["Score"],
                reverse=True,
            )
            if method_sell != "bottom":
                raise NotImplementedError(
                    f"不支持的卖出模式: method_sell={method_sell}"
                )

            # 先排除单票预算买不起一手的股票，再确定整组可成交的换仓计划。
            target_cash_per_stock = nav_series[-1] * risk_degree / self.top_k
            buy_pool = [
                inst
                for inst in buy_pool
                if daily_dict[inst]["VWAP"] * 100 <= target_cash_per_stock
            ]
            lot_costs = {
                inst: daily_dict[inst]["VWAP"] * 100
                + max(daily_dict[inst]["VWAP"] * 100 * commission, min_commission)
                for inst in buy_pool
            }
            n_today_need = n_drop + self.top_k - len(last_pool)
            while True:
                today_pool = buy_pool[: max(0, n_today_need)]
                comb = sorted(
                    last_pool + today_pool,
                    key=lambda inst: score_map.get(inst, float("-inf")),
                    reverse=True,
                )
                bottom_n = set(comb[self.top_k :])
                sell_candidates = [inst for inst in last_pool if inst in bottom_n]

                sell_queue = []
                blocked_hold, blocked_limit = 0, 0
                available_cash = cash
                for inst in sell_candidates:
                    if inst not in hold_start:
                        continue
                    hold_days = i - date_index[hold_start[inst]]
                    if hold_days < hold_thresh:
                        blocked_hold += 1
                        continue
                    if not _is_tradable(inst, Order.SELL):
                        blocked_limit += 1
                        continue
                    sell_value = (
                        positions[inst]
                        * daily_dict[inst]["VWAP"]
                        * daily_dict[inst][self.adj_factor_col]
                    )
                    available_cash += sell_value - (
                        max(sell_value * commission, min_commission)
                        + sell_value * stamp_tax
                    )
                    sell_queue.append(inst)

                n_buy_need = max(0, len(sell_queue) + self.top_k - len(last_pool))
                buy_queue = today_pool[:n_buy_need]
                buy_orders = []
                if not buy_queue:
                    break
                reserved_cash = sum(lot_costs[inst] for inst in buy_queue)
                if reserved_cash > available_cash:
                    # 资金不足时跳过一手成本最高的候选，重新核对替代关系。
                    buy_pool.remove(max(buy_queue, key=lot_costs.get))
                    continue

                cash_per_stock = min(
                    target_cash_per_stock, available_cash / len(buy_queue)
                )
                for inst in buy_queue:
                    price = daily_dict[inst]["VWAP"]
                    reserved_cash -= lot_costs[inst]
                    budget = max(cash_per_stock, price * 100)
                    shares = _buy_shares(price, budget, available_cash - reserved_cash)
                    if shares < 100:
                        break
                    invest = shares * price
                    comm = max(invest * commission, min_commission)
                    available_cash -= invest + comm
                    buy_orders.append((inst, shares, invest, comm))
                if len(buy_orders) == len(buy_queue):
                    break
                buy_pool.remove(buy_queue[len(buy_orders)])

            if blocked_hold or blocked_limit:
                self.logger.info(
                    f"  ⛔ 卖出阻塞: 持有期不足 {blocked_hold} 只, "
                    f"跌停/停牌/无效价格 {blocked_limit} 只"
                )

            sell_count = 0
            sold_list = []
            for inst in sell_queue:
                if inst not in positions:
                    continue
                price = daily_dict[inst].get("VWAP")
                adj_factor = daily_dict[inst].get(self.adj_factor_col)
                if not self._valid_price(price) or not self._valid_price(adj_factor):
                    continue
                # 卖出全部复权份额，等价于按当日原始 VWAP 卖出换算后的股数。
                adj_shares = positions.pop(inst)
                sell_value = adj_shares * float(price) * float(adj_factor)
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

            if buy_orders:
                self.logger.info(f"  ▶ 买入 {len(buy_orders)} 只")
                w, a = self._print_table_header(
                    "代码",
                    "价格",
                    "买入股数",
                    "佣金",
                    "净买入市值",
                    widths=[14, 10, 12, 10, 14],
                    indent=4,
                )
                for inst, shares, invest, comm in buy_orders:
                    price = daily_dict[inst].get("VWAP")
                    adj_factor = daily_dict[inst].get(self.adj_factor_col)
                    daily_buy_value += invest
                    cash -= invest + comm
                    positions[inst] = shares / float(adj_factor)
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

            portfolio = _record_day(
                date, daily_dict, daily_buy_value, daily_sell_value
            )
            self._log_daily_holdings(portfolio)

        rets = np.array(daily_returns)
        nav = np.array(nav_series)
        if len(rets) == 0:
            return {}
        ann_ret = (nav[-1] / initial_capital) ** (trading_days / len(rets)) - 1.0
        sharpe = (
            rets.mean() / rets.std() * np.sqrt(trading_days) if rets.std() > 0 else 0.0
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
