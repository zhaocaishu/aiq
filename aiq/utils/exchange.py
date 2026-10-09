import numpy as np
from typing import Dict, Optional, Any

from .decision import Order


class Exchange:
    """Simplified exchange simulator for trading restrictions (suspension & limit-up/down)."""

    def check_stock_limit(
        self,
        stock_id: str,
        price_col: str,
        up_limit_col: str,
        down_limit_col: str,
        daily_dict: Dict[str, Dict[str, Any]],
        direction: Optional[int] = None,
    ) -> bool:
        """
        返回 True 表示禁止交易，False 表示未触发限制。

        direction:
            None: 涨停、跌停均禁止交易。
            Order.BUY: 禁止买入涨停股票。
            Order.SELL: 禁止卖出跌停股票。

        价格字段应为数值类型；无效或缺失价格数据禁止交易。
        +inf/-inf 可分别表示没有涨停价/跌停价限制。
        """
        if direction not in (None, Order.BUY, Order.SELL):
            raise ValueError(f"direction {direction} is not supported")

        row = daily_dict.get(stock_id)
        if row is None:
            return True

        price = row.get(price_col)
        if price is None or not np.isfinite(price) or price <= 0:
            return True

        up_limit = row.get(up_limit_col)
        down_limit = row.get(down_limit_col)

        if up_limit is None or down_limit is None:
            return True

        valid_up = (np.isfinite(up_limit) and up_limit > 0) or up_limit == np.inf
        valid_down = (
            np.isfinite(down_limit) and down_limit > 0
        ) or down_limit == -np.inf

        if not valid_up or not valid_down or down_limit > up_limit:
            return True

        if direction == Order.BUY:
            return bool(price >= up_limit)

        if direction == Order.SELL:
            return bool(price <= down_limit)

        return bool(price >= up_limit or price <= down_limit)

    def check_stock_suspended(
        self, stock_id: str, daily_dict: Dict[str, Dict[str, Any]]
    ) -> bool:
        """Conservatively treat missing or invalid quotes as suspended."""
        row = daily_dict.get(stock_id)
        if row is None:
            return True

        close = row.get("Close")
        if close is None or not np.isfinite(close) or close <= 0:
            return True

        return False

    def is_stock_tradable(
        self,
        stock_id: str,
        price_col: str,
        up_limit_col: str,
        down_limit_col: str,
        daily_dict: Dict[str, Dict[str, Any]],
        direction: Optional[int] = None,
    ) -> bool:
        """Return True if stock is tradable (not suspended and not hitting limit)."""
        if self.check_stock_suspended(stock_id, daily_dict):
            return False

        return not self.check_stock_limit(
            stock_id, price_col, up_limit_col, down_limit_col, daily_dict, direction
        )
