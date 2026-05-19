import numpy as np
from typing import Dict, Optional, Any

from .decision import Order


class Exchange:
    """Simplified exchange simulator for trading restrictions (suspension & limit-up/down)."""

    def check_stock_limit(
        self,
        price: float,
        up_limit: float,
        down_limit: float,
        direction: Optional[int] = None,
    ) -> bool:
        """
        Check if price hits limit-up or limit-down.

        Returns:
            True if trading is limited (not tradable), False otherwise.
        """
        if direction is None:
            return price >= up_limit or price <= down_limit
        if direction == Order.BUY:
            return price >= up_limit
        if direction == Order.SELL:
            return price <= down_limit
        raise ValueError(f"direction {direction} is not supported")

    def check_stock_suspended(self, stock_id: str, daily_dict: Dict[str, Dict[str, Any]]) -> bool:
        """Return True if stock is suspended (no valid close price)."""
        stock_data = daily_dict.get(stock_id)
        if stock_data is None:
            return True
        close = stock_data.get("Close")  # 或 stock_data.get("adj_close")，取决于你的列名
        if close is None:
            return True
        return bool(np.isnan(close))

    def is_stock_tradable(
        self,
        stock_id: str,
        price: float,
        up_limit: float,
        down_limit: float,
        daily_dict: Dict[str, Dict[str, Any]],
        direction: Optional[int] = None,
    ) -> bool:
        """Return True if stock is tradable (not suspended and not hitting limit)."""
        suspended = self.check_stock_suspended(stock_id, daily_dict)
        limited = self.check_stock_limit(price, up_limit, down_limit, direction)
        return not (suspended or limited)
        