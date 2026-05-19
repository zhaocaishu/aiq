from enum import IntEnum
from dataclasses import dataclass
from typing import ClassVar


class OrderDir(IntEnum):
    # Order direction
    SELL = 0
    BUY = 1


@dataclass
class Order:
    """
    stock_id : str
    amount : float
    direction : int
        Order.SELL for sell; Order.BUY for buy
    """

    stock_id: str
    amount: float  # `amount` is a non-negative and adjusted value
    direction: OrderDir

    SELL: ClassVar[OrderDir] = OrderDir.SELL
    BUY: ClassVar[OrderDir] = OrderDir.BUY
