from .mr_loss import MarginRankingLoss
from .ic_loss import ICLoss
from .topk_loss import TopKLoss
from .fuse_loss import FuseLoss
from .cb_loss import ClassBalancedLoss

__all__ = [
    "MarginRankingLoss",
    "ICLoss",
    "TopKLoss",
    "FuseLoss",
    "ClassBalancedLoss",
]
