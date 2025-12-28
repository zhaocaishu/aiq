import torch
import torch.nn as nn

from .mr_loss import MarginRankingLoss
from .topk_loss import TopKLoss


class FuseLoss(nn.Module):
    """
    Hybrid Loss Function: Combined TopKLoss and Margin Ranking Loss.

    Args:
        alpha (float): Weight for MarginRankingLoss component. Default: 0.6
        margin (float): Margin for MarginRankingLoss component. Default: 0.1
    """

    def __init__(self, alpha: float = 0.6, margin: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.topk_loss = TopKLoss()
        self.ranking_loss = MarginRankingLoss(margin=margin)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining topk loss and ranking loss.

        Args:
            preds (torch.Tensor): Predicted scores, shape (N,)
            targets (torch.Tensor): Ground truth values (stock returns), shape (N,)

        Returns:
            torch.Tensor: Combined scalar loss value.
        """
        preds = preds.view(-1)
        targets = targets.view(-1)

        # Topk loss component
        topk_loss = self.topk_loss(preds, targets)  # Shape (N,)

        # Ranking loss component
        ranking_loss = self.ranking_loss(preds, targets)

        # Combined total loss
        total_loss = topk_loss + self.alpha * ranking_loss

        return total_loss
