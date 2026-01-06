import torch
import torch.nn as nn

from .topk_loss import TopKLoss
from .mr_loss import MarginRankingLoss


class HybridLoss(nn.Module):
    """
    Hybrid Loss Function: Main Loss + Auxiliary Loss.

    Args:
        aux_loss_weight (float): Weight coefficient for the auxiliary loss, default is 0.5.
    """

    def __init__(self, aux_loss_weight: float = 0.5):
        super().__init__()
        self.aux_loss_weight = aux_loss_weight
        
        self.main_loss_fn = TopKLoss()
        self.aux_loss_fn = MarginRankingLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the hybrid loss.

        Args:
            preds (torch.Tensor): Predicted scores, shape (N, 1)
            targets (torch.Tensor): Ground truth values (stock returns), shape (N, 1)

        Returns:
            torch.Tensor: A scalar tensor representing the weighted sum of losses
        """
        # Main loss component
        main_loss = self.main_loss_fn(preds, targets)

        # Auxiliary  loss component
        aux_loss = self.aux_loss_fn(preds, targets)

        # Combined total loss
        total_loss = main_loss + self.aux_loss_weight * aux_loss

        return total_loss
