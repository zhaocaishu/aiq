import torch
import torch.nn as nn

from .topk_loss import TopKLoss


class FuseLoss(nn.Module):
    """
    FuseLoss = weighted MSE (weights based on target ranking, optional) + ranking loss (TopKLoss).

    Args:
        alpha (float): balancing coefficient in [0, 1].
                       total_loss = (1-alpha) * mse_loss + alpha * topk_loss
    """
    
    def __init__(self, alpha: float = 0.6, top_k: int = 30):
        super().__init__()
        self.alpha = alpha
        self.top_k = top_k
        self.mse_loss = nn.MSELoss()
        self.topk_loss = TopKLoss(top_k=top_k)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining MSE and TopK loss.

        Args:
            preds (torch.Tensor): Predicted scores, shape (N, 1)
            targets (torch.Tensor): Ground truth values (stock returns), shape (N, 1)

        Returns:
            torch.Tensor: Combined scalar loss value.
        """
        # MSE loss component
        mse_loss = self.mse_loss(preds, targets)

        # Ranking loss component
        topk_loss = self.topk_loss(preds, targets)

        # Combined total loss
        total_loss = (1.0 - self.alpha) * mse_loss + self.alpha * topk_loss

        return total_loss
