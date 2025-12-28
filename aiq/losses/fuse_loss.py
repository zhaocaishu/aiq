import torch
import torch.nn as nn

from .topk_loss import TopKLoss


class FuseLoss(nn.Module):
    """
    Hybrid Loss Function: Combined MSE and Ranking Loss.

    Args:
        alpha (float): Scaling factor for the Ranking Loss component.
    """

    def __init__(self, alpha: float = 0.6):
        super().__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.ranking_loss = TopKLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the hybrid loss.

        Args:
            preds (torch.Tensor): Predicted scores, shape (N,)
            targets (torch.Tensor): Ground truth values (stock returns), shape (N,)

        Returns:
            torch.Tensor: A scalar tensor representing the weighted sum of losses.
        """
        # Flatten tensors to ensure they are 1D
        preds = preds.view(-1)
        targets = targets.view(-1)

        # MSE loss component
        mse_loss = self.mse_loss(preds, targets)  # Shape (N,)

        # Ranking loss component
        ranking_loss = self.ranking_loss(preds, targets)

        # Combined total loss
        total_loss = mse_loss + self.alpha * ranking_loss

        return total_loss
