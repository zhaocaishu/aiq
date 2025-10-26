import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds (torch.Tensor): Predicted scores, shape (N,)
            targets (torch.Tensor): Ground truth values, shape (N,)
        Returns:
            Margin ranking loss (scalar)
        """
        N = preds.size(0)

        # Compute pairwise differences
        pred_diff = preds.unsqueeze(1) - preds.unsqueeze(0)  # [N, N]
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)  # [N, N]

        # Create mask for valid pairs (upper triangle, excluding diagonal)
        mask = torch.triu(torch.ones(N, N, device=preds.device), diagonal=1).bool()
        mask = mask & (target_diff != 0)

        # Compute loss: max(0, margin - sign(target_diff) * pred_diff)
        loss = torch.clamp(self.margin - torch.sign(target_diff) * pred_diff, min=0)
        loss = loss[mask].mean()

        return loss


class MSERankLoss(nn.Module):
    """
    Combined loss: MSE (for regression accuracy) + MarginRankingLoss (for relative ranking consistency).

    This loss encourages predictions to be numerically close to targets (via MSE)
    while also maintaining correct ranking order (via MarginRankingLoss).

    Args:
        alpha (float): Weight for MSE loss component. Default: 0.7
        margin (float): Margin parameter for MarginRankingLoss. Default: 0.1
    """

    def __init__(self, alpha: float = 0.7, margin: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.ranking_loss = MarginRankingLoss(margin=margin)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining MSE and ranking loss.

        Args:
            preds (torch.Tensor): Predicted scores, shape (N,)
            targets (torch.Tensor): Ground truth values, shape (N,)

        Returns:
            torch.Tensor: Combined scalar loss value.
        """
        preds = preds.view(-1)
        targets = targets.view(-1)

        # MSE loss component
        mse_loss = F.mse_loss(preds, targets, reduction="mean")

        # Lambda ranking loss component
        ranking_loss = self.ranking_loss(preds, targets)

        # Combined total loss
        total_loss = (1.0 - self.alpha) * mse_loss + self.alpha * ranking_loss

        return total_loss
