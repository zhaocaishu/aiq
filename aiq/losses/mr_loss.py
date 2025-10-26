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
    Combined loss: Weighted MSE (for regression accuracy, weighted by target ranks) + MarginRankingLoss (for relative ranking consistency).

    The MSE loss is weighted based on the rank of targets (stock returns), giving higher weights to higher returns.
    The ranking loss ensures relative ranking consistency.

    Args:
        alpha (float): Weight for MarginRankingLoss component. Default: 0.7
        margin (float): Margin parameter for MarginRankingLoss. Default: 0.1
        weight_type (str): Type of weighting for MSE ('linear' or 'exponential'). Default: 'linear'
    """

    def __init__(
        self, alpha: float = 0.7, margin: float = 0.1, weight_type: str = "linear"
    ):
        super().__init__()
        self.alpha = alpha
        self.ranking_loss = MarginRankingLoss(margin=margin)
        self.weight_type = weight_type

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining weighted MSE and ranking loss.

        Args:
            preds (torch.Tensor): Predicted scores, shape (N,)
            targets (torch.Tensor): Ground truth values (stock returns), shape (N,)

        Returns:
            torch.Tensor: Combined scalar loss value.
        """
        preds = preds.view(-1)
        targets = targets.view(-1)
        N = preds.size(0)

        # Compute ranks of targets (higher returns get higher ranks)
        _, indices = targets.sort(descending=True)
        ranks = torch.zeros(N, device=targets.device)
        ranks[indices] = torch.arange(N, dtype=torch.float, device=targets.device)

        # Compute weights based on ranks (higher rank -> higher weight)
        if self.weight_type == "linear":
            # Linear weights: w = N - rank (highest return gets weight N, lowest gets weight 1)
            weights = N - ranks
            weights = weights / weights.sum()  # Normalize weights to sum to 1
        elif self.weight_type == "exponential":
            # Exponential weights: w = exp(-rank / N)
            weights = torch.exp(-ranks / N)
            weights = weights / weights.sum()  # Normalize weights to sum to 1
        else:
            raise ValueError("weight_type must be 'linear' or 'exponential'")

        # Weighted MSE loss component
        mse_loss = F.mse_loss(preds, targets, reduction="none")  # Shape (N,)
        weighted_mse_loss = (mse_loss * weights).sum()

        # Ranking loss component
        ranking_loss = self.ranking_loss(preds, targets)

        # Combined total loss
        total_loss = (1.0 - self.alpha) * weighted_mse_loss + self.alpha * ranking_loss

        return total_loss
