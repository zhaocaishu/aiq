import torch
import torch.nn as nn

from .mr_loss import MarginRankingLoss


class FuseLoss(nn.Module):
    """
    Combined loss: Weighted MSE (for regression accuracy, weighted by target ranks) + MarginRankingLoss (for relative ranking consistency).

    The MSE loss is weighted based on the rank of targets (stock returns), giving higher weights to higher returns.
    The ranking loss ensures relative ranking consistency.

    Args:
        alpha (float): Weight for MarginRankingLoss component. Default: 0.6
        margin (float): Margin parameter for MarginRankingLoss. Default: 0.1
        weight_type (str): Type of weighting for MSE ('linear' or 'exponential'). Default: 'linear'
    """

    def __init__(self, alpha: float = 0.6, margin: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss(reduction="none")
        self.ranking_loss = MarginRankingLoss(margin=margin, reduction="mean")

    # Compute normalized weights for targets, assigning higher weights to higher target values.
    def _compute_weights(self, targets):
        N = targets.numel()  # Get the size of the targets tensor
        if N == 0:
            return torch.tensor(
                [], device=targets.device, dtype=targets.dtype
            )  # Handle empty tensor
        elif N == 1:
            return torch.tensor(
                [1.0], device=targets.device, dtype=targets.dtype
            )  # Single target gets weight 1

        _, indices = targets.sort(
            descending=False
        )  # Ascending sort: higher values get higher indices
        ranks = torch.zeros(N, device=targets.device, dtype=targets.dtype)
        ranks[indices] = torch.arange(N, device=targets.device, dtype=targets.dtype)
        weights = ranks / (N - 1)  # Normalize ranks to [0, 1]
        return weights

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

        # Weighted MSE loss component
        weights = self._compute_weights(targets)
        mse_loss = self.mse_loss(preds, targets)  # Shape (N,)
        weighted_mse_loss = (mse_loss * weights).mean()

        # Ranking loss component
        ranking_loss = self.ranking_loss(preds, targets)

        # Combined total loss
        total_loss = (1.0 - self.alpha) * weighted_mse_loss + self.alpha * ranking_loss

        return total_loss
