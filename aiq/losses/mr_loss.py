import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LambdaLoss(nn.Module):
    """
    LambdaLoss for stock selection with continuous returns.

    Weights are set as absolute return differences to prioritize high-return pairs.

    Args:
        top_k: Top-k cutoff for ranking metrics. If None, use all stocks.
        sigma: Score difference scaling factor for sigmoid.
        eps: Epsilon for numerical stability.
    """

    def __init__(
        self, top_k: Optional[int] = None, sigma: float = 1.0, eps: float = 1e-10
    ):
        super().__init__()
        self.top_k = top_k
        self.sigma = sigma
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LambdaLoss.

        Args:
            preds: Predicted scores, shape (N, 1) or (N,)
            targets: True returns, shape (N, 1) or (N,)

        Returns:
            Computed LambdaLoss value
        """
        # Ensure tensors are 1D
        preds = preds.squeeze(-1)
        targets = targets.squeeze(-1)
        num_stocks = preds.shape[0]

        # Sort predictions and get corresponding targets
        preds_sorted, sorted_indices = torch.sort(preds, descending=True)
        targets_sorted = targets[sorted_indices]

        # Compute pairwise return differences and weights
        return_diffs = targets_sorted.unsqueeze(1) - targets_sorted.unsqueeze(0)
        valid_pairs = return_diffs > 0
        weights = torch.abs(return_diffs)

        # Apply top-k mask if specified
        k = self.top_k or num_stocks
        topk_mask = torch.zeros(
            (num_stocks, num_stocks), dtype=torch.bool, device=preds.device
        )
        topk_mask[:k, :k] = True
        final_mask = valid_pairs & topk_mask

        # Compute score differences and probabilities
        score_diffs = preds_sorted.unsqueeze(1) - preds_sorted.unsqueeze(0)
        score_diffs = torch.clamp(score_diffs, min=-1e8, max=1e8)

        probabilities = torch.sigmoid(self.sigma * score_diffs)
        probabilities = torch.clamp(probabilities, min=self.eps)

        # Compute weighted probabilities and losses
        weighted_probs = torch.pow(probabilities, weights)
        weighted_probs = torch.clamp(weighted_probs, min=self.eps)

        losses = torch.log2(weighted_probs)[final_mask]

        if losses.numel() == 0:
            return torch.tensor(0.0, device=preds.device)

        return -losses.mean()


class MSERankLoss(nn.Module):
    """
    Combined loss function: MSE + LambdaLoss.

    Args:
        alpha: Weight for MSE loss component.
        beta: Weight for LambdaLoss component.
        top_k: Top-k cutoff for LambdaLoss ranking.
        sigma: Score difference scaling factor for LambdaLoss.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        top_k: int = 30,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_loss = LambdaLoss(top_k=top_k, sigma=sigma)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for combined MSE + LambdaLoss.

        Args:
            preds: Predicted scores, shape (N, 1) or (N,)
            targets: True returns, shape (N, 1) or (N,)

        Returns:
            Combined loss value
        """
        # MSE loss component
        mse_loss = F.mse_loss(preds, targets, reduction="mean")

        # Lambda ranking loss component
        lambda_loss = self.lambda_loss(preds, targets)

        # Combined total loss
        total_loss = self.alpha * mse_loss + self.beta * lambda_loss

        return total_loss
