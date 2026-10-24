import math

import torch
import torch.nn as nn


class ICLoss(nn.Module):
    """
    Information Coefficient Loss for Stock Selection.

    This loss encourages positive cross-sectional Pearson correlation between
    predicted scores and ground-truth returns by minimizing one minus IC.
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and greater than zero")
        self.epsilon = epsilon

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes Pearson IC loss across valid stocks in the batch.

        Args:
            preds (torch.Tensor): Predicted scores of shape (N, 1).
            targets (torch.Tensor): Ground truth returns of shape (N, 1).

        Returns:
            torch.Tensor: Scalar one-minus-IC loss. Non-finite pairs are ignored;
            fewer than two valid stocks or constant targets produce zero loss.
        """
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)
        if preds.numel() != targets.numel():
            raise ValueError("preds and targets must contain the same number of values")

        # Accumulate low-precision inputs in float32 while preserving float64.
        dtype = (
            torch.float64
            if preds.dtype == torch.float64 or targets.dtype == torch.float64
            else torch.float32
        )
        preds = preds.to(dtype=dtype)
        targets = targets.to(dtype=dtype)

        mask = torch.isfinite(preds) & torch.isfinite(targets)
        preds = preds[mask]
        targets = targets[mask]

        N = preds.size(0)
        if N <= 1:
            return preds.sum() * 0.0
        if torch.all(targets == targets[0]):
            return preds.sum() * 0.0

        centered_preds = preds - preds.mean()
        centered_targets = targets - targets.mean()
        pred_var = centered_preds.square().mean()
        target_var = centered_targets.square().mean()

        # Clamp each variance before sqrt to keep zero-variance gradients finite.
        denominator = (
            pred_var.clamp_min(self.epsilon).sqrt()
            * target_var.clamp_min(self.epsilon).sqrt()
        )
        covariance = (centered_preds * centered_targets).mean()
        ic = (covariance / denominator).clamp(-1.0, 1.0)

        return 1.0 - ic
