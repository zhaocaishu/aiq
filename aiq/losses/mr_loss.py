import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginRankingLoss(nn.Module):
    def __init__(self, margin=1.0, reduction="mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds (torch.Tensor): Predicted scores, shape (N,)
            targets (torch.Tensor): Ground truth values, shape (N,)
        Returns:
            Margin ranking loss (scalar if reduction='mean' or 'sum', tensor if 'none')
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
        loss = loss[mask]

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss
