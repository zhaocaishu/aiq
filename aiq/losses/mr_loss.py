import torch
import torch.nn as nn


class MarginRankingLoss(nn.Module):
    """
    Pairwise Ranking Loss for Stock Selection.

    This loss encourages the model to rank stocks correctly by ensuring that
    stocks with higher ground-truth returns receive higher predicted scores
    than those with lower returns, maintaining at least a specified margin.
    """

    def __init__(self, margin: float = 0.1, epsilon: float = 1e-3):
        super().__init__()
        self.margin = margin
        self.epsilon = epsilon

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the ranking loss by comparing all pairs of stocks in the batch.

        Args:
            preds (torch.Tensor): Predicted scores of shape (N, 1).
            targets (torch.Tensor): Ground truth returns of shape (N, 1).

        Returns:
            torch.Tensor: Scalar mean loss across all valid pairs.
        """
        preds = preds.view(-1, 1)
        targets = targets.view(-1, 1)

        N = preds.size(0)
        if N <= 1:
            return preds.new_tensor(0.0, requires_grad=True)

        diff_r = targets - targets.T
        mask = diff_r.abs() > self.epsilon
        if not mask.any():
            return preds.new_tensor(0.0, requires_grad=True)

        y = torch.where(diff_r > 0, 1.0, -1.0)

        diff_s = preds - preds.T
        loss = torch.relu(self.margin - y * diff_s)

        return loss[mask].mean()
