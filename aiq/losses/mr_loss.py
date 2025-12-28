import torch
import torch.nn as nn


class MarginRankingLoss(nn.Module):
    """
    Pairwise Ranking Loss for Stock Selection.
    This loss ensures that stocks with higher ground-truth returns
    get higher predicted scores than stocks with lower returns.
    """

    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.MarginRankingLoss(margin=margin, reduction="mean")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds (torch.Tensor): Predicted scores, shape (N,)
            targets (torch.Tensor): Ground truth values, shape (N,)
        Returns:
            Margin ranking loss (scalar if reduction='mean' or 'sum', tensor if 'none')
        """
        # Expand (N,) to (N, N) to create all possible pairs (i, j)
        # s_i: every row contains the score of stock i
        # s_j: every row contains the score of all other stocks
        s_i = preds.unsqueeze(1)  # shape (N, 1)
        s_j = preds.unsqueeze(0)  # shape (1, N)

        r_i = targets.unsqueeze(1)  # shape (N, 1)
        r_j = targets.unsqueeze(0)  # shape (1, N)

        # Calculate the ranking label y
        # y = 1 if r_i > r_j; y = -1 if r_i < r_j; y = 0 if equal
        target_diff = r_i - r_j
        y = torch.sign(target_diff)

        # Filter valid pairs
        # We ignore diagonal elements (i == j) and cases where returns are identical
        mask = y != 0

        # Apply mask and flatten to 1D for MarginRankingLoss compatibility
        valid_s_i = s_i.expand(len(preds), len(preds))[mask]
        valid_s_j = s_j.expand(len(preds), len(preds))[mask]
        valid_y = y[mask]

        # Compute the mean loss across all valid pairs
        loss = self.loss_fn(valid_s_i, valid_s_j, valid_y)

        return loss
