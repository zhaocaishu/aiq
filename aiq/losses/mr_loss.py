import torch
import torch.nn as nn


class MarginRankingLoss(nn.Module):
    """
    Pairwise Ranking Loss for Stock Selection.
    This loss ensures that stocks with higher ground-truth returns
    get higher predicted scores than stocks with lower returns.
    """

    def __init__(self, margin: float = 0.1, target_thresh: float = 0.01):
        super().__init__()
        self.margin = margin
        self.target_thresh = target_thresh
        self.loss_fn = nn.MarginRankingLoss(margin=margin, reduction="mean")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds (torch.Tensor): Predicted scores, shape (N, 1)
            targets (torch.Tensor): Ground truth values, shape (N, 1)
        Returns:
            torch.Tensor: A scalar tensor representing the margin ranking loss
        """
        # Expand (N, 1) to (N, N) to create all possible pairs (i, j)
        # s_i: every row contains the score of stock i
        # s_j: every row contains the score of all other stocks
        s_i = preds  # shape (N, 1)
        s_j = preds.T  # shape (1, N)

        r_i = targets  # shape (N, 1)
        r_j = targets.T  # shape (1, N)

        # Calculate the ranking label y
        # y = 1 if r_i > r_j; y = -1 if r_i < r_j; y = 0 if equal
        target_diff = r_i - r_j
        y = torch.sign(target_diff)

        # Create mask to filter invalid/noisy pairs
        # Exclude diagonal (i == j) where r_diff is 0
        # Exclude pairs where the return difference is below target_thresh
        mask = torch.abs(target_diff) > self.target_thresh

        # Apply mask and flatten to 1D for MarginRankingLoss compatibility
        valid_s_i = s_i.expand(len(preds), len(preds))[mask]
        valid_s_j = s_j.expand(len(preds), len(preds))[mask]
        valid_y = y[mask]

        # Compute the mean loss across all valid pairs
        loss = self.loss_fn(valid_s_i, valid_s_j, valid_y)

        return loss
