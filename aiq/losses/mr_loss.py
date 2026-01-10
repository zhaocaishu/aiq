import torch
import torch.nn as nn


class MarginRankingLoss(nn.Module):
    """
    Pairwise Ranking Loss for Stock Selection.

    This loss encourages the model to rank stocks correctly by ensuring that
    stocks with higher ground-truth returns receive higher predicted scores
    than those with lower returns, maintaining at least a specified margin.
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
        # Use 'mean' reduction to average the loss across all valid stock pairs
        self.loss_fn = nn.MarginRankingLoss(margin=margin, reduction="mean")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the ranking loss by comparing all pairs of stocks in the batch.

        Args:
            preds (torch.Tensor): Predicted scores of shape (N, 1).
            targets (torch.Tensor): Ground truth returns of shape (N, 1).

        Returns:
            torch.Tensor: Scalar mean loss across all valid pairs.
        """
        # Generate all pairs (i, j) using broadcasting
        # s_i: Replicated across rows; element (i, j) represents score of stock i
        # s_j: Replicated across columns; element (i, j) represents score of stock j
        s_i = preds  # shape (N, 1)
        s_j = preds.T  # shape (1, N)

        r_i = targets  # shape (N, 1)
        r_j = targets.T  # shape (1, N)

        # Determine ranking labels (y) for all pairs
        # y_ij = 1  if return_i > return_j
        # y_ij = -1 if return_i < return_j
        # y_ij = 0  if return_i == return_j
        target_diff = r_i - r_j
        y = torch.sign(target_diff)

        # Filter pairs
        # We ignore the diagonal (i == j) and pairs where returns are identical,
        # as MarginRankingLoss requires labels to be 1 or -1.
        mask = y != 0

        # Extract valid pairs and flatten to 1D
        # Expanding (N, 1) to (N, N) matches the shape of the mask
        N = preds.size(0)
        valid_s_i = s_i.expand(N, N)[mask]
        valid_s_j = s_j.expand(N, N)[mask]
        valid_y = y[mask]

        # Compute the standard Margin Ranking Loss:
        # Loss(x1, x2, y) = max(0, -y * (x1 - x2) + margin)
        loss = self.loss_fn(valid_s_i, valid_s_j, valid_y)

        return loss
