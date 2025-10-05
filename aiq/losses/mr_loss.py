import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLoss(nn.Module):
    """
    Symmetric boundary pairwise loss for optimizing Precision@K.
    Focuses on pairs across predicted top-k vs rest, penalizing both weak separations
    and wrong orders. Uses log-sigmoid for stability.

    Loss includes:
    - For target_i > target_j: -w * log(sigmoid(sigma * (s_i - s_j)))
    - For target_i < target_j: -w * log(sigmoid(sigma * (s_j - s_i)))

    Args:
        top_k: Number of top items to emphasize.
        sigma: Scaling factor for score differences.
    """

    def __init__(
        self,
        top_k: int,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.top_k = top_k
        self.sigma = sigma

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the symmetric boundary ranking loss.

        Args:
            preds: Predicted scores, shape (N, 1) or (N,)
            targets: True returns, shape (N, 1) or (N,)

        Returns:
            Scalar loss (mean over selected pairs).
        """
        preds = preds.squeeze(-1)
        targets = targets.squeeze(-1)
        N = preds.shape[0]

        if self.top_k <= 0 or self.top_k >= N:
            return torch.tensor(0.0, device=preds.device)

        # Sort by predicted score descending
        preds_sorted, idx = torch.sort(preds, descending=True)
        targets_sorted = targets[idx]

        # Slice into top-k and bottom
        top_preds = preds_sorted[: self.top_k]
        bottom_preds = preds_sorted[self.top_k :]
        top_targets = targets_sorted[: self.top_k]
        bottom_targets = targets_sorted[self.top_k :]

        # Compute differences for boundary pairs only: i in top-k, j in bottom
        score_diffs = top_preds.unsqueeze(1) - bottom_preds.unsqueeze(
            0
        )  # (top_k, N - top_k)
        return_diffs = top_targets.unsqueeze(1) - bottom_targets.unsqueeze(
            0
        )  # (top_k, N - top_k)

        # Weights
        w = torch.log1p(torch.abs(return_diffs))

        # Positive case: target_i > target_j, penalize if s_i <= s_j (but since sorted, s_i >= s_j)
        true_pos_mask = return_diffs > 0
        losses_pos = torch.tensor(0.0, device=preds.device)
        if true_pos_mask.any():
            logits_pos = self.sigma * score_diffs[true_pos_mask]
            losses_pos = -(w[true_pos_mask] * F.logsigmoid(logits_pos)).mean()

        # Negative case: target_i < target_j, penalize if s_i > s_j
        true_neg_mask = return_diffs < 0
        losses_neg = torch.tensor(0.0, device=preds.device)
        if true_neg_mask.any():
            logits_neg = self.sigma * (
                -score_diffs[true_neg_mask]
            )  # sigma * (s_j - s_i)
            losses_neg = -(w[true_neg_mask] * F.logsigmoid(logits_neg)).mean()

        # Total loss: average of non-zero terms
        if (losses_pos.item() > 0.0) and (losses_neg.item() > 0.0):
            total_loss = 0.5 * (losses_pos + losses_neg)
        else:
            total_loss = losses_pos + losses_neg

        return total_loss


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
