import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginRankingLoss(nn.Module):
    """
    Return-Weighted Pairwise Ranking Loss for Stock Selection.

    This loss encourages higher-return stocks to receive higher scores using
    a smooth ranking penalty weighted by the absolute return difference.
    Scores are standardized across valid stocks to control their scale. With
    only two valid stocks, scores are centered without scaling to retain a
    useful ranking gradient.

    Args:
        margin (float): Soft margin in standardized score units, or raw score
            units when only two stocks are valid.
        epsilon (float): Minimum absolute target difference to include a pair,
            in the supplied targets' units. Defaults to zero.
        tau (float): Positive temperature of the smooth ranking penalty.
        top_k (int): Number of predicted leaders compared with every remaining
            stock. Zero disables these additional head comparisons.
        num_pairs (int, optional): Number of global candidate pairs sampled
            with replacement. Use all unique global pairs when this is None
            or the batch has fewer pairs. TopK-to-rest pairs are added on top
            of this budget, including overlaps with the global pairs, to
            emphasize the predicted leaders.
    """

    def __init__(
        self,
        margin: float = 0.0,
        epsilon: float = 0.0,
        tau: float = 1.0,
        top_k: int = 30,
        num_pairs: Optional[int] = 4096,
    ):
        super().__init__()
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("margin must be finite and non-negative")
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and non-negative")
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and greater than zero")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if num_pairs is not None and (
            isinstance(num_pairs, bool)
            or not isinstance(num_pairs, int)
            or num_pairs <= 0
        ):
            raise ValueError("num_pairs must be a positive integer or None")

        self.margin = margin
        self.epsilon = epsilon
        self.tau = tau
        self.top_k = top_k
        self.num_pairs = num_pairs

    @torch.no_grad()
    def _sample_pairs(self, preds: torch.Tensor) -> tuple:
        """Combine global comparisons with all predicted TopK-to-rest pairs."""
        N = preds.numel()
        if self.num_pairs is None or N * (N - 1) // 2 <= self.num_pairs:
            pair_i, pair_j = torch.triu_indices(N, N, offset=1, device=preds.device)
        else:
            pair_i = torch.randint(N, (self.num_pairs,), device=preds.device)
            pair_j = torch.randint(N - 1, (self.num_pairs,), device=preds.device)
            pair_j = pair_j + (pair_j >= pair_i).long()

        if 0 < self.top_k < N:
            order = torch.argsort(preds, descending=True, stable=True)
            head_i = order[: self.top_k].repeat_interleave(N - self.top_k)
            head_j = order[self.top_k :].repeat(self.top_k)
            pair_i = torch.cat([pair_i, head_i])
            pair_j = torch.cat([pair_j, head_j])
        return pair_i, pair_j

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the return-weighted smooth loss for one daily cross section.

        Args:
            preds (torch.Tensor): Predicted scores of shape (N, 1).
            targets (torch.Tensor): Ground truth returns of shape (N, 1).

        Returns:
            torch.Tensor: Scalar weighted mean over valid candidate pairs.
            Non-finite samples are ignored. Fewer than two valid stocks or
            no target differences above epsilon produce a connected zero.
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

        valid_mask = torch.isfinite(preds) & torch.isfinite(targets)
        preds = preds[valid_mask]
        targets = targets[valid_mask]

        N = preds.size(0)
        if N <= 1:
            return preds.sum() * 0.0
        if torch.all(targets == targets[0]):
            return preds.sum() * 0.0

        # Pair selection depends only on predictions, without index gradients.
        pair_i, pair_j = self._sample_pairs(preds)
        diff_r = (targets[pair_i] - targets[pair_j]).detach()
        weights = diff_r.abs()
        mask = weights > self.epsilon
        if not mask.any():
            return preds.sum() * 0.0

        # Two-point standardization would make every nonzero score gap fixed.
        centered_preds = preds - preds.mean()
        if N > 2:
            # Rescale before squaring to avoid overflow for large score gaps.
            scale = centered_preds.detach().abs().max().clamp_min(1.0)
            centered_preds = centered_preds / scale
            # Clamp before sqrt to keep constant-score gradients finite.
            pred_var = centered_preds.square().mean()
            preds = centered_preds / pred_var.clamp_min(1e-8).sqrt()
        else:
            preds = centered_preds

        y = diff_r[mask].sign()
        diff_s = preds[pair_i[mask]] - preds[pair_j[mask]]
        loss = F.softplus((self.margin - y * diff_s) / self.tau)

        # Rescaling all weights preserves their weighted mean and its units.
        weights = weights[mask]
        weights = weights / weights.max()
        return (weights * loss).sum() / weights.sum()
