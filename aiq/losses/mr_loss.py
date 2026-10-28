import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginRankingLoss(nn.Module):
    """Rank-distance-weighted pairwise loss for one daily cross-section.

    Global pairs and predicted-TopK-to-rest pairs are averaged separately and
    then combined with a fixed head_weight. Pair weights use cross-sectional
    target-rank distances rather than raw return magnitudes, limiting the
    influence of extreme returns.
    """

    def __init__(
        self,
        margin: float = 0.0,
        epsilon: float = 0.0,
        tau: float = 1.0,
        top_k: int = 30,
        num_pairs: Optional[int] = 4096,
        head_weight: float = 0.5,
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
        if not math.isfinite(head_weight) or not 0.0 <= head_weight <= 1.0:
            raise ValueError("head_weight must be in [0, 1]")

        self.margin = margin
        self.epsilon = epsilon
        self.tau = tau
        self.top_k = top_k
        self.num_pairs = num_pairs
        self.head_weight = head_weight

    @torch.no_grad()
    def _sample_pairs(
        self, preds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = preds.numel()
        if self.num_pairs is None or n * (n - 1) // 2 <= self.num_pairs:
            global_i, global_j = torch.triu_indices(n, n, offset=1, device=preds.device)
        else:
            global_i = torch.randint(n, (self.num_pairs,), device=preds.device)
            global_j = torch.randint(n - 1, (self.num_pairs,), device=preds.device)
            global_j = global_j + (global_j >= global_i).long()

        empty = torch.empty(0, dtype=torch.long, device=preds.device)
        head_i, head_j = empty, empty
        if 0 < self.top_k < n:
            order = torch.argsort(preds, descending=True, stable=True)
            head_i = order[: self.top_k].repeat_interleave(n - self.top_k)
            head_j = order[self.top_k :].repeat(self.top_k)
        return global_i, global_j, head_i, head_j

    def _pair_loss(
        self,
        pair_i: torch.Tensor,
        pair_j: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
        target_ranks: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if pair_i.numel() == 0:
            return None

        diff_r = (targets[pair_i] - targets[pair_j]).detach()
        pair_mask = diff_r.abs() > self.epsilon
        if not pair_mask.any():
            return None

        pair_i = pair_i[pair_mask]
        pair_j = pair_j[pair_mask]
        y = diff_r[pair_mask].sign()

        # Average ranks are unnecessary for the sign, and exact ties are
        # excluded above. Rank distance avoids domination by extreme returns.
        weights = (target_ranks[pair_i] - target_ranks[pair_j]).abs()
        diff_s = preds[pair_i] - preds[pair_j]
        loss = F.softplus((self.margin - y * diff_s) / self.tau)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            return None
        return (weights * loss).sum() / weight_sum

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)
        if preds.numel() != targets.numel():
            raise ValueError("preds and targets must contain the same number of values")

        dtype = (
            torch.float64
            if preds.dtype == torch.float64 or targets.dtype == torch.float64
            else torch.float32
        )
        preds = preds.to(dtype=dtype)
        targets = targets.to(dtype=dtype)

        if not torch.isfinite(preds).all():
            bad_count = (~torch.isfinite(preds)).sum().item()
            raise FloatingPointError(
                f"Non-finite model predictions detected: {bad_count} values"
            )

        valid = torch.isfinite(targets)
        preds = preds[valid]
        targets = targets[valid]

        n = preds.numel()
        if n <= 1:
            return preds.sum() * 0.0
        if torch.all(targets == targets[0]):
            return preds.sum() * 0.0

        global_i, global_j, head_i, head_j = self._sample_pairs(preds)

        # Cross-sectional ranks in [0, 1]. Exact target ties receive different
        # ordinal ranks but their pairs are excluded by diff_r == 0.
        order = torch.argsort(targets, stable=True)
        target_ranks = torch.empty_like(targets)
        target_ranks[order] = torch.arange(
            n, device=targets.device, dtype=targets.dtype
        ) / float(n - 1)

        centered_preds = preds - preds.mean()
        if n > 2:
            scale = centered_preds.detach().abs().max().clamp_min(1.0)
            centered_preds = centered_preds / scale
            pred_var = centered_preds.square().mean()
            preds = centered_preds / pred_var.clamp_min(1e-8).sqrt()
        else:
            preds = centered_preds

        global_loss = self._pair_loss(global_i, global_j, preds, targets, target_ranks)
        head_loss = self._pair_loss(head_i, head_j, preds, targets, target_ranks)

        if global_loss is None and head_loss is None:
            return preds.sum() * 0.0
        if global_loss is None:
            return head_loss
        if head_loss is None:
            return global_loss
        return (1.0 - self.head_weight) * global_loss + self.head_weight * head_loss
