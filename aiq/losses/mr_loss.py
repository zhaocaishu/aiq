import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginRankingLoss(nn.Module):
    """Return-gap-weighted RankNet for one date, with z-scored labels.

    Inputs: [N] or [N, 1]. Missing labels are excluded. Pairs touching true
    Top-K or predicted Top-K participate, including cutoff ties.
    epsilon/max_label_gap use label units; margin/tau use score units.
    This is a smooth logistic loss; reaching margin does not give zero loss.
    """

    def __init__(
        self,
        margin: float = 0.0,
        epsilon: float = 0.05,
        true_top_k: int = 30,
        pred_top_k: int = 60,
        tau: float = 1.0,
        max_label_gap: float = 2.0,
    ):
        super().__init__()
        self.margin = margin
        self.epsilon = epsilon
        self.true_top_k = true_top_k
        self.pred_top_k = pred_top_k
        self.tau = tau
        self.max_label_gap = max_label_gap

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.reshape(-1)
        targets = targets.detach().reshape(-1)
        valid = torch.isfinite(targets)
        preds, targets = preds[valid], targets[valid]
        if preds.dtype in (torch.float16, torch.bfloat16):
            preds = preds.float()
        targets = targets.to(dtype=preds.dtype)
        N = preds.numel()
        if N < 2:
            return preds[:0].sum()

        head = torch.zeros(N, dtype=torch.bool, device=preds.device)
        for values, k in (
            (targets, self.true_top_k),
            (preds.detach(), self.pred_top_k),
        ):
            if k > 0:
                cutoff = values.topk(min(k, N)).values[-1]
                head |= values >= cutoff

        diff_r = targets[:, None] - targets[None, :]
        # i is the better-return stock; self-pairs and near ties are excluded.
        mask = (diff_r > self.epsilon) & (head[:, None] | head[None, :])
        if not mask.any():
            return preds[:0].sum()

        diff_s = (preds[:, None] - preds[None, :])[mask]
        weights = diff_r[mask].clamp(max=self.max_label_gap)
        losses = F.softplus((self.margin - diff_s) / self.tau)
        return (weights * losses).sum() / weights.sum().clamp_min(
            torch.finfo(preds.dtype).tiny
        )
