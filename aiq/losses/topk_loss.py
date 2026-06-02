import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKLoss(nn.Module):
    def __init__(self, top_k=30, tau=1.0):
        super(TopKLoss, self).__init__()
        self.top_k = top_k
        self.tau = tau

    def _dftopk(self, x, k, tau):
        x_k, _ = torch.kthvalue(x=-x, k=k, dim=1)
        x_k_plus_1, _ = torch.kthvalue(x=-x, k=k + 1, dim=1)
        threshold = ((x_k + x_k_plus_1) / 2.0).unsqueeze(1)
        logits = x + threshold
        logits = logits / tau
        return logits

    def _topk_label(self, scores, k):
        """
        scores: shape [B, N]
        k: int
        return: mask of top-k scores, shape [B, N]
        """
        topk_indices = torch.topk(scores, k=k, dim=1).indices
        mask = torch.zeros_like(scores, dtype=torch.float32)
        mask.scatter_(dim=1, index=topk_indices, value=1)
        return mask

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predicted scores, shape (N, 1)
            targets (torch.Tensor): Ground truth values, shape (N, 1)
        Returns:
            torch.Tensor: A scalar tensor representing the margin ranking loss
        """
        preds = preds.transpose(0, 1)
        targets = targets.transpose(0, 1)
        N = preds.size(1)

        logits = self._dftopk(preds, self.top_k, self.tau)
        labels = self._topk_label(targets.float(), self.top_k)

        pos_weight = torch.sqrt(
            torch.tensor([(N - self.top_k) / self.top_k], device=preds.device)
        )
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight, reduction="mean"
        )
        return loss
