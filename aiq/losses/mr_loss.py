import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLoss(nn.Module):
    """
    LambdaLoss for stock selection (single column input: preds and targets),
    optimized for continuous returns.
    Weights are set as absolute return differences to prioritize high-return pairs.
    """

    def __init__(self, top_k: int = None, sigma: float = 1.0, eps: float = 1e-10):
        """
        :param k: top-k cutoff for ranking metrics (if None, use all)
        :param sigma: score difference scaling factor
        :param eps: epsilon for numerical stability
        """
        super().__init__()
        self.top_k = top_k
        self.sigma = sigma
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        :param preds: predicted scores, shape (N, 1)
        :param targets: true returns, shape (N, 1)
        """
        preds = preds.squeeze(-1)
        targets = targets.squeeze(-1)
        N = preds.shape[0]

        preds_sorted, indices_pred = torch.sort(preds, descending=True)
        targets_sorted = targets[indices_pred]

        return_diffs = targets_sorted[:, None] - targets_sorted[None, :]
        pair_mask = return_diffs > 0
        weights = torch.abs(return_diffs)

        k = self.top_k or N
        topk_mask = torch.zeros((N, N), dtype=torch.bool, device=preds.device)
        topk_mask[:k, :k] = 1
        final_mask = pair_mask & topk_mask

        score_diffs = preds_sorted[:, None] - preds_sorted[None, :]
        score_diffs = score_diffs.clamp(min=-1e8, max=1e8)

        probas = torch.sigmoid(self.sigma * score_diffs).clamp(min=self.eps)
        weighted_probas = (probas**weights).clamp(min=self.eps)

        losses = torch.log2(weighted_probas)[final_mask]

        return -losses.mean()


class MSELambdaLoss(nn.Module):
    """
    组合损失函数: MSE + LambdaLoss
    alpha: MSE 权重
    beta: LambdaLoss 权重
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        top_k: int = None,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_loss = LambdaLoss(top_k=top_k, sigma=sigma)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # MSE损失
        mse_loss = F.mse_loss(preds, targets, reduction="mean")
        # Lambda排序损失
        lambda_loss = self.lambda_loss(preds, targets)

        # 总损失
        total_loss = self.alpha * mse_loss + self.beta * lambda_loss
        return total_loss
