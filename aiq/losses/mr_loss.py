import torch
import torch.nn as nn
import torch.nn.functional as F


class MSERankLoss(nn.Module):
    """
    多头选股损失函数：
    - MSE回归损失：确保预测值接近真实收益率。
    - Pairwise排序损失：强化模型在top-k正样本与难负样本之间的区分能力。

    Args:
        alpha (float): MSE损失权重
        beta (float): Pairwise损失权重
        top_k_ratio (float): 正样本比例 (e.g., 0.2 表示前20%作为正样本)
        margin (float): Pairwise损失的hinge margin
        hard_neg_ratio (float): 难负样本比例，相对正样本k数量
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        top_k_ratio: float = 0.1,
        margin: float = 1.0,
        hard_neg_ratio: float = 0.7,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.top_k_ratio = top_k_ratio
        self.margin = margin
        self.hard_neg_ratio = hard_neg_ratio

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算总损失: alpha * MSE + beta * Pairwise
        """
        preds = preds.flatten()
        targets = targets.flatten()

        # MSE 损失 (scalar)
        mse_loss = F.mse_loss(preds, targets, reduction="mean")

        # Pairwise 损失 (scalar)
        pairwise_loss = self._compute_pairwise_loss(preds, targets)

        # 总损失 (scalar)
        total_loss = self.alpha * mse_loss + self.beta * pairwise_loss
        return total_loss

    def _compute_pairwise_loss(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Pairwise排序损失:
        - 正样本: 真实收益率最高的 top-k
        - 难负样本: 非正样本中预测值相对真实值被高估最多的 hard_k
        - Loss = mean(max(0, neg_pred - pos_pred + margin))
        """
        batch_size = preds.size(0)
        k = max(1, int(batch_size * self.top_k_ratio))
        hard_k = max(1, int(k * self.hard_neg_ratio))

        pos_mask, neg_mask = self._select_samples(preds, targets, k, hard_k)

        if not pos_mask.any() or not neg_mask.any():
            return preds.new_tensor(0.0)

        pos_preds = preds[pos_mask]  # [num_pos]
        neg_preds = preds[neg_mask]  # [num_neg]

        # 广播计算 pairwise hinge loss
        diff = neg_preds.unsqueeze(0) - pos_preds.unsqueeze(1) + self.margin
        loss = F.relu(diff).mean()

        return loss

    def _select_samples(
        self, preds: torch.Tensor, targets: torch.Tensor, k: int, hard_k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        根据真实值选择 top-k 正样本，并通过难负样本挖掘选 hard_k 个负样本。
        """
        batch_size = targets.size(0)

        # Top-k 正样本（真实值最高）
        _, topk_idx = torch.topk(targets, k, largest=True, sorted=False)
        pos_mask = torch.zeros(batch_size, dtype=torch.bool, device=targets.device)
        pos_mask[topk_idx] = True

        # 非正样本
        non_pos_mask = ~pos_mask
        non_pos_count = non_pos_mask.sum().item()
        if non_pos_count < hard_k:
            neg_mask = torch.zeros(batch_size, dtype=torch.bool, device=targets.device)
            return pos_mask, neg_mask

        # error_score = 被高估程度 (预测高于真实)
        error_score = (preds - targets)[non_pos_mask]
        non_pos_idx = non_pos_mask.nonzero(as_tuple=True)[0]

        # 选择 error_score 最大的 hard_k 作为负样本
        _, hard_neg_idx = torch.topk(error_score, hard_k, largest=True, sorted=False)
        neg_mask = torch.zeros(batch_size, dtype=torch.bool, device=targets.device)
        neg_mask[non_pos_idx[hard_neg_idx]] = True

        return pos_mask, neg_mask
