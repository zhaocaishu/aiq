import torch
import torch.nn as nn
import torch.nn.functional as F


class MSERankLoss(nn.Module):
    def __init__(
        self,
        alpha=1.0,
        beta=2.0,
        top_k_ratio=0.1,
        margin=1.0,
        reduction="mean",
        hard_neg_ratio=0.7,
    ):
        """
        多头选股损失函数，结合MSE回归损失和Pairwise排序损失。
        Pairwise损失聚焦于鼓励模型正确排序极端样本（top-k多头 vs hard negative mining的负样本）。

        Args:
            alpha: MSE损失的权重
            beta: Pairwise损失的权重
            top_k_ratio: 正样本比例 (top-k的比例，e.g., 0.2表示前20%作为正样本)
            margin: Pairwise损失的边界值（hinge loss的margin）
            reduction: 损失归约方式 ('mean', 'sum', 'none')，应用于最终总损失
            hard_neg_ratio: 难负样本挖掘的比例，相对于k (e.g., 1.0表示选择k个难负样本，0.5表示0.5*k)
        """
        super(MSERankLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.top_k_ratio = top_k_ratio
        self.margin = margin
        self.reduction = reduction
        self.hard_neg_ratio = hard_neg_ratio

    def forward(self, preds, targets):
        """
        计算总损失：alpha * MSE + beta * Pairwise

        Args:
            preds: 预测收益率, shape (n,) 或 (n,1)
            targets: 真实收益率, shape (n,) 或 (n,1)

        Returns:
            total_loss: 根据reduction归约的损失（scalar或tensor）
        """
        # 确保输入是2D的 [batch_size, 1]
        if preds.dim() == 1:
            preds = preds.unsqueeze(1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        # 计算MSE损失
        mse_loss = F.mse_loss(preds, targets, reduction="none")  # [batch_size, 1]

        # 计算Pairwise损失（scalar）
        pairwise_loss = self._compute_pairwise_loss(
            preds.squeeze(1), targets.squeeze(1)
        )

        # 组合损失
        total_loss = self.alpha * mse_loss + self.beta * pairwise_loss

        # 应用reduction
        if self.reduction == "mean":
            total_loss = total_loss.mean()
        elif self.reduction == "sum":
            total_loss = total_loss.sum()
        elif self.reduction == "none":
            pass  # 返回 [batch_size, 1]
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

        return total_loss

    def _compute_pairwise_loss(self, preds, targets):
        """
        计算Pairwise排序损失，基于top-k正样本和hard negative mining的负样本。
        鼓励正样本的预测收益率 > 负样本的预测收益率 + margin。

        Args:
            preds: 预测收益率, shape (batch_size,)
            targets: 真实收益率, shape (batch_size,)

        Returns:
            pairwise_loss: 平均pairwise hinge loss (scalar)
        """
        batch_size = preds.shape[0]
        k = max(1, int(batch_size * self.top_k_ratio))  # top-k 正样本数量
        hard_k = max(1, int(k * self.hard_neg_ratio))  # 难负样本数量

        # 选择正样本（top-k）和负样本（hard negatives）
        positive_mask, negative_mask = self._select_samples(preds, targets, k, hard_k)

        # 如果没有足够的正样本或负样本，返回0损失
        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            return torch.tensor(0.0, device=preds.device)

        # 获取正样本和负样本的预测收益率
        pos_preds = preds[positive_mask]  # [num_pos]
        neg_preds = preds[negative_mask]  # [num_neg]

        # 计算所有正负样本对的差异
        num_pos = pos_preds.size(0)
        num_neg = neg_preds.size(0)

        # 扩展维度以便广播
        pos_expanded = pos_preds.unsqueeze(1).expand(
            num_pos, num_neg
        )  # [num_pos, num_neg]
        neg_expanded = neg_preds.unsqueeze(0).expand(
            num_pos, num_neg
        )  # [num_pos, num_neg]

        # 计算pairwise hinge loss: max(0, neg - pos + margin)
        pair_diff = neg_expanded - pos_expanded + self.margin
        pair_loss = F.relu(pair_diff)

        # 平均所有对的损失
        pairwise_loss = pair_loss.mean()

        return pairwise_loss

    def _select_samples(self, preds, targets, k, hard_k):
        """
        根据真实收益率选择正样本（top-k最高收益率），并使用难样本挖掘选择负样本。
        难样本挖掘：从所有非正样本中，选择预测收益率最高的hard_k个作为负样本。
        这提升了模型的学习能力，聚焦于模型最容易混淆的负样本（被高估的负样本）。

        Args:
            preds: 预测收益率, shape (batch_size,)
            targets: 真实收益率, shape (batch_size,)
            k: 正样本数量
            hard_k: 难负样本数量

        Returns:
            positive_mask: 正样本mask, shape (batch_size,)
            negative_mask: 负样本mask, shape (batch_size,)
        """
        batch_size = targets.size(0)

        # 选择top-k作为正样本（最高收益率）
        _, topk_indices = torch.topk(targets, k, largest=True, sorted=True)
        positive_mask = torch.zeros(batch_size, dtype=torch.bool, device=targets.device)
        positive_mask[topk_indices] = True

        # 潜在负样本：所有非正样本
        non_positive_mask = ~positive_mask

        # 如果潜在负样本不足hard_k，返回空负样本
        if non_positive_mask.sum() < hard_k:
            negative_mask = torch.zeros(
                batch_size, dtype=torch.bool, device=targets.device
            )
            return positive_mask, negative_mask

        # 从非正样本中提取preds和indices
        non_pos_preds = preds[non_positive_mask]
        non_pos_indices = torch.nonzero(non_positive_mask).squeeze(1)

        # 选择预测收益率最高的hard_k个作为难负样本
        _, hard_neg_indices_in_non_pos = torch.topk(
            non_pos_preds, hard_k, largest=True, sorted=True
        )
        hard_neg_indices = non_pos_indices[hard_neg_indices_in_non_pos]

        # 创建负样本mask
        negative_mask = torch.zeros(batch_size, dtype=torch.bool, device=targets.device)
        negative_mask[hard_neg_indices] = True

        return positive_mask, negative_mask
