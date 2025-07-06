import torch
import torch.nn as nn
import torch.nn.functional as F


class MSERankLoss(nn.Module):
    def __init__(self, alpha=3.0, min_diff=0.1):
        super(MSERankLoss, self).__init__()
        self.alpha = alpha
        self.min_diff = min_diff
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (Tensor): shape (N, 1), 预测收益率 \hat{r}_{i,t}
            target (Tensor): shape (N, 1), 真实收益率 r_{i,t}
        """
        # 去掉多余维度 => (N,)
        pred = pred.view(-1)
        target = target.view(-1)
        N = pred.size(0)

        # 回归损失（MSE）
        regression_loss = self.mse_loss(pred, target)

        # 生成所有股票对索引 (i, j)，其中 i < j
        rows, cols = torch.triu_indices(N, N, offset=1)
        pred_i, pred_j = pred[rows], pred[cols]
        target_i, target_j = target[rows], target[cols]

        # 计算目标差异和置信度
        diff_target = target_i - target_j
        confidence = torch.abs(diff_target).detach()  # 差异绝对值作为置信度
        mask = confidence > self.min_diff  # 过滤微弱差异

        # 若无有效股票对，直接返回MSE损失
        if not torch.any(mask):
            return regression_loss

        # 计算加权BPR损失
        bpr_diff = F.logsigmoid((pred_i - pred_j) * torch.sign(diff_target.detach()))
        pairwise_loss = -confidence * bpr_diff

        # 应用掩码并求平均
        pairwise_loss = pairwise_loss[mask].mean()
        total_loss = regression_loss + self.alpha * pairwise_loss
        return total_loss
