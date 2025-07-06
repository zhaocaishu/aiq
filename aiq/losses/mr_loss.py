import torch
import torch.nn as nn


class MSERankLoss(nn.Module):
    def __init__(self, alpha=4.0):
        super(MSERankLoss, self).__init__()
        self.alpha = alpha
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
        pred_i = pred[rows]
        pred_j = pred[cols]
        target_i = target[rows]
        target_j = target[cols]

        # 计算目标收益差异并创建掩码
        diff_target = target_i - target_j
        mask = diff_target != 0  # 只处理收益不同的股票对

        # 计算BPR Loss
        # 核心思想：若股票i收益高于j，则鼓励pred_i > pred_j；反之鼓励pred_j > pred_i
        bpr_diff = (pred_i - pred_j) * torch.sign(diff_target.detach())
        pairwise_loss = -torch.log(torch.sigmoid(bpr_diff) + 1e-8)  # 加1e-8防止数值溢出

        # 应用掩码并求平均
        pairwise_loss = pairwise_loss[mask].mean()

        total_loss = regression_loss + self.alpha * pairwise_loss
        return total_loss
