import torch
import torch.nn as nn


class MSERankLoss(nn.Module):
    def __init__(self, alpha=4.0, margin=1.0):
        super(MSERankLoss, self).__init__()
        self.alpha = alpha
        self.margin = margin
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

        # 排序损失（Pairwise）
        rows, cols = torch.triu_indices(N, N, offset=1)
        diff_pred = pred[rows] - pred[cols]  # L = N(N-1)/2
        diff_target = target[rows] - target[cols]

        mask = (diff_target != 0)
        pairwise_loss = torch.relu(self.margin - diff_pred[mask] * torch.sign(diff_target[mask])).mean()

        total_loss = regression_loss + self.alpha * pairwise_loss
        return total_loss
