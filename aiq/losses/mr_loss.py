import torch
import torch.nn as nn


class MSERankLoss(nn.Module):
    def __init__(self, alpha=1.0):
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

        # 回归损失（MSE）
        regression_loss = self.mse_loss(pred, target)

        # 排序损失（Pairwise）
        diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)  # (N, N)
        diff_target = target.unsqueeze(1) - target.unsqueeze(0)  # (N, N)

        pairwise_loss = torch.relu(-diff_pred * diff_target).mean()

        total_loss = regression_loss + self.alpha * pairwise_loss
        return total_loss
