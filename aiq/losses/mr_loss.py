import torch
import torch.nn as nn
import torch.nn.functional as F


class MSERankLoss(nn.Module):
    def __init__(self, alpha=3.0, eps=1e-8):
        super(MSERankLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def _ic_loss(self, preds: torch.Tensor, targets: torch.Tensor):
        # 中心化
        p_mean = preds.mean()
        t_mean = targets.mean()
        p_centered = preds - p_mean
        t_centered = targets - t_mean

        # 协方差
        cov = (p_centered * t_centered).sum()

        # 标准差
        p_var = (p_centered.pow(2)).sum()
        t_var = (t_centered.pow(2)).sum()
        denom = torch.sqrt(p_var * t_var + self.eps)

        ic = cov / denom
        loss = -ic
        return loss

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds (Tensor): shape (N, 1), 预测收益率 \hat{r}_{i,t}
            targets (Tensor): shape (N, 1), 真实收益率 r_{i,t}
        """
        # 去掉多余维度 => (N,)
        preds = preds.view(-1)
        targets = targets.view(-1)

        # 回归损失（MSE）
        reg_loss = F.mse_loss(preds, targets)

        # 排序损失（IC)
        ic_loss = self._ic_loss(preds, targets)

        loss = reg_loss + self.alpha * ic_loss
        return loss
