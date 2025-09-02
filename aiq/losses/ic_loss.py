import torch
import torch.nn as nn


class ICLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(ICLoss, self).__init__()
        self.eps = eps

    def forward(self, preds, targets):
        """
        preds: Tensor of shape (N, 1) - predicted values
        targets: Tensor of shape (N, 1) - true values
        """
        preds = preds.view(-1)
        targets = targets.view(-1)

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
