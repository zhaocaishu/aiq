import torch
import torch.nn as nn
import torch.nn.functional as F


class MSERankLoss(nn.Module):
    def __init__(self, alpha=2.0, top_p=0.3, eps=1e-8):
        """
        Args:
            alpha: 排序损失的权重
            top_p: 只对收益率最高的前 p 比例样本计算 MSE 损失 (0.0-1.0)
            eps: 数值稳定性的小常数
        """
        super(MSERankLoss, self).__init__()
        self.alpha = alpha
        self.top_p = top_p
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

        # 回归损失（MSE） - 只对 top_p 比例的样本计算
        if self.top_p < 1.0:
            # 计算需要保留的样本数
            k = int(self.top_p * len(targets))

            # 按目标值排序（不使用绝对值），选择最大的 k 个
            _, indices = torch.topk(targets, k)

            # 选择 top k 样本
            top_preds = preds[indices]
            top_targets = targets[indices]

            reg_loss = F.mse_loss(top_preds, top_targets)
        else:
            # 使用所有样本计算 MSE
            reg_loss = F.mse_loss(preds, targets)

        # 排序损失（IC) - 仍然使用所有样本
        ic_loss = self._ic_loss(preds, targets)

        loss = reg_loss + self.alpha * ic_loss
        return loss
