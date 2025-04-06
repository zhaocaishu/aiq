import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedLoss(nn.Module):
    def __init__(self, count_per_class, beta=0.9999):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta

        # 计算有效样本数
        effective_num = (1.0 - torch.pow(beta, count_per_class)) / (1.0 - beta)
        # 计算归一化权重（和为1）
        weights = 1.0 / effective_num
        self.weights = (
            weights / torch.sum(weights) * len(count_per_class)
        )  # 缩放至平均权重为1

    def forward(self, logits, targets):
        # 确保权重在相同设备上
        self.weights = self.weights.to(logits.device)

        loss = F.cross_entropy(logits, targets, reduction="none")

        # 根据目标类别获取权重
        weights_per_sample = self.weights[targets]
        # 加权损失
        weighted_loss = loss * weights_per_sample
        return weighted_loss.mean()
