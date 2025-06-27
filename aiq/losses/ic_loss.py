import torch
import torch.nn as nn


class ICLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(ICLoss, self).__init__()
        self.eps = eps

    def forward(self, preds, targets):
        """
        preds: Tensor of shape (N, 1) - predicted values
        targets: Tensor of shape (N, 1) - true values
        """
        preds = preds.view(-1)
        targets = targets.view(-1)

        preds_std = (preds - preds.mean()) / (preds.std() + self.eps)
        targets_std = (targets - targets.mean()) / (targets.std() + self.eps)

        ic = torch.mean(preds_std * targets_std)
        loss = -ic  # maximize IC = minimize -IC
        return loss
