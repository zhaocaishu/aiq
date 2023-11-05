import torch
import torch.nn as nn


class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, preds, targets):
        assert not torch.any(torch.isnan(preds))
        assert not torch.any(torch.isnan(targets))
        preds_mean = preds.mean()
        targets_mean = targets.mean()
        covariance = (preds - preds_mean) * (targets - targets_mean)
        preds_var = torch.square(preds - preds_mean).mean()
        targets_var = torch.square(targets - targets_mean).mean()
        ccc = 2.0 * covariance / (preds_var + targets_var + torch.square(preds_mean - targets_mean) + 1e-5)
        ccc = 1.0 - ccc.mean()
        return ccc


if __name__ == '__main__':
    preds = torch.randn(4)
    targets = torch.randn(4)

    ccc_loss = CCCLoss()
    print(ccc_loss(preds, targets))
