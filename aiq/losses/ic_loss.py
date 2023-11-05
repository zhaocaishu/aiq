import torch
import torch.nn as nn


class ICLoss(nn.Module):
    def __init__(self):
        super(ICLoss, self).__init__()

    def forward(self, preds, targets):
        assert not torch.any(torch.isnan(preds))
        assert not torch.any(torch.isnan(targets))
        preds = preds - preds.mean()
        targets = targets - targets.mean()
        ic = torch.sum(preds * targets) / (
                    torch.sqrt(torch.sum(preds ** 2)) * torch.sqrt(torch.sum(targets ** 2)) + 1e-5)
        ic = 1.0 - ic
        return ic


if __name__ == '__main__':
    preds = torch.randn(4)
    targets = torch.randn(4)

    ic_loss = ICLoss()
    print(ic_loss(preds, targets))
