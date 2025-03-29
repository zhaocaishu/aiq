import torch

class ReturnWeightedMSELoss(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred, target, returns):
        weights = torch.abs(returns) ** self.alpha
        loss = torch.mean(weights * (pred - target) ** 2)
        return loss