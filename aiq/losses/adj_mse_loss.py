import torch
import torch.nn as nn


class AdjMSELoss2(nn.Module):
    def __init__(self, beta=2.5):
        super(AdjMSELoss2, self).__init__()
        self.beta = beta
               
    def forward(self, outputs, labels):
        outputs = torch.squeeze(outputs)
        loss = (outputs - labels)**2
        adj_loss = self.beta - (self.beta - 0.5) / (1 + torch.exp(10000 * torch.mul(outputs, labels)))
        loss = self.beta * loss / (1 + adj_loss)
        return torch.mean(loss)
