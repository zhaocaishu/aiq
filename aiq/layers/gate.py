import torch
from torch import nn


class GateNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        output_dim=None,
        dropout_rate=0.0,
        batch_norm=False,
    ):
        super(GateNN, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        gate_layers = [nn.Linear(input_dim, hidden_dim)]
        if batch_norm:
            gate_layers.append(nn.BatchNorm1d(hidden_dim))
        gate_layers.append(nn.ReLU())
        if dropout_rate > 0:
            gate_layers.append(nn.Dropout(dropout_rate))
        gate_layers.append(nn.Linear(hidden_dim, output_dim))
        gate_layers.append(nn.Sigmoid())
        self.gate = nn.Sequential(*gate_layers)

    def forward(self, inputs):
        return self.gate(inputs) * 2


class GatedFusion(nn.Module):
    def __init__(self, input_dim):
        super(GatedFusion, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.gate_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x1, x2):
        h1 = self.linear1(x1)
        h2 = self.linear2(x2)
        gate_input = h1 + h2
        gate = torch.sigmoid(self.gate_linear(gate_input))
        fused = gate * x1 + (1 - gate) * x2
        return fused
