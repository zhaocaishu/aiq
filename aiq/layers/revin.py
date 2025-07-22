import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(
        self, num_features: int, eps=1e-5, affine=False, clamp_bounds: tuple = None
    ):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        :param clamp_bounds: optional (min, max) tuple to clip normalized values
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        # clamp_bounds should be (min, max) or None
        self.clamp_bounds = clamp_bounds

        if self.affine:
            self._init_params()

    def forward(self, x):
        self._get_statistics(x)
        x = self._normalize(x)
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=(0, 1), keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=(0, 1), keepdim=True, unbiased=False) + self.eps
        ).detach()
        print(self.mean.shape, self.stdev.shape)

    def _normalize(self, x):
        # subtract mean, divide by std
        x = (x - self.mean) / self.stdev

        # optional clamp
        if self.clamp_bounds is not None:
            min_val, max_val = self.clamp_bounds
            x = torch.clamp(x, min=min_val, max=max_val)

        # apply learnable affine after clamp
        if self.affine:
            x = x * self.affine_weight.view(1, 1, -1)
            x = x + self.affine_bias.view(1, 1, -1)
        return x


if __name__ == "__main__":
    x_in = torch.randn((128, 8, 32))
    # Example: clip normalized outputs between -3 and 3
    revin_layer = RevIN(32, affine=True, clamp_bounds=(-3.0, 3.0))
    x_out = revin_layer(x_in)
    print(x_out.shape)  # torch.Size([128, 8, 32])
