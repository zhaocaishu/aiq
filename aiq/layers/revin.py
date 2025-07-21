import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        """
        :param num_features: 特征维度 D
        :param eps: 数值稳定项
        :param affine: 是否在最后加可学习的仿射变换
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            # 在最后阶段统一做 affine
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias   = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入 x: (N, T, D)
        1) 对每个样本 i 在时间轴上做标准化
        2) 对每个时间步 t 在样本维度上做标准化
        3) （可选）仿射变换
        """
        # —— 第一阶段：样本内标准化 —— #
        #  compute mean & std over time dim (T)
        #  结果形状 (N, 1, D)，可直接广播到 (N, T, D)
        mean_inst = x.mean(dim=1, keepdim=True)
        var_inst  = x.var(dim=1, keepdim=True, unbiased=False)
        std_inst  = torch.sqrt(var_inst + self.eps)

        x_inst_norm = (x - mean_inst) / std_inst  # shape (N, T, D)

        # —— 第二阶段：样本间标准化 —— #
        #  compute mean & std over batch dim (N)
        #  结果形状 (1, T, D)
        mean_batch = x_inst_norm.mean(dim=0, keepdim=True)
        var_batch  = x_inst_norm.var(dim=0, keepdim=True, unbiased=False)
        std_batch  = torch.sqrt(var_batch + self.eps)

        x_norm = (x_inst_norm - mean_batch) / std_batch  # shape (N, T, D)

        # —— 可学习仿射变换 —— #
        if self.affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm


if __name__ == "__main__":
    x_in = torch.randn((128, 8, 32))
    revin_layer = RevIN(32, affine=True)
    x_out = revin_layer(x_in)
    print(x_out.shape)
