import torch
import numpy as np


def discretize(
    data: torch.Tensor,
    min_value: float = -1.0,
    max_value: float = 1.0,
    num_bins: int = 22,
) -> torch.Tensor:
    """
    将连续数据离散化为指定的区间。
    """
    bins = torch.Tensor(np.linspace(min_value, max_value, num_bins))
    ids = torch.bucketize(
        torch.clamp(data, min_value, max_value),
        boundaries=bins,
    )
    return ids


def dediscretize(
    ids: torch.Tensor,
    min_value: float = -1.0,
    max_value: float = 1.0,
    num_bins: int = 22,
) -> torch.Tensor:
    """
    将离散化的区间编号还原为连续数据的近似值。
    """
    bins = torch.Tensor(np.linspace(min_value, max_value, num_bins))
    data = bins[ids.long()]
    return data
