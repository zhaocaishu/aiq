import torch


def discretize(
    data: torch.Tensor,
    min_value: float = -3.0,
    max_value: float = 3.0,
    num_bins: int = 60,
) -> torch.Tensor:
    """
    将连续数据离散化为指定的区间。
    """
    bins = torch.linspace(min_value, max_value, num_bins, device=data.device)
    ids = torch.bucketize(
        torch.clamp(data, min_value, max_value),
        boundaries=bins,
    )
    return ids


def undiscretize(
    ids: torch.Tensor,
    min_value: float = -3.0,
    max_value: float = 3.0,
    num_bins: int = 60,
) -> torch.Tensor:
    """
    将离散化的区间编号还原为连续数据的近似值。
    """
    interval = (max_value - min_value) / (num_bins - 1)
    bins = torch.linspace(min_value, max_value, num_bins, device=ids.device)
    data = bins[ids] + 0.5 * interval
    return data
