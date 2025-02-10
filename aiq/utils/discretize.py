import numpy as np
import pandas as pd
from typing import Union, List


def discretize(
    data: Union[List[float], np.ndarray],
    min_value: float = -0.1,
    max_value: float = 0.1,
    num_bins: int = 22,
) -> np.ndarray:
    """
    将连续数据离散化为指定的区间。
    """
    bins = np.linspace(min_value, max_value, num_bins)
    ids = pd.cut(
        np.clip(data, min_value, max_value),
        bins=bins,
        labels=False,
        include_lowest=True,
    )
    return ids


def dediscretize(
    ids: Union[List[int], np.ndarray],
    min_value: float = -0.1,
    max_value: float = 0.1,
    num_bins: int = 22,
) -> np.ndarray:
    """
    将离散化的区间编号还原为连续数据的近似值。
    """
    bins = np.linspace(min_value, max_value, num_bins)
    data = (bins[ids] + bins[ids + 1]) / 2
    return data
