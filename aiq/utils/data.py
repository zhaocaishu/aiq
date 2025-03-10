from typing import Union

import torch
import pandas as pd
import numpy as np

from aiq.utils.discretize import discretize


def robust_zscore(x: pd.Series, zscore=False):
    """Robust ZScore Normalization

    Use robust statistics for Z-Score normalization:
        mean(x) = median(x)
        std(x) = MAD(x) * 1.4826

    Reference:
        https://en.wikipedia.org/wiki/Median_absolute_deviation.
    """
    x = x - x.median()
    mad = x.abs().median()
    x = np.clip(x / mad / 1.4826, -3, 3)
    if zscore:
        x -= x.mean()
        x /= x.std()
    return x


def zscore(x: Union[pd.Series, pd.DataFrame]):
    return (x - x.mean()).div(x.std() + 1e-12)


def compute_discretized_class_counts(data_loader, class_boundaries):
    num_classes = len(class_boundaries)  # 确定类别总数
    counts = torch.zeros(num_classes, dtype=torch.int64)  # 初始化统计张量

    for i, (_, batch_x, batch_y) in enumerate(data_loader):
        batch_y = batch_y.squeeze(0).float()
        batch_y_discrete = discretize(batch_y, bins=class_boundaries)
        batch_y_discrete = batch_y_discrete.long()

        current_counts = torch.bincount(batch_y_discrete, minlength=num_classes)
        # 累计全局统计结果
        counts += current_counts.cpu()  # 确保在CPU上累加避免GPU内存问题

    return counts
