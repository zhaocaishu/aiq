import re
from typing import Union, List

import torch
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


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


def zscore(x: Union[pd.Series, pd.DataFrame, np.ndarray]):
    if isinstance(x, np.ndarray):
        return (x - x.mean()) / (x.std() + 1e-12)
    else:
        return (x - x.mean()).div(x.std() + 1e-12)


def neutralize(
    df: pd.DataFrame, industry_col: str, cap_col: str, factor_cols: List[str]
) -> pd.DataFrame:
    """
    Neutralize specified factor columns by industry and market capitalization.
    Supports regex/expression patterns in factor_cols to match multiple columns.

    Parameters:
    - df: DataFrame with MultiIndex columns; level 'feature' contains input columns.
    - industry_col: Name of the industry category column under 'feature'.
    - cap_col: Name of the market cap column under 'feature'.
    - factor_cols: List of factor column names (under 'feature') to be neutralized.

    Returns:
    - DataFrame with each specified factor column replaced by its regression residuals.
    """
    # Extract the feature-level DataFrame for clarity
    feats = df["feature"]

    # Combine all patterns into one big regex using alternation (|)
    # Each pattern is grouped to preserve its regex semantics
    combined_pattern = "|".join(f"({pat})" for pat in factor_cols)

    # Filter column names in one pass; original order is preserved
    actual_factors = [
        col for col in feats.columns if re.search(combined_pattern, str(col))
    ]

    # Build design matrix: industry dummies + cap + intercept
    industry = feats[industry_col].astype("category")
    cap = feats[cap_col].astype(float)

    X = pd.get_dummies(industry, prefix="IND", drop_first=True)
    X["CAP"] = cap
    X["CONST"] = 1.0
    X_values = X.astype(float).values

    # Prepare regression model (no intercept since CONST is included)
    model = LinearRegression(fit_intercept=False)

    # Identify which factors never have NaNs and which have some NaN in y
    no_nan_factors = [f for f in actual_factors if not feats[f].isna().any()]
    with_nan_factors = [f for f in actual_factors if feats[f].isna().any()]

    # Batch fit for all-no-NaN factors
    if no_nan_factors:
        # construct Y (n_samples × k) array
        Y = feats[no_nan_factors].astype(float).to_numpy()

        # fit a multi-output regressor
        model.fit(X_values, Y)

        # predict and compute residuals
        Y_pred = model.predict(X_values)
        residuals = Y - Y_pred  # same shape

        # write back into df
        for idx, f in enumerate(no_nan_factors):
            df.loc[:, ("feature", f)] = residuals[:, idx].astype("float32")

    # Loop for each factor that has NaNs
    for f in with_nan_factors:
        y = feats[f].astype(float)

        # mask out only the rows where y is present
        mask = ~y.isna()
        if not mask.any():
            continue

        # fit and predict on the subset
        model.fit(X_values[mask], y[mask].to_numpy())
        y_pred = model.predict(X_values[mask])

        # compute residuals and write back
        resid = y.copy()
        resid.loc[mask] = y.loc[mask] - y_pred
        df.loc[mask, ("feature", f)] = resid.astype("float32")

    return df


def drop_extreme_label(x: np.array):
    sorted_indices = np.argsort(x)
    N = x.shape[0]
    percent_2_5 = int(0.025 * N)
    filtered_indices = sorted_indices[percent_2_5:-percent_2_5]
    mask = np.zeros_like(x, dtype=bool)
    mask[filtered_indices] = True
    return mask, x[mask]


def discretize(data: torch.Tensor, bins: List[float]) -> torch.Tensor:
    """
    将连续数据离散化为指定的区间。
    """
    bins_tensor = torch.Tensor(bins, device=data.device)
    ids = torch.bucketize(
        torch.clamp(data, bins[0], bins[-1]),
        boundaries=bins_tensor,
    )
    return ids


def undiscretize(ids: torch.Tensor, bins: List[float]) -> torch.Tensor:
    """
    将离散化的区间编号还原为连续数据的近似值。
    """
    bins_tensor = torch.Tensor(bins, device=ids.device)
    # 获取对应区间的左右边界
    left = bins_tensor[ids]
    right = bins_tensor[torch.clamp(ids + 1, max=len(bins) - 1)]  # 防止越界
    # 计算中间值
    data = (left + right) / 2
    return data


def count_samples_per_bin(data_loader, class_boundaries):
    num_classes = len(class_boundaries)  # 确定类别总数
    counts = torch.zeros(num_classes, dtype=torch.int64)  # 初始化统计张量

    for i, (_, batch_x, batch_y) in enumerate(data_loader):
        batch_y = batch_y.flatten().float()
        discreted_batch_y = discretize(batch_y, bins=class_boundaries).long()

        current_counts = torch.bincount(discreted_batch_y, minlength=num_classes)
        # 累计全局统计结果
        counts += current_counts.cpu()  # 确保在CPU上累加避免GPU内存问题

    return counts
