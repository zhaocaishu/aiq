import re
from typing import List, Union

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


def ts_robust_zscore(x: np.ndarray, clip_outlier: bool = False) -> np.ndarray:
    """
    Time-series Robust Z-Score Normalization

    Normalize along the time dimension (T) for each (N, D) using median and MAD.

    Parameters
    ----------
    x : np.ndarray
        Input data of shape (N, T, D), where N is the batch size, T is the time length,
        and D is the feature dimension.
    clip_outlier : bool, optional
        If True, clip the resulting z-scores to the range [-3, 3] to limit extreme outliers.
        Default is False.

    Returns
    -------
    np.ndarray
        The normalized data of the same shape as input.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    if x.ndim != 3:
        raise ValueError(f"Input array must be 3D (N, T, D), but got shape {x.shape}")

    # Compute global median over samples and time: shape (1, 1, D)
    med = np.nanmedian(x, axis=1, keepdims=True)

    # Center the data
    x_centered = x - med

    # Compute MAD over time
    mad = np.nanmedian(np.abs(x_centered), axis=1, keepdims=True)

    # Scale factor for consistency
    std = mad * 1.4826 + 1e-12

    # Compute robust z-score
    z = x_centered / std

    if clip_outlier:
        z = np.clip(z, -3.0, 3.0)

    return z


def ts_cs_robust_zscore(x: np.ndarray, clip_outlier: bool = False) -> np.ndarray:
    """
    Time-series mean scaling, then cross-sectional robust z-score.

    Args:
        x: array of shape (N, T, D)
        clip_outlier: whether to clip output into [-3, 3]

    Returns:
        normalized array with shape (N, T, D)
    """
    if x.ndim != 3:
        raise ValueError(f"Input array must be 3D (N, T, D), but got shape {x.shape}")

    # Time-series scaling (per sample)
    t_mean = np.nanmean(x, axis=1, keepdims=True) + 1e-12
    x_t_norm = x / t_mean

    # Cross-sectional robust z-score (per timestamp)
    c_med = np.nanmedian(x_t_norm, axis=0, keepdims=True)
    c_centered = x_t_norm - c_med
    c_mad = np.nanmedian(np.abs(c_centered), axis=0, keepdims=True)
    c_std = c_mad * 1.4826 + 1e-12

    z = c_centered / c_std

    if clip_outlier:
        z = np.clip(z, -3.0, 3.0)

    return z


def robust_zscore(
    x: Union[pd.Series, np.ndarray], clip_outlier: bool = False
) -> Union[pd.Series, np.ndarray]:
    """
    Robust ZScore Normalization using median and MAD.

    Uses robust statistics for Z-Score normalization:
        center = median(x)
        scale = MAD(x) * 1.4826

    The result can be optionally clipped to [-3, 3] range.

    NaN values are ignored in median and MAD calculations but remain in output.

    Parameters
    ----------
    x : pd.Series or np.ndarray
        Input data
    clip_outlier : bool, optional
        If True, clip the resulting z-scores to the range [-3, 3] to limit extreme outliers.
        Default is False.

    Returns
    -------
    pd.Series or np.ndarray
        Normalized data with same type as input. NaN values remain in place.

    References
    ----------
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    if len(x) == 0:
        return x

    # 保存输入类型和元数据
    is_series = isinstance(x, pd.Series)
    index = x.index if is_series else None
    name = x.name if is_series else None

    # 转换为ndarray进行计算，避免修改原始数据
    arr = np.asarray(x, dtype=np.float32).copy()

    # 计算中位数并中心化
    med = np.nanmedian(arr)
    arr_centered = arr - med

    # 计算MAD并标准化
    mad = np.nanmedian(np.abs(arr_centered))
    std = mad * 1.4826 + 1e-12
    result = arr_centered / std

    if clip_outlier:
        result = np.clip(result, -3.0, 3.0)

    # 转换回原始类型
    if is_series:
        return pd.Series(result, index=index, name=name)

    return result


def zscore(x, clip_min=-3.0, clip_max=3.0):
    return np.clip((x - x.mean()) / (x.std() + 1e-8), clip_min, clip_max)


def neutralize(
    df: pd.DataFrame,
    industry_col: str,
    cap_col: str = None,
    factor_cols: List[str] = [],
) -> pd.DataFrame:
    """
    Neutralize specified factor columns by regressing out industry and market cap effects.
    Supports regex patterns in factor_cols to match multiple column names.

    Parameters:
    - df: DataFrame with a top‑level column label “feature” containing all features.
    - industry_col: Name of the column under “feature” that holds industry categories.
    - cap_col: Name of the column under “feature” that holds market capitalization values.
    - factor_cols: List of regex patterns (as strings) to select which factor columns to neutralize.

    Returns:
    - The same DataFrame, but with each matched factor column replaced by its regression residuals.
    """
    # Extract the “feature” sub‑DataFrame
    res_df = df.copy()
    feats = res_df["feature"]

    # Build a combined regex to match all requested factor columns
    combined_pattern = "|".join(f"({pat})" for pat in factor_cols)
    actual_factors = [
        col for col in feats.columns if re.search(combined_pattern, str(col))
    ]

    # Create design matrix: industry dummies + cap + constant
    industry_dummies = pd.get_dummies(
        feats[industry_col].astype("category"), prefix="IND", drop_first=True
    )
    if cap_col is not None:
        cap_series = feats[[cap_col]].astype(float)
        X = pd.concat([industry_dummies, cap_series], axis=1)
    else:
        X = industry_dummies
    X["CONST"] = 1.0

    # Initialize linear regression (no intercept, since CONST is included)
    model = LinearRegression(fit_intercept=False)

    # Loop through each factor, fit on non‑missing rows, and store residuals
    for factor in actual_factors:
        y = feats[factor].astype(float)

        valid_mask = y.notna()
        if not valid_mask.any():
            # skip if all values are missing
            continue

        # Fit on rows where y is present
        X_sub = X.loc[valid_mask].values
        y_sub = y.loc[valid_mask].values

        model.fit(X_sub, y_sub)
        residuals = y_sub - model.predict(X_sub)

        # Write residuals back into the original DataFrame
        res_df.loc[valid_mask, ("feature", factor)] = residuals.astype("float32")

    return res_df


def drop_extreme_label(x: np.ndarray, percentile: float = 2.5):
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 1:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        else:
            raise ValueError(f"Expected input shape (N, 1) or (N,), got {x.shape}")

    lower, upper = np.percentile(x, [percentile, 100 - percentile])
    mask = (x[:, 0] >= lower) & (x[:, 0] <= upper)
    return mask, x[mask]


def fillna(x: np.ndarray, fill_value=0.0):
    if not isinstance(x, np.ndarray):
        raise TypeError("输入必须是 numpy.ndarray 类型")

    x_filled = np.where(np.isnan(x), fill_value, x)
    return x_filled
