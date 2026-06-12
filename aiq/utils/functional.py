import re
from typing import List, Union

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


def ts_ohlcv_normalize(x: np.ndarray) -> np.ndarray:
    """Normalize OHLCV time-series data (sample-wise normalization).

    Price features use log relative values: log(price / last_close)
    Volume/Amount use temporal mean normalization: value / temporal_mean

    Args:
        x: Input data with shape (N, T, D), where D must contain at least 6 features:
           [open, high, low, close, volume, amount]
           N: number of samples, T: time steps, D: feature dimensions

    Returns:
        Normalized data with same shape as input, dtype float32

    Raises:
        ValueError: When input dimensions are incorrect or insufficient features
    """
    # Input validation
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array (N, T, D), got dimension: {x.ndim}")
    if x.shape[2] < 6:
        raise ValueError(f"Expected at least 6 features, got: {x.shape[2]}")

    # Feature index constants
    OPEN, HIGH, LOW, CLOSE, VOLUME, AMOUNT = 0, 1, 2, 3, 4, 5

    # Create copy and convert to float32
    normalized = x.astype(np.float32, copy=True)

    # Price feature normalization: log relative values
    # Using the last closing price of each sample as reference
    last_close = normalized[:, -1:, CLOSE][:, :, np.newaxis]  # Shape: (N, 1, 1)
    price_indices = (OPEN, HIGH, LOW, CLOSE)

    # Calculate log relative values for all price features at once
    # Add epsilon to avoid division by zero
    epsilon = 1e-10
    price_ratio = normalized[:, :, price_indices] / (last_close + epsilon)
    normalized[:, :, price_indices] = np.log(
        price_ratio
    )  # Equivalent to np.log(price_ratio)

    # Volume normalization: temporal mean normalization
    volume_data = normalized[:, :, VOLUME]
    volume_mean = np.mean(volume_data, axis=1, keepdims=True)
    # Safe division to handle zero mean
    normalized[:, :, VOLUME] = np.divide(
        volume_data, volume_mean, out=np.zeros_like(volume_data), where=volume_mean != 0
    )

    # Amount normalization: temporal mean normalization
    amount_data = normalized[:, :, AMOUNT]
    amount_mean = np.mean(amount_data, axis=1, keepdims=True)
    normalized[:, :, AMOUNT] = np.divide(
        amount_data, amount_mean, out=np.zeros_like(amount_data), where=amount_mean != 0
    )

    return normalized


def ts_robust_zscore(x: np.ndarray, clip_outlier: bool = False) -> np.ndarray:
    """
    Robust z-score normalization along the time axis.

    For each sample and feature in (N, T, D), subtract the median over T and
    divide by MAD (median absolute deviation).

    Args:
        x: Input array with shape (N, T, D).
        clip_outlier: Whether to clip z-scores into [-3, 3].

    Returns:
        Normalized array with shape (N, T, D)
    """
    if x.ndim != 3:
        raise ValueError(f"Input array must be 3D (N, T, D), but got shape {x.shape}")

    med = np.nanmedian(x, axis=1, keepdims=True)
    x_centered = x - med
    mad = np.nanmedian(np.abs(x_centered), axis=1, keepdims=True)

    std = mad * 1.4826 + 1e-12
    z = x_centered / std

    if clip_outlier:
        z = np.clip(z, -3.0, 3.0)

    return z


def robust_zscore(
    x: Union[pd.Series, np.ndarray], clip_outlier: bool = False
) -> Union[pd.Series, np.ndarray]:
    """
    Robust Z-score normalization using median and MAD.

    Computes:
        z = (x - median(x)) / (MAD(x) * 1.4826)

    NaNs are ignored in median/MAD computation but preserved in output.
    Optionally clips z-scores to [-3, 3] to limit extreme outliers.

    Parameters
    ----------
    x : pd.Series or np.ndarray
        Input data
    clip_outlier : bool, default False
        Whether to clip z-scores to [-3, 3].

    Returns
    -------
    pd.Series or np.ndarray
        Normalized data of same type as input.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    if len(x) == 0:
        return x

    is_series = isinstance(x, pd.Series)
    index, name = (x.index, x.name) if is_series else (None, None)

    arr = np.asarray(x, dtype=np.float32).copy()
    med = np.nanmedian(arr)
    arr_centered = arr - med
    mad = np.nanmedian(np.abs(arr_centered))
    z = arr_centered / (mad * 1.4826 + 1e-12)

    if clip_outlier:
        z = np.clip(z, -3.0, 3.0)

    return pd.Series(z, index=index, name=name) if is_series else z


def zscore(x, clip_min=-3.0, clip_max=3.0):
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    return np.clip((x - mean) / (std + 1e-8), clip_min, clip_max)


def neutralize(
    df: pd.DataFrame,
    industry_col: str = None,
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
    # If no factor columns are provided, return the original DataFrame
    if not factor_cols:
        return df.copy()

    if industry_col is None and cap_col is None:
        return df.copy()

    res_df = df.copy()

    # Extract the “feature” sub‑DataFrame
    features = res_df["feature"]

    # Build a combined regex to match all requested factor columns
    combined_pattern = "|".join(f"({pat})" for pat in factor_cols)
    actual_factors = [
        col for col in features.columns if re.search(combined_pattern, str(col))
    ]

    # Create design matrix: industry dummies + cap + constant
    X_parts = []
    if industry_col is not None:
        industry_dummies = pd.get_dummies(
            features[industry_col].astype("category"), prefix="IND", drop_first=True
        )
        X_parts.append(industry_dummies)

    if cap_col is not None:
        cap_series = features[[cap_col]].astype(float)
        X_parts.append(cap_series)

    X = pd.concat(X_parts, axis=1)
    X["CONST"] = 1.0

    # Initialize linear regression (no intercept, since CONST is included)
    model = LinearRegression(fit_intercept=False)

    # Loop through each factor, fit on non‑missing rows, and store residuals
    for factor in actual_factors:
        y = features[factor].astype(float)

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
