import re
from typing import List, Union

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


def ts_ohlcv_normalize(x: np.ndarray) -> np.ndarray:
    """Normalize OHLCV time-series data.

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
    x_norm = x.astype(np.float32, copy=True)

    # Price feature normalization: log relative values
    # Using the last closing price of each sample as reference
    ref_close = x_norm[:, -1:, CLOSE][:, :, np.newaxis]  # Shape: (N, 1, 1)
    PRICE_IDX = (OPEN, HIGH, LOW, CLOSE)

    # Calculate log relative values for all price features at once
    # Add epsilon to avoid division by zero
    epsilon = 1e-5
    price_rel = x_norm[:, :, PRICE_IDX] / (ref_close + epsilon)
    x_norm[:, :, PRICE_IDX] = np.log(price_rel)

    # Volume normalization: temporal mean normalization
    vol = x_norm[:, :, VOLUME]
    vol_mean = np.mean(vol, axis=1, keepdims=True)
    # Safe division to handle zero mean
    x_norm[:, :, VOLUME] = np.divide(
        vol, vol_mean, out=np.zeros_like(vol), where=vol_mean != 0
    )

    # Amount normalization: temporal mean normalization
    amt = x_norm[:, :, AMOUNT]
    amt_mean = np.mean(amt, axis=1, keepdims=True)
    x_norm[:, :, AMOUNT] = np.divide(
        amt, amt_mean, out=np.zeros_like(amt), where=amt_mean != 0
    )

    return x_norm


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


def robust_zscore(x, clip_outlier: bool = False):
    """Robust ZScore Normalization

    Use robust statistics for Z-Score normalization:
        mean(x) = median(x)
        std(x) = MAD(x) * 1.4826

    Reference:
        https://en.wikipedia.org/wiki/Median_absolute_deviation.
    """
    med = np.nanmedian(x, axis=0)
    x_centered = x - med
    mad = np.nanmedian(np.abs(x_centered), axis=0)
    z = x_centered / (mad * 1.4826 + 1e-12)

    if clip_outlier:
        z = np.clip(z, -3.0, 3.0)

    return z


def zscore(x, clip_min=-3.0, clip_max=3.0):
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    return np.clip((x - mean) / (std + 1e-8), clip_min, clip_max)


def neutralize(
    df: pd.DataFrame,
    industry_col: str = None,
    cap_col: str = None,
    factor_cols: List[str] = [],
    add_suffix: bool = False,
    feature_group: str = "feature",
    label_group: str = "label",
) -> pd.DataFrame:
    """
    Neutralize specified factor columns by regressing out industry and market cap effects.
    Supports regex patterns in factor_cols to match multiple column names.

    Parameters:
    - df: DataFrame with a top‑level column label “feature” containing all features.
    - industry_col: Name of the column under “feature” that holds industry categories.
    - cap_col: Name of the column under “feature” that holds market capitalization values.
    - factor_cols: List of regex patterns (as strings) to select which factor columns to neutralize.
    - add_suffix: If True, keep original columns and add new columns with "_NEU" suffix.
                  If False, replace original columns with residuals (default).
    - feature_group : str, default "feature"
        Top-level column name for the feature sub-DataFrame.
    - label_group : str, default "label"
        Top-level column name for the label sub-DataFrame.

    Returns:
    - The same DataFrame, but with each matched factor column replaced by its regression residuals, or with new "_NEU" columns added if add_suffix=True.
    """
    # If no factor columns are provided, return the original DataFrame
    if not factor_cols:
        return df.copy()

    if industry_col is None and cap_col is None:
        return df.copy()

    # Extract the “feature” sub‑DataFrame
    res_df = df.copy()
    features = res_df[feature_group]

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

    # Build a combined regex to match all requested factor columns
    combined_pattern = "|".join(f"({pat})" for pat in factor_cols)

    # Collect all matched (group, column_name) pairs
    matched_factors = []
    for group in [feature_group, label_group]:
        if group not in res_df.columns:
            continue
        for col in res_df[group].columns:
            if re.search(combined_pattern, str(col)):
                matched_factors.append((group, col))

    if not matched_factors:
        return res_df

    # Initialize linear regression (no intercept, since CONST is included)
    model = LinearRegression(fit_intercept=False)

    # Loop through each factor, fit on non‑missing rows, and store residuals
    for group, factor in matched_factors:
        y = res_df[group][factor].astype(float)

        valid_mask = y.notna()
        if not valid_mask.any():
            # Skip if all values are missing
            continue

        # Fit on rows where y is present
        X_sub = X.loc[valid_mask].values
        y_sub = y.loc[valid_mask].values

        model.fit(X_sub, y_sub)
        residuals = y_sub - model.predict(X_sub)

        # Write residuals back
        if add_suffix:
            # Keep original column and add a new column with "_NEU" suffix
            neu_col = f"{factor}_NEU"
            res_df.loc[valid_mask, (group, neu_col)] = residuals.astype("float32")
        else:
            # Replace original column with residuals
            res_df.loc[valid_mask, (group, factor)] = residuals.astype("float32")

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
    return mask


def fillna(x: np.ndarray, fill_value=0.0):
    if not isinstance(x, np.ndarray):
        raise TypeError("输入必须是 numpy.ndarray 类型")

    x_filled = np.where(np.isnan(x), fill_value, x)
    return x_filled
