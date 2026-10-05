import re
from typing import List, Union

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


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
    fields_group: str = "feature",
    exposure_group: str = "feature",
    industry_col: str = None,
    cap_col: str = None,
    factor_cols: List[str] = None,
    add_suffix: bool = False,
    suffix: str = "_NEU",
    min_samples: int = 10,
) -> pd.DataFrame:
    """
    Neutralize specified factor columns by regressing out industry and market cap effects.
    Supports regex patterns in factor_cols to match multiple column names.

    Parameters:
    - df: DataFrame with a top‑level column label exposure_group containing all features.
    - fields_group: str, default "feature"
        Top-level column name for the factor sub-DataFrame.
    - exposure_group: str, default "feature"
        Top-level column name for the exposure sub-DataFrame.
    - industry_col: Name of the column under exposure_group that holds industry categories.
    - cap_col: Name of the column under exposure_group that holds market capitalization values.
    - factor_cols: List of regex patterns (as strings) to select which factor columns to neutralize.
    - add_suffix: If True, keep original columns and add new columns with "_NEU" suffix.
                  If False, replace original columns with residuals (default).
    - suffix: str, default "_NEU"
        Suffix for the new columns.
    - min_samples: int, default 10
        Minimum number of valid samples to fit the regression model.

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
    factor_df = res_df[fields_group]
    exposure_df = res_df[exposure_group]

    # Create design matrix: industry dummies + cap + constant
    X_parts = []
    valid_x = pd.Series(True, index=res_df.index)

    if industry_col is not None:
        industry = exposure_df[industry_col]

        # Missing industry should not silently become baseline industry.
        valid_x &= industry.notna()

        industry_dummies = pd.get_dummies(
            industry.astype("category"), prefix="IND", drop_first=True
        )

        X_parts.append(industry_dummies)

    if cap_col is not None:
        cap = exposure_df[cap_col]

        # Missing cap should not silently become baseline cap.
        valid_x &= cap.notna() & np.isfinite(cap)

        cap_series = cap.astype(float)
        X_parts.append(cap_series)

    X = pd.concat(X_parts, axis=1)
    X["CONST"] = 1.0

    # Build a combined regex to match all requested factor columns
    combined_pattern = "|".join(f"({pat})" for pat in factor_cols)

    # Collect all matched (group, column_name) pairs
    matched_factors = []
    for col in res_df[fields_group].columns:
        if re.search(combined_pattern, str(col)):
            matched_factors.append(col)

    if not matched_factors:
        return res_df

    # Initialize linear regression (no intercept, since CONST is included)
    model = LinearRegression(fit_intercept=False)

    # Loop through each factor, fit on non‑missing rows, and store residuals
    for factor in matched_factors:
        y = factor_df[factor].astype(float)

        valid = valid_x & y.notna() & np.isfinite(y)

        n_valid = int(valid.sum())

        if n_valid < min_samples:
            continue

        # Fit on rows where y is present
        X_sub = X.loc[valid].values
        y_sub = y.loc[valid].values

        model.fit(X_sub, y_sub)
        residuals = y_sub - model.predict(X_sub)

        # Write residuals back
        if add_suffix:
            # Keep original column and add a new column with "_NEU" suffix
            neu_col = f"{factor}{suffix}"
            res_df.loc[valid, (fields_group, neu_col)] = residuals.astype("float32")
        else:
            # Replace original column with residuals
            res_df.loc[valid, (fields_group, factor)] = residuals.astype("float32")

    return res_df


def drop_extreme_label(x: np.ndarray, percentile: float = 2.5):
    x = np.asarray(x)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim != 2 or x.shape[1] != 1:
        raise ValueError(f"Expected input shape (N, 1) or (N,), got {x.shape}")

    if not 0 <= percentile < 50:
        raise ValueError(f"percentile must be in [0, 50), got {percentile}")

    values = x[:, 0]

    valid = np.isfinite(values)

    # 没有有效 label
    if not valid.any():
        return np.zeros(len(values), dtype=bool)

    lower, upper = np.percentile(
        values[valid],
        [percentile, 100 - percentile],
    )

    return valid & (values >= lower) & (values <= upper)


def fillna(x: np.ndarray, fill_value=0.0):
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(x)}")

    x_filled = np.where(np.isnan(x), fill_value, x)
    return x_filled
