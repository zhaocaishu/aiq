import re
from typing import List

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


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def neutralize(
    df: pd.DataFrame, industry_col: str, cap_col: str, factor_cols: List[str]
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
    feats = df["feature"]

    # Build a combined regex to match all requested factor columns
    combined_pattern = "|".join(f"({pat})" for pat in factor_cols)
    actual_factors = [
        col for col in feats.columns
        if re.search(combined_pattern, str(col))
    ]

    # Create design matrix: industry dummies + cap + constant
    industry_dummies = pd.get_dummies(
        feats[industry_col].astype("category"),
        prefix="IND",
        drop_first=True
    )
    cap_series = feats[[cap_col]].astype(float)
    X = pd.concat([industry_dummies, cap_series], axis=1)
    X["CONST"] = 1.0
    X_values = X.values

    # Initialize linear regression (no intercept, since CONST is included)
    model = LinearRegression(fit_intercept=False)

    # Loop through each factor, fit on non‑missing rows, and store residuals
    for factor in actual_factors:
        y = feats[factor].astype(float)
        mask = y.notna()
        if not mask.any():
            # skip if all values are missing
            continue

        # Fit on rows where y is present
        model.fit(X_values[mask], y[mask])
        y_pred = model.predict(X_values[mask])
        residuals = y[mask] - y_pred

        # Write residuals back into the original DataFrame
        df.loc[mask, ("feature", factor)] = residuals.astype("float32")

    return df


def drop_extreme_label(x: np.ndarray, percentile: float = 2.5):
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError(f"Expected input shape (N, 1), got {x.shape}")

    # Compute thresholds across the flattened data
    lower, upper = np.percentile(x, [percentile, 100 - percentile])

    # Build mask of shape (N,)
    mask = (x[:, 0] >= lower) & (x[:, 0] <= upper)

    # Extract filtered values; result has shape (M, 1)
    filtered_x = x[mask]
    return mask, filtered_x
