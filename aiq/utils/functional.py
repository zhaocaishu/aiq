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


def calculate_adx(df: pd.DataFrame, n: int = 10) -> pd.Series:
    """
    Calculate the ADX indicator for a given DataFrame.
    The DataFrame must contain columns: 'High', 'Low', 'Close'.

    Parameters:
    - df: pd.DataFrame with 'High', 'Low', 'Close' columns
    - n: period for smoothing (default 10)

    Returns:
    - pd.Series with 'ADX' column
    """
    df = df.copy()
    alpha = 1 / n

    # Precompute shifted close
    close_prev = df["Close"].shift(1)

    # Calculate True Range (TR)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - close_prev).abs()
    tr3 = (df["Low"] - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate Directional Movements
    up = df["High"] - df["High"].shift(1)
    dn = df["Low"].shift(1) - df["Low"]

    # Create masks
    pos = np.where((up > dn) & (up > 0), up, 0.0)
    neg = np.where((dn > up) & (dn > 0), dn, 0.0)

    # Smoothing with EMA
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    pos_smoothed = pd.Series(pos, index=df.index).ewm(alpha=alpha, adjust=False).mean()
    neg_smoothed = pd.Series(neg, index=df.index).ewm(alpha=alpha, adjust=False).mean()

    # Calculate Directional Indicators
    df["DI+"] = 100 * (pos_smoothed / atr)
    df["DI-"] = 100 * (neg_smoothed / atr)

    # Calculate DX (handle division by zero)
    di_sum = df["DI+"] + df["DI-"]
    di_diff = (df["DI+"] - df["DI-"]).abs()
    df["DX"] = 100 * (di_diff / di_sum.replace(0, np.nan))

    # Calculate ADX
    df["ADX"] = df["DX"].ewm(alpha=alpha, adjust=False).mean()

    return df["ADX"]
