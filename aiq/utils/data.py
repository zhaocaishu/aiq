from copy import deepcopy
from typing import List, Union

import pandas as pd
import numpy as np
import numpy.linalg as la

import statsmodels.api as sm


def mad_filter(x: pd.DataFrame):
    """Robust statistics for outlier filter:
    """
    xm = x.median()
    xam = (x - xm).abs().median()
    for i in range(x.shape[1]):
        col = x.columns[i]
        x[col].where(x[col] > xm[i] + 5 * xam[i], xm[i] + 5 * xam[i], inplace=True)
        x[col].where(x[col] < xm[i] - 5 * xam[i], xm[i] - 5 * xam[i], inplace=True)
    return x


def neutralize(x: pd.DataFrame, industry_num, industry_col=None, market_cap_col=None, target_cols=None):
    """Factor neutralize
    """
    x_data = np.zeros((x.shape[0], industry_num))
    x_data[np.arange(x.shape[0]), x[industry_col].values] = 1
    x_data = np.hstack((np.ones((x.shape[0], 1)), x_data, np.log(x[market_cap_col].values).reshape(-1, 1)))
    y_data = x[target_cols].values
    model = sm.OLS(y_data, x_data).fit()
    x[target_cols] = model.resid.reshape(y_data.shape)
    return x


def zscore(x: Union[pd.Series, pd.DataFrame]):
    return (x - x.mean()).div(x.std())
