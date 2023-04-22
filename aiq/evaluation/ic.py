import abc

import pandas as pd
from scipy import stats

import alphalens
import alphalens.performance as perf


class IC(abc.ABC):
    def __init__(self, timestamp_col='Date', symbol_col='Symbol', price_col='Close'):
        self.price_col = price_col
        self.timestamp_col = timestamp_col
        self.symbol_col = symbol_col

    def eval(self, df, factor_col):
        factors = df[factor_col]

        df = df.reset_index()
        prices = df[[self.timestamp_col, self.symbol_col, self.price_col]]
        prices = prices.pivot(self.timestamp_col, self.symbol_col, values=self.price_col)

        factor_data = alphalens.utils.get_clean_factor_and_forward_returns(factors, prices, quantiles=5)
        ic_data = perf.factor_information_coefficient(factor_data)
        ic_summary_table = pd.DataFrame()
        ic_summary_table["IC Mean"] = ic_data.mean()
        ic_summary_table["IC Std."] = ic_data.std()
        ic_summary_table["Risk-Adjusted IC"] = \
            ic_data.mean() / ic_data.std()
        t_stat, p_value = stats.ttest_1samp(ic_data, 0)
        ic_summary_table["t-stat(IC)"] = t_stat
        ic_summary_table["p-value(IC)"] = p_value
        ic_summary_table["IC Skew"] = stats.skew(ic_data)
        ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)
        return ic_summary_table
