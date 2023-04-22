import alphalens

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from aiq.ops import Ref, Mean, Std, Cov, Corr, CSRank
from aiq.dataset.processor import CSFilter, CSNeutralize, CSZScore, CSFillna
from aiq.evaluation import IC


if __name__ == '__main__':
    df = pd.read_csv('/Users/darren/Downloads/all.csv')
    df = df[(df['Date'] >= '2022-01-01') & (df['Date'] <= '2022-12-31')]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(['Date', 'Symbol'])

    df['CLOSE_CSRANK'] = CSRank(df['Close'])
    df['VOLUME_CSRANK'] = CSRank(df['Volume'])
    df['OPEN_CSRANK'] = CSRank(df['Open'])

    def feature_func(x):
        x['CVRANKCOV5'] = Cov(x['CLOSE_CSRANK'], x['VOLUME_CSRANK'], 5)
        # x['OVRANKCORR10'] = -1.0 * Corr(x['OPEN_CSRANK'], x['VOLUME_CSRANK'], 10)
        return x
    df = df.groupby('Symbol', group_keys=False).apply(feature_func)

    df['CVRANKCOV5'] = -1.0 * CSRank(df['CVRANKCOV5'])

    # fill nan
    fillna = CSFillna(target_cols=['CVRANKCOV5'])
    df = fillna(df)

    # remove outlier
    outlier_filter = CSFilter(target_cols=['CVRANKCOV5'])
    df = outlier_filter(df)

    # factor neutralize
    cs_neut = CSNeutralize(industry_num=110, industry_col='Industry_id', market_cap_col='Total_mv',
                           target_cols=['CVRANKCOV5'])
    df = cs_neut(df)

    # factor standardization
    cs_score = CSZScore(target_cols=['CVRANKCOV5'])
    df = cs_score(df)

    ic_analysis = IC(factor_col='CVRANKCOV5', price_col='Close')
    print('=====:', ic_analysis.eval(df))
