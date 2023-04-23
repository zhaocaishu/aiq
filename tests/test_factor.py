import alphalens

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from aiq.ops import Ref, Mean, Std, Cov, Corr, CSRank, Sum, Max, Min
from aiq.dataset.processor import CSFilter, CSNeutralize, CSZScore, CSFillna
from aiq.evaluation import IC


if __name__ == '__main__':
    df = pd.read_csv('/Users/darren/Downloads/all.csv')
    df = df[(df['Date'] >= '2022-01-01') & (df['Date'] <= '2022-12-31')]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(['Date', 'Symbol'])

    df['VWAP_CSRANK'] = CSRank((df['High'] + df['Low']) / 2.0)
    df['CLOSE_CSRANK'] = CSRank(df['Close'])
    df['VOLUME_CSRANK'] = CSRank(df['Volume'])
    df['OPEN_CSRANK'] = CSRank(df['Open'])
    df['HIGH_CSRANK'] = CSRank(df['High'])

    def ts_func_lv1(x):
        x['OVRANKCORR10'] = -1.0 * Corr(x['OPEN_CSRANK'], x['VOLUME_CSRANK'], 10)
        x['CVRANKCOV5'] = Cov(x['CLOSE_CSRANK'], x['VOLUME_CSRANK'], 5)
        x['HVRANKCORR3'] = Corr(x['HIGH_CSRANK'], x['VOLUME_CSRANK'], 5)
        x['HVRANKCORR5'] = -1.0 * Corr(x['High'], x['VOLUME_CSRANK'], 3)
        x['HVRANKCOV5'] = Cov(x['HIGH_CSRANK'], x['VOLUME_CSRANK'], 5)
        x['WVRANKCORR5'] = Corr(x['HIGH_CSRANK'], x['VOLUME_CSRANK'], 5)
        return x
    df = df.groupby('Symbol', group_keys=False).apply(ts_func_lv1)

    df['CVRANKCOV5'] = -1.0 * CSRank(df['CVRANKCOV5'])
    df['HVRANKCORR3'] = CSRank(df['HVRANKCORR3'])
    df['HVRANKCOV5'] = -1.0 * CSRank(df['HVRANKCOV5'])
    df['WVRANKCORR5'] = CSRank(df['WVRANKCORR5'])

    def ts_func_lv2(x):
        x['HVRANKCORR3'] = -1.0 * Sum(x['HVRANKCORR3'], 3)
        x['WVRANKCORR5'] = -1.0 * Max(x['WVRANKCORR5'], 5)
        x['CHLRANKCORR12'] = (x['Close'] - Min(x['Low'], 12)) / (Max(x['High'], 12) - Min(x['Low'], 12))
        return x
    df = df.groupby('Symbol', group_keys=False).apply(ts_func_lv2)

    df['CHLRANKCORR12'] = CSRank(df['CHLRANKCORR12'])

    def ts_func_lv3(x):
        x['CHLRANKCORR12'] = -1.0 * Corr(x['CHLRANKCORR12'], x['VOLUME_CSRANK'], 6)
        return x
    df = df.groupby('Symbol', group_keys=False).apply(ts_func_lv3)

    factor_cols = ['OVRANKCORR10', 'CVRANKCOV5', 'HVRANKCORR3', 'HVRANKCORR5', 'HVRANKCOV5', 'WVRANKCORR5', 'CHLRANKCORR12']

    # fill nan
    fillna = CSFillna(target_cols=factor_cols)
    df = fillna(df)

    # remove outlier
    outlier_filter = CSFilter(target_cols=factor_cols)
    df = outlier_filter(df)

    # factor neutralize
    cs_neut = CSNeutralize(industry_num=110, industry_col='Industry_id', market_cap_col='Total_mv',
                           target_cols=factor_cols)
    df = cs_neut(df)

    # factor standardization
    cs_score = CSZScore(target_cols=factor_cols)
    df = cs_score(df)

    ic_analysis = IC()
    for factor in factor_cols:
        print('======================%s============================' % factor)
        print(ic_analysis.eval(df, factor_col=factor))

