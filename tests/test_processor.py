import pandas as pd

from aiq.dataset.processor import CSFilter, CSNeutralize, TSStandardize


if __name__ == '__main__':
    df = pd.DataFrame({'Date': ['2022-01-01', '2022-01-01', '2022-01-01', '2022-01-01'], 'Ind_class': [1, 2, 1, 1],
                       'Market_cap': [3, 5, 4, 100], 'Factor_0': [0.1, 2.4, 1.2, 1.8],
                       'Factor_1': [2.0, 1.0, 3.0, 4.0],
                       'Symbol': ['a', 'b', 'b', 'a']})
    df = df.set_index('Date')

    outlier_filter = CSFilter(target_cols=['Factor_0', 'Factor_1'])
    print(outlier_filter(df))

    neutralize = CSNeutralize(industry_num=3, industry_col='Ind_class', market_cap_col='Market_cap',
                              target_cols=['Factor_0', 'Factor_1'])
    print(neutralize(df))

    df = df.reset_index()
    df = df.set_index('Symbol')

    ts_standardize = TSStandardize(target_cols=['Factor_0', 'Factor_1'], save_dir='./temp')
    ts_standardize.fit(df)
    df = ts_standardize(df)
