import pandas as pd

from aiq.dataset.processor import FeatureGroupMean


if __name__ == '__main__':
    df = pd.DataFrame({'Date': ['2022-01-01', '2022-01-01', '2022-01-01'], 'Industry_id': [1, 2, 1], 'Value': [3, 2, 4]})

    group_mean = FeatureGroupMean(fields_group=['Value'])
    print(group_mean(df))
