import pandas as pd

from aiq.dataset.processor import FeatureGroupMean, RandomLabelSampling


if __name__ == '__main__':
    df = pd.DataFrame({'Date': ['2022-01-01', '2022-01-01', '2022-01-01'], 'Industry_id': [1, 2, 1], 'Value': [3, 2, 4]})

    # group_mean = FeatureGroupMean(fields_group=['Value'])
    # print(group_mean(df))

    random_sample = RandomLabelSampling(label_name='Value', bound_value=[2, 3])
    print(random_sample(df))

