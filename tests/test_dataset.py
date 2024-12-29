import pandas as pd
from aiq.dataset import Dataset, Alpha158, Alpha101, ts_split


if __name__ == '__main__':
    dataset = Dataset('./data', instruments='csi1000', handlers=(Alpha158(), Alpha101()),
                      start_time='2012-01-01', end_time='2023-05-01', adjust_price=False)
    train_dataset, val_dataset = ts_split(dataset, [('2012-07-01', '2021-12-31'), ('2022-01-01', '2022-11-31')])
    print(dataset.to_dataframe().shape, train_dataset.to_dataframe().shape, val_dataset.to_dataframe().shape)
