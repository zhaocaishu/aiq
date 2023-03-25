from aiq.dataset import Dataset, Alpha158, ts_split


if __name__ == '__main__':
    dataset = Dataset('./data', instruments='all', handler=Alpha158(), adjust_price=False)
    train_dataset, val_dataset = ts_split(dataset, [('2021-08-30', '2022-04-28'), ('2022-04-29', '2022-08-26')])
    print(dataset.to_dataframe().shape, train_dataset.to_dataframe().shape, val_dataset.to_dataframe().shape)
