from aiq.dataset import Dataset, Alpha158, random_split


if __name__ == '__main__':
    dataset = Dataset('./data', instruments='all', handler=Alpha158(), adjust_price=False, shuffle=False)

    print(dataset.to_dataframe().shape)

    train_dataset, val_dataset = random_split(dataset, [('2021-08-30', '2022-04-28'), ('2022-04-29', '2022-08-26')])
    print(train_dataset.to_dataframe().shape, val_dataset.to_dataframe().shape)
