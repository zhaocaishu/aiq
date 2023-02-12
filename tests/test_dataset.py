from aiq.dataset import Dataset, Alpha158


if __name__ == '__main__':
    dataset = Dataset('./data', instruments='all', handler=Alpha158(), shuffle=False)

    print(dataset.to_dataframe().shape)
