from aiq.dataset import Dataset, Alpha100


if __name__ == '__main__':
    dataset = Dataset('./data', handler=Alpha100(), shuffle=False)

    print(dataset.to_dataframe().shape)
