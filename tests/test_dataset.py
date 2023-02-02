from aiq.dataset import Dataset, Alpha100

if __name__ == '__main__':
    dataset = Dataset('./data/features', symbols=['BABA'], handler=Alpha100())

    print(dataset.to_dataframe())
