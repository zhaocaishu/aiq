import torch

from aiq.dataset import TSDataset


if __name__ == '__main__':
    ts_dataset = TSDataset(data_dir='./data', save_dir='./temp', instruments='csi1000', start_time='2020-01-01',
                           end_time='2023-05-31', adjust_price=False, training=True,
                           feature_names=['Adj_Open', 'Adj_Close', 'Adj_High', 'Adj_Low', 'Volume'],
                           label_names=['Adj_Close'])

    data_loader = torch.utils.data.DataLoader(ts_dataset, batch_size=1)

    for batch_idx, sample in enumerate(data_loader):
        input, target = sample
        print(input.shape, target.shape)
