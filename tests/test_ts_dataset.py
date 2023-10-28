import torch

from aiq.dataset import TSDataset


if __name__ == '__main__':
    train_dataset = TSDataset(data_dir='./data', save_dir='./temp', instruments='csi1000', start_time='2020-01-01',
                              end_time='2022-11-31', adjust_price=True, training=True,
                              feature_names=['Adj_Open', 'Adj_Close', 'Adj_High', 'Adj_Low', 'Volume'],
                              label_names=['Label'])

    val_dataset = TSDataset(data_dir='./data', save_dir='./temp', instruments='csi1000', start_time='2023-01-01',
                            end_time='2023-05-31', adjust_price=True, training=False,
                            feature_names=['Adj_Open', 'Adj_Close', 'Adj_High', 'Adj_Low', 'Volume'],
                            label_names=['Label'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024)
    for batch_idx, sample in enumerate(train_loader):
        input, target = sample
        print('Train:', input, target)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024)
    for batch_idx, sample in enumerate(val_loader):
        input, target = sample
        print('Validation:', input, target)
