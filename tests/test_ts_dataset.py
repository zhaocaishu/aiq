import torch

from aiq.dataset import TSDataset

if __name__ == '__main__':
    ts_dataset = TSDataset(data_dir='./data', save_dir='./tmp', instruments='csi1000', start_time='2020-01-01',
                           end_time='2023-05-31', adjust_price=False, training=True)

    data_loader = torch.utils.data.DataLoader(ts_dataset, batch_size=1)

    for batch_idx, sample in enumerate(data_loader):
        input, target = sample
        print(input.shape, target.shape)
