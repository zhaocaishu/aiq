import os
import time
import json

import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

from .base import BaseModel


class NLinear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, configs):
        super(NLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]


class NLinearModel(BaseModel):
    def __init__(self, model_params=None):
        if torch.cuda.device_count() == 1:
            self.device = torch.device('cuda:0')
        else:
            self.device = 'cpu'
        self.model_params = model_params
        self.model = NLinear(configs=self.model_params).to(self.device)

    def fit(self, train_dataset: Dataset, val_dataset: Dataset = None):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.model_params.batch_size, shuffle=True)

        time_now = time.time()

        train_steps = len(train_loader)

        optimizer = optim.Adam(self.model.parameters(), lr=self.model_params.learning_rate)
        criterion = nn.MSELoss()
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.model_params.pct_start,
                                            epochs=self.model_params.train_epochs,
                                            max_lr=self.model_params.learning_rate)
        for epoch in range(self.model_params.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()
                batch_x = batch_x.squeeze(0).float().to(self.device)
                batch_y = batch_y.squeeze(0).float()

                outputs = self.model(batch_x)

                f_dim = -1 if self.model_params.features == 'MS' else 0
                outputs = outputs[:, -self.model_params.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.model_params.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.model_params.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                optimizer.step()

                scheduler.step()

            train_loss = np.average(train_loss)
            val_loss = self.eval(val_dataset, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Val Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, val_loss))
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

    def eval(self, dataset: Dataset, criterion):
        self.model.eval()

        total_loss = []
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=self.model_params.batch_size)
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.squeeze(0).float().to(self.device)
                batch_y = batch_y.squeeze(0).float()

                outputs = self.model(batch_x)

                f_dim = -1 if self.model_params.features == 'MS' else 0
                outputs = outputs[:, -self.model_params.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.model_params.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def predict(self, dataset: Dataset) -> object:
        self.model.eval()

        preds = []
        pred_loader = torch.utils.data.DataLoader(dataset, batch_size=self.model_params.batch_size)
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(pred_loader):
                batch_x = batch_x.squeeze(0).float().to(self.device)
                outputs = self.model(batch_x)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.vstack(preds)
        dataset.add_column('PREDICTION', preds)
        return dataset

    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_file = os.path.join(model_dir, 'model.pth')
        torch.save(self.model, model_file)

        model_params = {
            'model_params': self.model_params
        }
        with open(os.path.join(model_dir, 'model.params'), 'w') as f:
            json.dump(model_params, f)

    def load(self, model_dir):
        model_file = os.path.join(model_dir, 'model.pth')
        self.model = torch.load(model_file)
        with open(os.path.join(model_dir, 'model.params'), 'r') as f:
            model_params = json.load(f)
            self.model_params = model_params['model_params']
