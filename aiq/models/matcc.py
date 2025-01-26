import os
import time
import json

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from transformers import get_scheduler

from aiq.layers import MATCC
from aiq.losses import ICLoss, CCCLoss

from .base import BaseModel


class MATCCModel(BaseModel):
    def __init__(
        self,
        feature_cols=None,
        label_col=None,
        d_feat=158,
        d_model=256,
        t_nhead=4,
        s_nhead=2,
        seq_len=8,
        pred_len=1,
        dropout=0.5,
        epochs=5,
        batch_size=1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        learning_rate=0.01,
        criterion_name="MSE",
    ):
        # input parameters
        self._feature_cols = feature_cols
        self._label_col = label_col

        self.d_feat = d_feat
        self.d_model = d_model
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.warmup_ratio = warmup_ratio
        self.lr_scheduler_type = lr_scheduler_type
        self.learning_rate = learning_rate
        self.criterion_name = criterion_name

        if torch.cuda.device_count() == 1:
            self.device = torch.device("cuda:0")
        else:
            self.device = "cpu"
        self.model = MATCC(
            d_feat=self.d_feat,
            d_model=self.d_model,
            t_nhead=self.t_nhead,
            s_nhead=self.s_nhead,
            seq_len=self.seq_len,
            dropout=self.dropout,
        ).to(self.device)
        if self.criterion_name == "IC":
            self.criterion = ICLoss()
        elif self.criterion_name == "CCC":
            self.criterion = CCCLoss()
        elif self.criterion_name == "MSE":
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError

    def fit(self, train_dataset: Dataset, val_dataset: Dataset = None):
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        train_steps_epoch = len(train_loader)

        time_now = time.time()

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        num_training_steps = self.epochs * train_steps_epoch
        num_warmup_steps = int(self.warmup_ratio * num_training_steps)
        lr_scheduler = get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        for epoch in range(self.epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (_, batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                batch_x = batch_x.squeeze(0).float().to(self.device)
                batch_y = batch_y.squeeze(0).float()

                outputs = self.model(batch_x)

                loss = self.criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.epochs - epoch) * train_steps_epoch - i)
                    print(
                        "Epoch: {0}, iters: {1}, train loss: {2:.7f}, speed: {3:.4f}s/iter, left time: {4:.4f}s".format(
                            epoch + 1, i + 1, loss.item(), speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
                optimizer.step()

            lr_scheduler.step()

            train_loss = np.average(train_loss)
            val_loss = self.eval(val_dataset)
            print(
                "Epoch: {0}, cost time: {1}, train loss: {2:.7f}, val loss: {3:.7f}".format(
                    epoch + 1, time.time() - epoch_time, train_loss, val_loss
                )
            )

    def eval(self, dataset: Dataset):
        self.model.eval()
        val_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        total_loss = []
        with torch.no_grad():
            for i, (_, batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.squeeze(0).float().to(self.device)
                batch_y = batch_y.squeeze(0).float()

                outputs = self.model(batch_x)

                loss = self.criterion(outputs, batch_y)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        return total_loss

    def predict(self, dataset: Dataset) -> object:
        self.model.eval()
        pred_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        preds = np.zeros(dataset.data.shape[0])
        with torch.no_grad():
            for i, (index, batch_x, batch_y) in enumerate(pred_loader):
                batch_x = batch_x.squeeze(0).float().to(self.device)
                outputs = self.model(batch_x)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds[index] = pred

        dataset.insert("PREDICTION", preds)
        return dataset

    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_file = os.path.join(model_dir, "model.pth")
        torch.save(self.model, model_file)

        model_params = {
            "feature_cols": self._feature_cols,
            "label_col": self._label_col,
            "d_feat": self.d_feat,
            "d_model": self.d_model,
            "t_nhead": self.t_nhead,
            "s_nhead": self.s_nhead,
            "dropout": self.dropout,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "batch_size": self.batch_size,
        }

        with open(os.path.join(model_dir, "model.params"), "w") as f:
            json.dump(model_params, f)

    def load(self, model_dir):
        model_file = os.path.join(model_dir, "model.pth")
        self.model = torch.load(model_file)
        with open(os.path.join(model_dir, "model.params"), "r") as f:
            model_params = json.load(f)
            self._feature_cols = model_params["feature_cols"]
            self._label_col = model_params["label_col"]
            self.d_feat = model_params["d_feat"]
            self.d_model = model_params["d_model"]
            self.t_nhead = model_params["t_nhead"]
            self.s_nhead = model_params["s_nhead"]
            self.dropout = model_params["dropout"]
            self.seq_len = model_params["seq_len"]
            self.pred_len = model_params["pred_len"]
            self.batch_size = model_params["batch_size"]
