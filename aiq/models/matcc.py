import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from transformers import get_scheduler

from aiq.layers import MATCC
from aiq.losses import ClassBalancedLoss
from aiq.utils.discretize import discretize, undiscretize

from .base import BaseModel


class MATCCModel(BaseModel):
    def __init__(
        self,
        feature_cols=None,
        label_cols=None,
        d_feat=158,
        d_model=256,
        t_nhead=4,
        s_nhead=2,
        seq_len=8,
        pred_len=1,
        dropout=0.5,
        gate_input_start_index=158,
        gate_input_end_index=221,
        epochs=5,
        batch_size=1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        learning_rate=0.01,
        criterion_name="MSE",
        class_boundaries=None,
        class_weight=None,
        logger=None,
    ):
        # input parameters
        self._feature_cols = feature_cols
        self._label_cols = label_cols

        self.d_feat = d_feat
        self.d_model = d_model
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout = dropout
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.epochs = epochs
        self.batch_size = batch_size
        self.warmup_ratio = warmup_ratio
        self.lr_scheduler_type = lr_scheduler_type
        self.learning_rate = learning_rate
        self.criterion_name = criterion_name
        self.class_boundaries = class_boundaries
        self.num_classes = (
            len(class_boundaries) if self.class_boundaries is not None else None
        )
        self.class_weight = class_weight
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = MATCC(
            d_feat=self.d_feat,
            d_model=self.d_model,
            t_nhead=self.t_nhead,
            s_nhead=self.s_nhead,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            dropout=self.dropout,
            gate_input_start_index=self.gate_input_start_index,
            gate_input_end_index=self.gate_input_end_index,
            num_classes=self.num_classes,
        ).to(self.device)

        self.logger = logger

    def fit(self, train_dataset: Dataset, val_dataset: Dataset = None):
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        train_steps_epoch = len(train_loader)
        num_training_steps = self.epochs * train_steps_epoch
        num_warmup_steps = int(self.warmup_ratio * num_training_steps)

        time_now = time.time()

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.001,
        )
        lr_scheduler = get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        if self.criterion_name == "MSE":
            self.criterion = nn.MSELoss()
        elif self.criterion_name == "CE":
            class_weight = (
                torch.Tensor(self.class_weight)
                if self.class_weight is not None
                else None
            )
            self.criterion = nn.CrossEntropyLoss(weight=class_weight)
        else:
            raise NotImplementedError
    
        for epoch in range(self.epochs):
            self.logger.info("=" * 20 + " Epoch {} ".format(epoch + 1) + "=" * 20)

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (_, batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                batch_x = batch_x.squeeze(0).float().to(self.device)
                batch_y = batch_y.squeeze(0).float()

                outputs = self.model(batch_x)

                if self.num_classes is not None:
                    batch_y = discretize(
                        batch_y,
                        bins=self.class_boundaries,
                    )
                    loss = sum(
                        self.criterion(outputs[k], batch_y[:, k])
                        for k in range(len(outputs))
                    )
                else:
                    loss = self.criterion(outputs, batch_y)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.epochs - epoch) * train_steps_epoch - i)
                    self.logger.info(
                        "Epoch: {0}, step: {1}, lr: {2:.5f} train loss: {3:.7f}, speed: {4:.4f}s/iter, left time: {5:.4f}s".format(
                            epoch + 1,
                            i + 1,
                            lr_scheduler.get_last_lr()[0],
                            loss.item(),
                            speed,
                            left_time,
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
            self.logger.info(
                "Epoch: {0}, cost time: {1:.4f}s, train loss: {2:.7f}, val loss: {3:.7f}".format(
                    epoch + 1, time.time() - epoch_time, train_loss, val_loss
                )
            )

            # save checkpoints
            checkpoints_dir = "./checkpoints"
            os.makedirs(checkpoints_dir, exist_ok=True)
            model_file = os.path.join(
                checkpoints_dir, "model_epoch_{}.pth".format(epoch + 1)
            )
            torch.save(self.model.state_dict(), model_file)

    def eval(self, val_dataset: Dataset):
        self.model.eval()
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        total_loss = []
        with torch.no_grad():
            for i, (_, batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.squeeze(0).float().to(self.device)
                batch_y = batch_y.squeeze(0).float()

                outputs = self.model(batch_x)

                if self.num_classes is not None:
                    batch_y = discretize(
                        batch_y,
                        bins=self.class_boundaries,
                    )
                    loss = sum(
                        self.criterion(outputs[k], batch_y[:, k])
                        for k in range(len(outputs))
                    )
                else:
                    loss = self.criterion(outputs, batch_y)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        return total_loss

    def predict(self, test_dataset: Dataset) -> object:
        self.model.eval()
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        num_samples = test_dataset.data.shape[0]

        preds = np.zeros((num_samples, self.pred_len))
        for index, batch_x, *batch_y in test_loader:
            index = index.cpu().numpy()  # 确保索引为 numpy 数组
            batch_x = batch_x.squeeze(0).float().to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_x)
            
            if self.num_classes is not None:
                for k, output in enumerate(outputs):
                    probs = torch.softmax(output, dim=1)
                    cls_ids = torch.argmax(probs, dim=1)
                    preds[index, k] = (
                        undiscretize(cls_ids, bins=self.class_boundaries).cpu().numpy()
                    )
            else:
                preds[index] = outputs.cpu().numpy()
        
        # 统一数据插入逻辑
        label_names = test_dataset.label_names or [str(i) for i in range(self.pred_len)]
        test_dataset.insert(cols=[f"PRED_{name}" for name in label_names], data=preds)
        return test_dataset

    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_file = os.path.join(model_dir, "model.pth")
        torch.save(self.model.state_dict(), model_file)

    def load(self, model_dir):
        model_file = os.path.join(model_dir, "model.pth")
        self.model.load_state_dict(
            torch.load(model_file, map_location=self.device, weights_only=True)
        )
