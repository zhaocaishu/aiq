import os
import time
import copy

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler

from aiq.layers import PPNet, Muon
from aiq.losses import TopKLoss, HybridLoss

from .base import BaseModel


class PPNetModel(BaseModel):
    def __init__(
        self,
        feature_names=None,
        label_names=None,
        d_ts_feat=6,
        d_intraday_ts_feat=6,
        d_cs_feat=125,
        d_fund_feat=3,
        d_mkt_feat=63,
        d_emb=8,
        d_model=256,
        t_nhead=4,
        s_nhead=2,
        dropout=0.5,
        beta=5.0,
        epochs=50,
        batch_size=1,
        warmup_steps=500,
        use_muon=False,
        lr_scheduler_type="cosine",
        learning_rate=0.001,
        criterion_name="MSE",
        early_stopping_patience=5,
        pretrained=None,
        save_dir=None,
        logger=None,
    ):
        # input args
        self.feature_names = feature_names
        self.label_names = label_names
        self.epochs = epochs
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.use_muon = use_muon
        self.lr_scheduler_type = lr_scheduler_type
        self.learning_rate = float(learning_rate)
        self.criterion_name = criterion_name
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # model
        self.model = PPNet(
            d_ts_feat=d_ts_feat,
            d_intraday_ts_feat=d_intraday_ts_feat,
            d_cs_feat=d_cs_feat,
            d_fund_feat=d_fund_feat,
            d_mkt_feat=d_mkt_feat,
            d_emb=d_emb,
            d_model=d_model,
            t_nhead=t_nhead,
            s_nhead=s_nhead,
            dropout=dropout,
            beta=beta,
        )

        if pretrained is not None:
            try:
                state_dict = torch.load(pretrained)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading pretrained weights from {pretrained}: {e}")

        self.model = self.model.to(self.device)

        self.best_model_state = None

        # loss function
        if self.criterion_name == "MSE":
            self.criterion = nn.MSELoss()
        elif self.criterion_name == "Hybrid":
            self.criterion = HybridLoss()
        elif self.criterion_name == "TopK":
            self.criterion = TopKLoss()
        else:
            raise NotImplementedError

        self.save_dir = save_dir

        self.logger = logger

    def to_device(self, tensor):
        """统一设备转换方法"""
        return tensor.squeeze(0).to(device=self.device)

    def get_muon_adamw_params(self, model):
        muon_params = []
        adamw_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # ===== 1️⃣ Muon：只给核心2D权重 =====
            if (
                p.ndim >= 2
                and "embedding" not in name
                and "norm" not in name
                and "prediction_head" not in name
            ):
                muon_params.append(p)
            # ===== 2️⃣ AdamW：其他全部 =====
            else:
                adamw_params.append(p)
        return muon_params, adamw_params

    def fit(self, train_dataset: Dataset, val_dataset: Dataset = None):
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        train_steps_epoch = len(train_loader)
        num_training_steps = self.epochs * train_steps_epoch

        # Optimizer
        if self.use_muon:
            muon_params, adamw_params = self.get_muon_adamw_params(self.model)
            optimizer = Muon(
                lr=self.learning_rate,
                wd=0.05,
                muon_params=muon_params,
                adamw_params=adamw_params,
            )
        else:
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.05,
            )

        # Cosine scheduler
        lr_scheduler = get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Step-based early stopping
        patience_steps = self.early_stopping_patience * train_steps_epoch
        best_val_loss = float("inf")
        best_step = 0
        global_step = 0
        val_check_interval = max(500, train_steps_epoch // 2)
        stop_training = False

        # Warmup-safe + early-stop protection
        min_train_steps_before_stop = int(0.1 * num_training_steps)

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []

            iter_count = 0
            time_now = time.time()

            for i, batch_dict in enumerate(train_loader):
                # 检查是否需要停止训练
                if stop_training:
                    break

                iter_count += 1
                global_step += 1

                batch_industry_ids = self.to_device(batch_dict["stock_industry_ids"])
                batch_ts_features = self.to_device(batch_dict["stock_ts_features"])
                batch_intraday_ts_features = self.to_device(
                    batch_dict["stock_intraday_ts_features"]
                )
                batch_cs_features = self.to_device(batch_dict["stock_cs_features"])
                batch_fund_features = self.to_device(batch_dict["stock_fund_features"])
                batch_market_features = self.to_device(
                    batch_dict["market_state_features"]
                )
                batch_labels = self.to_device(batch_dict["labels"])

                assert not torch.isnan(
                    batch_industry_ids
                ).any(), "NaN at batch_industry_ids"
                assert not torch.isnan(
                    batch_ts_features
                ).any(), "NaN at batch_ts_features"
                assert not torch.isnan(
                    batch_cs_features
                ).any(), "NaN at batch_cs_features"
                assert not torch.isnan(
                    batch_fund_features
                ).any(), "NaN at batch_fund_features"
                assert not torch.isnan(
                    batch_market_features
                ).any(), "NaN at batch_market_features"
                assert not torch.isnan(batch_labels).any(), "NaN at batch_labels"

                optimizer.zero_grad()
                outputs = self.model(
                    batch_ts_features,
                    batch_cs_features,
                    batch_market_features,
                    batch_industry_ids,
                    batch_fund_features,
                    batch_intraday_ts_features
                )
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                optimizer.step()
                lr_scheduler.step()

                train_losses.append(loss.item())

                # Log training progress
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.epochs - epoch - 1) * train_steps_epoch
                        + (train_steps_epoch - i - 1)
                    )
                    cur_lr = optimizer.param_groups[0]["lr"]
                    self.logger.info(
                        f"[Epoch {epoch+1}/{self.epochs}][Step {global_step}] "
                        f"LR: {cur_lr:.6e}, Train Loss: {loss.item():.6f}, "
                        f"Speed: {speed:.2f}s/iter, ETA: {int(left_time)}s"
                    )
                    iter_count = 0
                    time_now = time.time()

                # Step-based validation check
                if val_dataset is not None and (
                    global_step % val_check_interval == 0 or i == train_steps_epoch - 1
                ):

                    val_loss = self.eval(val_dataset)
                    steps_since_best = global_step - best_step
                    msg = f"[Step {global_step}] Validation Loss: {val_loss:.8f}"

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_step = global_step
                        self.best_model_state = copy.deepcopy(self.model.state_dict())
                        msg += f" | 🔥 New Best at Step {best_step}"
                    else:
                        msg += f" | 😐 No improvement, {steps_since_best}/{patience_steps} steps"

                        # Warmup-safe early stop
                        if (
                            global_step > min_train_steps_before_stop
                            and steps_since_best >= patience_steps
                        ):
                            msg += " | ⏹ Early stopping triggered"
                            stop_training = True

                    self.logger.info(msg)

            train_loss = np.mean(train_losses)
            if val_dataset is not None:
                val_loss = self.eval(val_dataset)
                self.logger.info(
                    f"[Epoch {epoch+1}/{self.epochs}] "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )
            else:
                self.logger.info(
                    f"[Epoch {epoch+1}/{self.epochs}] Train Loss: {train_loss:.6f}"
                )

            if stop_training:
                break

        # load the best weights back into the model after training
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info(
                f"Best model restored from step {best_step}, val_loss {best_val_loss:.8f}"
            )

    def eval(self, val_dataset: Dataset):
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        total_losses = []

        self.model.eval()

        for i, batch_dict in enumerate(val_loader):
            batch_industry_ids = self.to_device(batch_dict["stock_industry_ids"])
            batch_ts_features = self.to_device(batch_dict["stock_ts_features"])
            batch_intraday_ts_features = self.to_device(
                batch_dict["stock_intraday_ts_features"]
            )
            batch_cs_features = self.to_device(batch_dict["stock_cs_features"])
            batch_fund_features = self.to_device(batch_dict["stock_fund_features"])
            batch_market_features = self.to_device(batch_dict["market_state_features"])
            batch_labels = self.to_device(batch_dict["labels"])

            with torch.no_grad():
                outputs = self.model(
                    batch_ts_features,
                    batch_cs_features,
                    batch_market_features,
                    batch_industry_ids,
                    batch_fund_features,
                    batch_intraday_ts_features,
                )

                loss = self.criterion(outputs, batch_labels)

            total_losses.append(loss.item())

        total_loss = np.mean(total_losses)

        return total_loss

    def predict(self, test_dataset: Dataset) -> object:
        self.model.eval()

        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        indices = []
        preds = []
        for i, batch_dict in enumerate(test_loader):
            batch_sample_indices = batch_dict["sample_indices"]
            batch_industry_ids = self.to_device(batch_dict["stock_industry_ids"])
            batch_ts_features = self.to_device(batch_dict["stock_ts_features"])
            batch_intraday_ts_features = self.to_device(
                batch_dict["stock_intraday_ts_features"]
            )
            batch_cs_features = self.to_device(batch_dict["stock_cs_features"])
            batch_fund_features = self.to_device(batch_dict["stock_fund_features"])
            batch_market_features = self.to_device(batch_dict["market_state_features"])

            with torch.no_grad():
                outputs = self.model(
                    batch_ts_features,
                    batch_cs_features,
                    batch_market_features,
                    batch_industry_ids,
                    batch_fund_features,
                    batch_intraday_ts_features,
                )

            indices.append(batch_sample_indices.squeeze(0).numpy())
            preds.append(outputs.cpu().numpy())

        indices = np.concatenate(indices, axis=0)
        preds = np.concatenate(preds, axis=0)

        label_names = test_dataset.label_names
        pred_df = test_dataset.data.iloc[indices].copy()
        pred_df[[f"PRED_{name}" for name in label_names]] = preds
        return pred_df

    def load(self, model_name=None):
        model_name = "model.pth" if model_name is None else model_name
        model_file = os.path.join(self.save_dir, model_name)
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        self.model.load_state_dict(
            torch.load(model_file, map_location=self.device, weights_only=True)
        )
        self.logger.info(f"Successfully loaded model from {model_file}")

    def save(self, model_name=None):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        model_name = "model.pth" if model_name is None else model_name
        model_file = os.path.join(self.save_dir, model_name)

        torch.save(self.model.state_dict(), model_file)
        self.logger.info(f"Model saved to {model_file}")
