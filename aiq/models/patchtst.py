import time
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

from aiq.layers import PatchTSTBackbone, SeriesDecompose

from .base import BaseModel


class PatchTST(nn.Module):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True,
                 pretrain_head: bool = False, head_type='flatten', verbose: bool = False, **kwargs):

        super().__init__()

        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = SeriesDecompose(kernel_size)
            self.model_trend = PatchTSTBackbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                                patch_len=patch_len, stride=stride,
                                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                attn_dropout=attn_dropout,
                                                dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                                padding_var=padding_var,
                                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                                store_attn=store_attn,
                                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                                head_dropout=head_dropout, padding_patch=padding_patch,
                                                pretrain_head=pretrain_head, head_type=head_type,
                                                individual=individual, revin=revin, affine=affine,
                                                subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTSTBackbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                              patch_len=patch_len, stride=stride,
                                              max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                              n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                              attn_dropout=attn_dropout,
                                              dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                              padding_var=padding_var,
                                              attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                              store_attn=store_attn,
                                              pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                              head_dropout=head_dropout, padding_patch=padding_patch,
                                              pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                              revin=revin, affine=affine,
                                              subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTSTBackbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                          patch_len=patch_len, stride=stride,
                                          max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                          n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                          attn_dropout=attn_dropout,
                                          dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                          padding_var=padding_var,
                                          attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                          store_attn=store_attn,
                                          pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                                          padding_patch=padding_patch,
                                          pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                          revin=revin, affine=affine,
                                          subtract_last=subtract_last, verbose=verbose, **kwargs)

    def forward(self, x):  # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2,
                                                                                 1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        return x


class PatchTSTModel(BaseModel):
    def __init__(self, model_params=None):
        self.device = 'cpu'
        self.model_params = model_params
        self.model = PatchTST(configs=self.model_params)

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
                batch_y = batch_y.squeeze(0).float().to(self.device)

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

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        return preds
