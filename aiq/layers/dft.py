import math

import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm

from aiq.layers.dlinear import DLinear, DLinear_Init
from aiq.layers.rwkv import Block, RWKV_Init


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim :]
                kh = k[:, :, i * dim :]
                vh = v[:, :, i * dim :]
            else:
                qh = q[:, :, i * dim : (i + 1) * dim]
                kh = k[:, :, i * dim : (i + 1) * dim]
                vh = v[:, :, i * dim : (i + 1) * dim]

            atten_ave_matrixh = torch.softmax(
                torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1
            )
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class Filter(nn.Module):
    def __init__(self, d_input, d_output, seq_len, kernel=5, stride=5):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.seq_len = seq_len

        self.trans = nn.Linear(d_input, d_output)

        self.aggregate = nn.Conv1d(
            d_output, d_output, kernel_size=kernel, stride=stride, groups=d_output
        )

        # 输入是[N, T, d_feat]
        conv_feat = math.floor((self.seq_len - kernel) / stride + 1)

        self.proj_out = nn.Linear(conv_feat, 1)

    def forward(self, x):
        x = self.trans.forward(x)  # [N, T, d_feat]
        x_trans = x.transpose(-1, -2)  # [N, d_feat, T]
        x_agg = self.aggregate.forward(x_trans)  # [N, d_feat, conv_feat]
        out = self.proj_out.forward(x_agg)  # [N, d_feat, 1]
        return out.transpose(-1, -2)  # [N, 1, d_feat]


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)  # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        # [N, 1, T], [N, T, D] --> [N, 1, D]
        output = torch.matmul(lam, z).squeeze(1)
        return output


class DFT(nn.Module):
    def __init__(
        self,
        d_feat=158,
        d_model=256,
        t_nhead=4,
        s_nhead=2,
        seq_len=8,
        pred_len=1,
        dropout=0.5,
        gate_input_start_index=158,
        gate_input_end_index=221,
        num_classes=None,
    ):
        super().__init__()

        self.d_feat = d_feat
        self.d_model = d_model
        self.n_attn = d_model
        self.n_head = t_nhead
        self.num_classes = num_classes

        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = gate_input_end_index - gate_input_start_index  # F'
        self.feature_gate = Filter(self.d_gate_input, self.d_feat, seq_len)

        self.rwkv_trend = Block(
            layer_id=0,
            n_embd=self.d_model,
            n_attn=self.n_attn,
            n_head=self.n_head,
            ctx_len=300,
            n_ffn=self.d_model,
            hidden_sz=self.d_model,
        )
        RWKV_Init(
            self.rwkv_trend,
            vocab_size=self.d_model,
            n_embd=self.d_model,
            rwkv_emb_scale=1.0,
        )
        self.rwkv_season = Block(
            layer_id=0,
            n_embd=self.d_model,
            n_attn=self.n_attn,
            n_head=self.n_head,
            ctx_len=300,
            n_ffn=self.d_model,
            hidden_sz=self.d_model,
        )
        RWKV_Init(
            self.rwkv_season,
            vocab_size=self.d_model,
            n_embd=self.d_model,
            rwkv_emb_scale=1.0,
        )

        self.feat_to_model = nn.Linear(d_feat, d_model)
        self.dlinear = DLinear(
            seq_len=seq_len,
            pred_len=seq_len,
            enc_in=self.d_model,
            kernel_size=3,
            individual=False,
            merge_outputs=False
        )
        DLinear_Init(self.dlinear, min_val=-5e-2, max_val=8e-2)

        self.trend_TC = nn.Sequential(
            SAttention(
                d_model=d_model, nhead=s_nhead, dropout=dropout
            ),  # Stock correlation
            self.rwkv_trend,  # Time correlation
        )
        self.season_TC = nn.Sequential(
            self.rwkv_season,  # Time correlation
            SAttention(
                d_model=d_model, nhead=s_nhead, dropout=dropout
            ),  # Stock correlation
        )

        self.market_linear = nn.Linear(d_feat, d_model)

        if num_classes is not None:
            self.classifiers = nn.ModuleList(
                [
                    nn.Sequential(
                        TemporalAttention(d_model=d_model),
                        nn.Linear(d_model, num_classes),
                    )
                    for _ in range(pred_len)
                ]
            )
        else:
            self.decoder = nn.Sequential(
                TemporalAttention(d_model=d_model),
                nn.Linear(d_model, pred_len),
            )

    def forward(self, x):
        # Slice input features
        continuous_feats = x[:, :, 1:self.gate_input_start_index]         # Continuous features (N, T, D_c)
        gate_input_feats = x[:, :, self.gate_input_start_index:self.gate_input_end_index]  # Gate-related features

        # Feature gating (e.g., market attention)
        market_signal = self.feature_gate(gate_input_feats)

        # Map features to model-specific space and decompose
        encoded_feats = self.feat_to_model(continuous_feats)
        trend_component, season_component = self.dlinear(encoded_feats)

        # Apply trend/seasonal transformations
        trend_component = self.trend_TC(trend_component) + self.market_linear(market_signal)
        season_component = self.season_TC(season_component)

        # Combine and generate output
        fused_component = trend_component + season_component
        if self.num_classes is not None:
            outputs = [classifier(fused_component) for classifier in self.classifiers]
        else:
            outputs = self.decoder(fused_component)

        return outputs
