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
    
class GateNN(nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dim=None,
        output_dim=None,
        dropout_rate=0.0,
        batch_norm=False
    ):
        super(GateNN, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        gate_layers = [nn.Linear(input_dim, hidden_dim)]
        if batch_norm:
            gate_layers.append(nn.BatchNorm1d(hidden_dim))
        gate_layers.append(nn.ReLU())
        if dropout_rate > 0:
            gate_layers.append(nn.Dropout(dropout_rate))
        gate_layers.append(nn.Linear(hidden_dim, output_dim))
        gate_layers.append(nn.Sigmoid())
        self.gate = nn.Sequential(*gate_layers)

    def forward(self, inputs):
        return self.gate(inputs) * 2


class PPNet(nn.Module):
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
        gate_input_end_index=221
    ):
        super().__init__()

        self.d_feat = d_feat
        self.d_model = d_model
        self.n_attn = d_model
        self.n_head = t_nhead

        # industry embeddings
        self.ind_embedding_dim = 16
        self.ind_embedding = nn.Embedding(192, self.ind_embedding_dim)

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

        self.temporal_attn = TemporalAttention(d_model=d_model)

        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model + self.ind_embedding_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5)
        )
        
        self.fc = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: [N, T, D]
        ind_class = x[:, -1, 0].long()
        assert ind_class.max() < self.ind_embedding.num_embeddings, "Index out of embedding range"
        cat_feats = self.ind_embedding(ind_class)

        cont_feats = x[:, :, 1:self.gate_input_start_index]
        cont_feats = self.feat_to_model(cont_feats)
        trend_feat, season_feat = self.dlinear(cont_feats)

        gate_input = x[:, :, self.gate_input_start_index:self.gate_input_end_index]
        market_feat = self.feature_gate(gate_input)

        trend_out = self.trend_TC(trend_feat) + self.market_linear(market_feat)
        season_out = self.season_TC(season_feat)

        fused_out = trend_out + season_out
        temporal_out = self.temporal_attn(fused_out)

        fused_feats = torch.cat([cat_feats, temporal_out], dim=1)
        fused_feats = self.fusion_layer(fused_feats)

        output = self.fc(fused_feats)
        return output
