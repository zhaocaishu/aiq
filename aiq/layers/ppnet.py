import math
import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm

from aiq.layers.revin import RevIN


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.shape[1], :]


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


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

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
                torch.matmul(qh, kh.transpose(1, 2)), dim=-1
            )
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.concat(att_output, dim=-1)

        # FFN
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output


class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output / self.t, dim=-1)
        return self.d_output * output


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        h = self.trans(z)  # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        return output


class PPNet(nn.Module):
    def __init__(
        self,
        pv_feature_start_index,
        market_feature_start_index,
        market_feature_end_index,
        ind_feature_index,
        ind_embedding_dim,
        seq_len,
        pred_len,
        d_feat,
        d_model,
        t_nhead,
        s_nhead,
        dropout,
        beta,
    ):
        super(PPNet, self).__init__()

        # price-volume-based features
        self.pv_feature_start_index = pv_feature_start_index

        # industry feature index
        self.ind_feature_index = ind_feature_index

        # market features
        self.market_feature_start_index = market_feature_start_index
        self.market_feature_end_index = market_feature_end_index
        self.market_feature_dim = market_feature_end_index - market_feature_start_index
        self.market_gating_layer = Gate(self.market_feature_dim, d_feat, beta=beta)

        # pre-normalization
        self.revin_norm = RevIN(d_feat)

        # industry embedding
        self.ind_embedding = nn.Embedding(256, ind_embedding_dim)

        # feature projection
        self.feature_projection = nn.Linear(d_feat, d_model)

        # positional encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # intra-stock attention
        self.temporal_attention = TAttention(
            d_model=d_model, nhead=t_nhead, dropout=dropout
        )

        # inter-stock attention
        self.spatial_projection = nn.Linear(d_model + ind_embedding_dim, d_model)
        self.spatial_attention = SAttention(
            d_model=d_model, nhead=s_nhead, dropout=dropout
        )

        # temporal aggregation
        self.temporal_aggregation = TemporalAttention(d_model=d_model)

        # decoder
        self.decoder = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # Extract instrument features and normalize
        pv_features = x[
            :, :, self.pv_feature_start_index : self.market_feature_start_index
        ]  # Shape: (N, T, D)
        pv_features = self.revin_norm(pv_features)

        # Extract market features and apply gating
        market_features = x[
            :, -1, self.market_feature_start_index : self.market_feature_end_index
        ]  # Shape: (N, F')
        gated_weights = self.market_gating_layer(market_features)  # (N, D)
        gated_pv_features = pv_features * gated_weights.unsqueeze(1)  # (N, T, D)

        # Feature projection
        x_proj = self.feature_projection(gated_pv_features)  # (N, T, d_model)

        # Positional encoding
        x_encoded = self.positional_encoding(x_proj)

        # Intra-stock temporal attention
        x_temporal = self.temporal_attention(x_encoded)

        # Inter-stock spatial attention
        ind_ids = x[:, :, self.ind_feature_index].long()
        x_ind = self.ind_embedding(ind_ids)  # (N, T, ind_dim)
        x_spatial_input = torch.cat([x_temporal, x_ind], dim=-1)
        x_spatial_input = self.spatial_projection(x_spatial_input)  # (N, T, d_model)
        x_spatial = self.spatial_attention(x_spatial_input)

        # Temporal aggregation across time dimension
        x_aggregated = self.temporal_aggregation(x_spatial)  # (N, d_model)

        # Prediction decoder
        output = self.decoder(x_aggregated)  # (N, pred_len)

        return output
