import math
import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm


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
    def __init__(self, d_model, nhead, dropout, d_emb):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.temperature = math.sqrt(self.head_dim)

        # Q, K, V projection
        self.qtrans = nn.Linear(d_model + d_emb, d_model, bias=False)
        self.ktrans = nn.Linear(d_model + d_emb, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(nhead)])

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_emb, eps=1e-5)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x, industry_emb):
        # x: (N, D)  — 股票特征
        # industry_emb: (N, d_emb) — 行业 embedding
        x = self.norm1(x)
        industry_emb = self.norm2(industry_emb)

        # Q / K 用行业信息引导，V 只用股票特征
        q = torch.cat([x, industry_emb], dim=-1)
        k = torch.cat([x, industry_emb], dim=-1)
        v = x

        q = self.qtrans(q)
        k = self.ktrans(k)
        v = self.vtrans(v)

        # 多头拆分
        q = q.view(-1, self.nhead, self.head_dim)
        k = k.view(-1, self.nhead, self.head_dim)
        v = v.view(-1, self.nhead, self.head_dim)

        att_output_heads = []
        for i in range(self.nhead):
            qh = q[:, i, :]  # (N, head_dim)
            kh = k[:, i, :]  # (N, head_dim)
            vh = v[:, i, :]  # (N, head_dim)

            attn_weights = torch.softmax(
                torch.matmul(qh, kh.transpose(0, 1)) / self.temperature, dim=-1
            )
            attn_weights = self.attn_dropout[i](attn_weights)

            out = torch.matmul(attn_weights, vh)
            att_output_heads.append(out)

        att_output = torch.cat(att_output_heads, dim=-1)  # (N, D)

        xt = x + att_output
        out = xt + self.ffn(self.norm3(xt))
        return out


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
        output = xt + self.ffn(self.norm2(xt))

        return output


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
        d_feat,
        d_market,
        d_emb,
        d_model,
        t_nhead,
        s_nhead,
        dropout,
        beta,
    ):
        super(PPNet, self).__init__()

        # market features
        self.market_gating_layer = Gate(d_market, d_feat, beta=beta)

        # industry embedding
        self.industry_embedding = nn.Embedding(256, d_emb)

        # feature projection
        self.feature_projection = nn.Linear(d_feat, d_model)

        # positional encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # intra-stock attention
        self.temporal_attention = TAttention(
            d_model=d_model, nhead=t_nhead, dropout=dropout
        )

        # inter-stock attention
        self.spatial_attention = SAttention(
            d_model=d_model, d_emb=d_emb, nhead=s_nhead, dropout=dropout
        )

        # temporal aggregation
        self.temporal_aggregation = TemporalAttention(d_model=d_model)

        # decoder
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, industry_ids, stock_features, market_features):
        # Industry embedding
        industry_emb = self.industry_embedding(industry_ids)  # (N, d_emb)

        # Extract market features and apply gating
        market_features = market_features[:, -1, :]  # Shape: (N, F')
        gated_weights = self.market_gating_layer(market_features)  # (N, D)
        gated_pv_features = stock_features * gated_weights.unsqueeze(1)  # (N, T, D)

        # Feature projection
        x_proj = self.feature_projection(gated_pv_features)  # (N, T, d_model)

        # Positional encoding
        x_encoded = self.positional_encoding(x_proj)

        # Intra-stock temporal attention
        x_temporal = self.temporal_attention(x_encoded)

        # Temporal aggregation across time dimension
        x_aggregated = self.temporal_aggregation(x_temporal)  # (N, d_model)

        # Inter-stock spatial attention
        x_spatial = self.spatial_attention(x_aggregated, industry_emb=industry_emb)

        # Prediction decoder
        output = self.decoder(x_spatial)  # (N, 1)

        return output
