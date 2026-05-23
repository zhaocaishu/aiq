import math

import torch
from torch import nn

from .embed import DataEmbedding


class AttnPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):
        w = torch.softmax(self.score(x), dim=1)
        return torch.sum(w * x, dim=1)


class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        N, T, D = k.shape

        q = self.q_proj(q).view(N, self.nhead, self.head_dim)
        k = self.k_proj(k).view(N, T, self.nhead, self.head_dim)
        v = self.v_proj(v).view(N, T, self.nhead, self.head_dim)

        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        q = q.unsqueeze(2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.squeeze(2).reshape(N, D)

        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MarketGate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_input, d_output),
            nn.SiLU(),
            nn.Linear(d_output, d_output),
        )
        self.d_output = d_output
        self.t = beta

    def forward(self, x):
        x_enc = self.enc(x)
        x_scale = torch.softmax(x_enc / self.t, dim=-1)
        x_scale = self.d_output * x_scale
        return x_scale


class FusionBlock(nn.Module):
    def __init__(self, d_in, d_model, dropout=0.1):
        super().__init__()

        self.proj = nn.Linear(d_in, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.mlp = MLP(
            hidden_size=d_model,
            intermediate_size=d_model * 2,
        )

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)
        h = self.norm(x)

        feat = self.mlp(h)
        gate = self.gate(h)

        return x + self.dropout(gate * feat)


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(nhead)])

        self.o_proj = nn.Linear(d_model, d_model)

        self.mlp = MLP(d_model, 2 * d_model)
        self.input_layernorm = nn.LayerNorm(d_model, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x):
        # Self Attention
        residual = x
        hidden_states = self.input_layernorm(x)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        attn_outputs = []
        for i in range(self.nhead):
            start = i * self.head_dim
            end = (i + 1) * self.head_dim
            qh = q[:, :, start:end]
            kh = k[:, :, start:end]
            vh = v[:, :, start:end]

            attn_logits = torch.matmul(qh, kh.transpose(1, 2)) / math.sqrt(
                self.head_dim
            )

            attn_weights = torch.softmax(attn_logits, dim=-1)
            attn_weights = self.attn_dropout[i](attn_weights)
            attn_outputs.append(torch.matmul(attn_weights, vh))

        hidden_states = torch.cat(attn_outputs, dim=-1)
        hidden_states = self.o_proj(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SAttention(nn.Module):
    def __init__(self, d_model, d_emb, nhead, dropout):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.temperature = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(nhead)])

        self.o_proj = nn.Linear(d_model, d_model)

        self.mlp = MLP(d_model, 2 * d_model)
        self.input_layernorm = nn.LayerNorm(d_model, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(d_model, eps=1e-5)

        # Learnable scale
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def _build_industry_bias(
        self, industry_ids: torch.Tensor, delta1: float = 0.65, delta2: float = 0.2
    ) -> torch.Tensor:
        N = industry_ids.shape[0]
        assert industry_ids.shape[1] == 2, "industry_ids must have shape (N, 2)"
        assert (
            1 > delta1 > delta2 >= 0
        ), "Bias factors must satisfy 1 > delta1 > delta2 >= 0"

        l1 = industry_ids[:, 0]
        l2 = industry_ids[:, 1]

        l2_eq = l2.unsqueeze(1) == l2.unsqueeze(0)
        l1_eq = l1.unsqueeze(1) == l1.unsqueeze(0)

        D = torch.full((N, N), delta2, dtype=torch.float32, device=industry_ids.device)

        D[l1_eq & ~l2_eq] = delta1
        D[l2_eq] = 1.0
        D.fill_diagonal_(1.0)

        return D

    def forward(self, x, industry_ids):
        # Industry bias: (N, N) — 行业偏差矩阵
        industry_bias = self._build_industry_bias(industry_ids).detach()

        # x: (N, D)  — 股票特征
        residual = x
        x_states = self.input_layernorm(x)

        # Self Attention
        q = self.q_proj(x_states)
        k = self.k_proj(x_states)
        v = self.v_proj(x_states)

        q = q.view(-1, self.nhead, self.head_dim)
        k = k.view(-1, self.nhead, self.head_dim)
        v = v.view(-1, self.nhead, self.head_dim)

        attn_outputs = []
        for i in range(self.nhead):
            qh = q[:, i, :]
            kh = k[:, i, :]
            vh = v[:, i, :]

            attn_logits = torch.matmul(qh, kh.transpose(0, 1)) / self.temperature
            attn_logits += self.alpha * industry_bias

            attn_weights = torch.softmax(attn_logits, dim=-1)
            attn_weights = self.attn_dropout[i](attn_weights)

            attn_out = torch.matmul(attn_weights, vh)
            attn_outputs.append(attn_out)

        hidden_states = torch.cat(attn_outputs, dim=-1)
        hidden_states = self.o_proj(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class PPNet(nn.Module):
    def __init__(
        self,
        d_ts_feat,
        d_intraday_ts_feat,
        d_cs_feat,
        d_fund_feat,
        d_mkt_feat,
        d_emb,
        d_model,
        t_nhead,
        s_nhead,
        dropout,
        beta,
        use_intraday=True,
    ):
        super().__init__()

        self.use_intraday = use_intraday

        # Feature Dimensions
        self.d_temporal_hidden = d_model // 4
        self.d_fusion_input = self.d_temporal_hidden + d_cs_feat
        if self.use_intraday:
            self.d_fusion_input += self.d_temporal_hidden

        # Temporal Encoder (Intra-stock) for daily data
        self.data_embedding = DataEmbedding(
            c_in=d_ts_feat,
            d_model=self.d_temporal_hidden,
            dropout=dropout,
        )
        self.temporal_layers = nn.Sequential(
            *[
                TAttention(
                    d_model=self.d_temporal_hidden, nhead=t_nhead, dropout=dropout
                )
                for _ in range(2)
            ]
        )
        self.temporal_pool = AttnPooling(self.d_temporal_hidden)

        # Temporal Encoder (Intra-stock) for intraday data
        if self.use_intraday:
            self.intraday_data_embedding = DataEmbedding(
                c_in=d_intraday_ts_feat,
                d_model=self.d_temporal_hidden,
                dropout=dropout,
            )
            self.intraday_temporal_layers = nn.Sequential(
                *[
                    TAttention(
                        d_model=self.d_temporal_hidden, nhead=t_nhead, dropout=dropout
                    )
                    for _ in range(2)
                ]
            )

            # Cross attention
            self.cross_attn = CrossAttention(self.d_temporal_hidden, t_nhead)

        # Market-Conditioned Feature Gating
        self.market_gate = MarketGate(d_input=d_mkt_feat, d_output=d_cs_feat, beta=beta)

        # Fusion Encoder (Temporal + Cross-Sectional Feature Integration)
        self.fusion_block = FusionBlock(
            d_in=self.d_fusion_input,
            d_model=d_model,
            dropout=dropout,
        )

        # Spatial Encoder (Inter-stock / Industry-aware)
        self.spatial_encoder = SAttention(
            d_model=d_model,
            d_emb=d_emb,
            nhead=s_nhead,
            dropout=dropout,
        )

        # Prediction head
        self.prediction_head = nn.Linear(d_model, 1, bias=False)

    def forward(
        self,
        stock_industry_ids,
        stock_ts_features,
        stock_intraday_ts_features=None,
        stock_cs_features=None,
        stock_fund_features=None,
        market_state_features=None,
    ):
        # Intra-Stock Temporal Modeling (Daily)
        stock_temporal_embeds = self.data_embedding(stock_ts_features)
        stock_temporal_states = self.temporal_layers(stock_temporal_embeds)
        stock_temporal_features = self.temporal_pool(stock_temporal_states)

        # Intra-Stock Temporal Modeling (Intraday)
        if self.use_intraday:
            assert stock_intraday_ts_features is not None
            stock_intraday_temporal_embeds = self.intraday_data_embedding(
                stock_intraday_ts_features
            )
            stock_intraday_temporal_states = self.intraday_temporal_layers(
                stock_intraday_temporal_embeds
            )

            stock_intraday_temporal_features = self.cross_attn(
                stock_temporal_features,
                stock_intraday_temporal_states,
                stock_intraday_temporal_states,
            )

        # Market-Conditioned Feature Gating
        gate_weights = self.market_gate(market_state_features)
        gated_stock_cs_features = stock_cs_features * gate_weights

        # Feature Fusion
        if self.use_intraday:
            stock_features = torch.cat(
                [
                    stock_temporal_features,
                    stock_intraday_temporal_features,
                    gated_stock_cs_features,
                ],
                dim=-1,
            )
        else:
            stock_features = torch.cat(
                [
                    stock_temporal_features,
                    gated_stock_cs_features,
                ],
                dim=-1,
            )

        fused_states = self.fusion_block(stock_features)

        # Industry-aware Inter-Stock Attention
        spatial_states = self.spatial_encoder(fused_states, stock_industry_ids)

        # Final Prediction on Spatial Representations
        predictions = self.prediction_head(spatial_states)
        return predictions
