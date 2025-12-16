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


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(MLP, self).__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, d_emb):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.temperature = math.sqrt(self.head_dim)

        # Q, K, V projection
        self.q_proj = nn.Linear(d_model + d_emb, d_model, bias=False)
        self.k_proj = nn.Linear(d_model + d_emb, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(nhead)])

        self.o_proj = nn.Linear(d_model, d_model)

        self.mlp = MLP(d_model, 2 * d_model)
        self.norm_x = nn.LayerNorm(d_model, eps=1e-5)
        self.norm_ind = nn.LayerNorm(d_emb, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x, industry_embeds):
        # x: (N, D)  — 股票特征
        # industry_embeds: (N, d_emb) — 行业embedding
        residual = x
        x_states = self.norm_x(x)
        ind_states = self.norm_ind(industry_embeds)

        # Self Attention
        # Q / K 用行业信息引导，V 只用股票特征
        qk_input = torch.cat([x_states, ind_states], dim=-1)
        q = self.q_proj(qk_input)
        k = self.k_proj(qk_input)
        v = self.v_proj(x_states)

        # 多头拆分
        q = q.view(-1, self.nhead, self.head_dim)
        k = k.view(-1, self.nhead, self.head_dim)
        v = v.view(-1, self.nhead, self.head_dim)

        attn_outputs = []
        for i in range(self.nhead):
            qh = q[:, i, :]  # (N, head_dim)
            kh = k[:, i, :]  # (N, head_dim)
            vh = v[:, i, :]  # (N, head_dim)

            attn_weights = torch.softmax(
                torch.matmul(qh, kh.transpose(0, 1)) / self.temperature, dim=-1
            )
            attn_weights = self.attn_dropout[i](attn_weights)

            out = torch.matmul(attn_weights, vh)
            attn_outputs.append(out)

        hidden_states = torch.cat(attn_outputs, dim=-1)  # (N, D)
        hidden_states = self.o_proj(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


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
        self.input_layernorm = LayerNorm(d_model, eps=1e-5)
        self.post_attention_layernorm = LayerNorm(d_model, eps=1e-5)

    def forward(self, x):
        residual = x
        hidden_states = self.input_layernorm(x)

        # Self Attention
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        attn_outputs = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * self.head_dim :]
                kh = k[:, :, i * self.head_dim :]
                vh = v[:, :, i * self.head_dim :]
            else:
                qh = q[:, :, i * self.head_dim : (i + 1) * self.head_dim]
                kh = k[:, :, i * self.head_dim : (i + 1) * self.head_dim]
                vh = v[:, :, i * self.head_dim : (i + 1) * self.head_dim]
            attn_weights = torch.softmax(
                torch.matmul(qh, kh.transpose(1, 2)) / math.sqrt(self.head_dim), dim=-1
            )
            attn_weights = self.attn_dropout[i](attn_weights)
            attn_outputs.append(torch.matmul(attn_weights, vh))
        hidden_states = torch.concat(attn_outputs, dim=-1)
        hidden_states = self.o_proj(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


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
        d_ts_feat,
        d_cs_feat,
        d_market,
        d_emb,
        d_model,
        t_nhead,
        s_nhead,
        dropout,
        beta,
    ):
        super(PPNet, self).__init__()

        # Temporal layers
        self.temporal_hidden_dim = 64
        self.temporal_proj = nn.Linear(d_ts_feat, self.temporal_hidden_dim)
        self.temporal_pos_embed = PositionalEncoding(self.temporal_hidden_dim)
        self.temporal_self_attn = TAttention(
            d_model=self.temporal_hidden_dim, nhead=t_nhead, dropout=dropout
        )
        self.temporal_aggregator = TemporalAttention(d_model=self.temporal_hidden_dim)

        # Market layers
        self.market_gate = Gate(
            d_market, self.temporal_hidden_dim + d_cs_feat, beta=beta
        )

        # Fusion layers
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.temporal_hidden_dim + d_cs_feat, 2 * d_model),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(2 * d_model, d_model),
        )

        # Spatial layers
        self.industry_embed = nn.Embedding(256, d_emb)
        self.spatial_attn = SAttention(
            d_model=d_model,
            d_emb=d_emb,
            nhead=s_nhead,
            dropout=dropout,
        )

        # Prediction head
        self.prediction_head = nn.Linear(d_model, 1, bias=False)

    def forward(
        self,
        industry_indices,
        stock_ts_features,
        stock_cs_features,
        market_features,
    ):
        """
        Args:
            industry_indices: (N,) industry index for each stock
            stock_ts_features: (N, T, d_ts_feat) temporal features of stocks
            stock_cs_features: (N, d_cs_feat) cross-sectional features of stocks
            market_features: (N, d_market) market features
        Returns:
            predictions: (N, 1) prediction for each stock
        """

        # Process temporal stock features
        temporal_states = self.temporal_proj(
            stock_ts_features
        )  # (N, T, temporal_hidden_dim)
        temporal_with_pos = self.temporal_pos_embed(temporal_states)
        temporal_attn_output = self.temporal_self_attn(
            temporal_with_pos
        )  # Intra-stock temporal attention
        temporal_aggregated = self.temporal_aggregator(
            temporal_attn_output
        )  # (N, temporal_hidden_dim), aggregate over time

        # Concat temporal and cross-sectional representations
        concat_states = torch.cat([temporal_aggregated, stock_cs_features], dim=-1)

        # Apply gating to market features and modulate stock states
        gated_weights = self.market_gate(market_features)
        gated_states = (
            concat_states * gated_weights
        )  # (N, temporal_hidden_dim + d_cs_feat)

        # Fuse temporal and cross-sectional representations
        fused_states = self.fusion_proj(gated_states)  # (N, d_model)

        # Embed industries and apply spatial attention
        industry_embeds = self.industry_embed(industry_indices)  # (N, d_emb)
        attn_output = self.spatial_attn(
            fused_states, industry_embeds=industry_embeds
        )  # (N, d_model)

        # Generate final prediction
        predictions = self.prediction_head(attn_output)  # (N, 1)
        return predictions
