import math

import torch
from torch import nn

from .embed import DataEmbedding


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
            if i == self.nhead - 1:
                qh = q[:, :, i * self.head_dim :]
                kh = k[:, :, i * self.head_dim :]
                vh = v[:, :, i * self.head_dim :]
            else:
                qh = q[:, :, i * self.head_dim : (i + 1) * self.head_dim]
                kh = k[:, :, i * self.head_dim : (i + 1) * self.head_dim]
                vh = v[:, :, i * self.head_dim : (i + 1) * self.head_dim]

            # (N, T, head_dim) @ (N, head_dim, T) -> (N, T, T)
            attn_logits = torch.matmul(qh, kh.transpose(1, 2)) / math.sqrt(
                self.head_dim
            )

            attn_weights = torch.softmax(attn_logits, dim=-1)
            attn_weights = self.attn_dropout[i](attn_weights)
            attn_outputs.append(torch.matmul(attn_weights, vh))  # (N, T, head_dim)

        hidden_states = torch.concat(attn_outputs, dim=-1)
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
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.temperature = math.sqrt(self.head_dim)

        # Q, K, V projection
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(nhead)])

        self.o_proj = nn.Linear(d_model, d_model)

        self.mlp = MLP(d_model, 2 * d_model)
        self.input_layernorm = nn.LayerNorm(d_model, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(d_model, eps=1e-5)

    def _build_industry_decay(
        self, industry_indices: torch.Tensor, delta1: float = 0.65, delta2: float = 0.2
    ) -> torch.Tensor:
        """
        Build Industry Decay Matrix as described in the paper.

        Args:
            industry_indices: (N, 2) tensor where each row contains [l1_i, l2_i],
                            l1 = primary industry index, l2 = secondary industry index (integer encoded)
            delta1: Decay factor for same primary industry but different secondary industry (1 > delta1 > delta2 ≥ 0)
            delta2: Decay factor for different primary industries

        Returns:
            D: (N, N) Industry Decay Matrix where D[i,j] represents industry association weight between stock i and j
        """
        # Input validation
        N = industry_indices.shape[0]
        assert industry_indices.shape[1] == 2, "industry_indices must have shape (N, 2)"
        assert (
            1 > delta1 > delta2 >= 0
        ), "Decay factors must satisfy 1 > delta1 > delta2 ≥ 0"

        # Extract industry indices
        l1 = industry_indices[:, 0]  # (N,) primary industry indices
        l2 = industry_indices[:, 1]  # (N,) secondary industry indices

        # Create comparison matrices using broadcasting for vectorized operations
        # This replaces the nested loops for better performance
        l2_eq = l2.unsqueeze(1) == l2.unsqueeze(
            0
        )  # (N, N) boolean matrix for same secondary industry
        l1_eq = l1.unsqueeze(1) == l1.unsqueeze(
            0
        )  # (N, N) boolean matrix for same primary industry

        # Initialize matrix with delta2 (different primary industry case)
        D = torch.full(
            (N, N), delta2, dtype=torch.float32, device=industry_indices.device
        )

        # Update to delta1 for same primary but different secondary industry
        D[(l1_eq & ~l2_eq)] = delta1

        # Update to 1.0 for same secondary industry
        D[l2_eq] = 1.0

        # Set diagonal to 1.0 (self-connection)
        D.fill_diagonal_(1.0)

        return D

    def forward(self, x, industry_indices):
        # industry_decay: (N, N) — 行业衰减矩阵
        industry_decay = self._build_industry_decay(industry_indices).detach()

        # x: (N, D)  — 股票特征
        residual = x
        x_states = self.input_layernorm(x)

        # Self Attention
        q = self.q_proj(x_states)
        k = self.k_proj(x_states)
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

            attn_logits = torch.matmul(qh, kh.transpose(0, 1)) / self.temperature
            attn_logits = attn_logits + torch.log(industry_decay + 1e-8)
            attn_weights = torch.softmax(attn_logits, dim=-1)
            attn_weights = self.attn_dropout[i](attn_weights)

            attn_out = torch.matmul(attn_weights, vh)
            attn_outputs.append(attn_out)

        hidden_states = torch.cat(attn_outputs, dim=-1)  # (N, D)
        hidden_states = self.o_proj(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


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


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        last = z[:, -1, :]

        h = self.trans(z) # [N, T, D]
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1) / math.sqrt(h.shape[-1]) # [N, T, D] --> [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        context = torch.matmul(lam, z).squeeze(1) # [N, 1, T], [N, T, D] --> [N, D]

        return last + context


class FusionBlock(nn.Module):
    def __init__(self, d_in, d_model, dropout):
        super().__init__()
        self.mlp = MLP(hidden_size=d_in, intermediate_size=2 * d_in)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(2 * d_in, d_model)

    def forward(self, x):
        residual = x
        x = self.mlp(x)
        x = self.dropout(x)
        x = torch.cat([residual, x], dim=-1)
        x = self.proj(x)
        return x


class PPNet(nn.Module):
    def __init__(
        self,
        d_ts_feat,
        d_cs_feat,
        d_fund_feat,
        d_mkt_feat,
        d_emb,
        d_model,
        t_nhead,
        s_nhead,
        dropout,
        beta,
    ):
        super(PPNet, self).__init__()

        # Feature Dimensions
        self.d_temporal_hidden = d_model // 4
        self.d_fusion_input = self.d_temporal_hidden + d_cs_feat

        # Temporal Encoder (Intra-stock)
        self.data_embedding = DataEmbedding(
            c_in=d_ts_feat,
            d_model=self.d_temporal_hidden,
            dropout=dropout,
        )
        self.temporal_encoder = TAttention(
            d_model=self.d_temporal_hidden,
            nhead=t_nhead,
            dropout=dropout,
        )
        self.temporal_aggregator = TemporalAttention(
            d_model=self.d_temporal_hidden,
        )

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
        industry_indices,
        stock_ts_features,
        stock_cs_features,
        stock_fund_features,
        market_features,
    ):
        """
        Args:
            industry_indices: (N, 2) l1 and l2 industry index for each stock
            stock_ts_features: (N, T, d_ts_feat) temporal features of stocks
            stock_cs_features: (N, d_cs_feat) cross-sectional features of stocks
            stock_fund_features: (N, d_fund_feat) fundamental features of stocks
            market_features: (N, d_market) market features
        Returns:
            predictions: (N, 1) prediction for each stock
        """
        # Intra-Stock Temporal Modeling
        stock_temporal_embeds = self.data_embedding(stock_ts_features)
        stock_temporal_states = self.temporal_encoder(stock_temporal_embeds)
        stock_temporal_features = self.temporal_aggregator(stock_temporal_states)

        # Market-Conditioned Feature Gating
        gate_weights = self.market_gate(market_features)
        gated_stock_cs_features = stock_cs_features * gate_weights

        # Feature Fusion: Nonlinear Enhancement + Dimension Alignment
        stock_features = torch.cat(
            [stock_temporal_features, gated_stock_cs_features], dim=-1
        )
        fused_states = self.fusion_block(stock_features)

        # Industry-aware Inter-Stock Attention
        spatial_states = self.spatial_encoder(
            fused_states, industry_indices=industry_indices
        )

        # Final Prediction on Spatial Representations
        predictions = self.prediction_head(spatial_states)  # (N, 1)
        return predictions
