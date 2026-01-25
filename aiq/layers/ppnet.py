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
    def __init__(self, d_in, d_model, nhead, dropout):
        super().__init__()

        self.enc_embedding = DataEmbedding(d_in, d_model, dropout)

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
        # Embedding
        x_enc = self.enc_embedding(x)

        # Self Attention
        residual = x_enc
        hidden_states = self.input_layernorm(x_enc)

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
    def __init__(self, d_model, nhead, dropout, d_emb):
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

    def forward(self, x, industry_decay):
        # x: (N, D)  — 股票特征
        # industry_decay: (N, N) — 行业衰减矩阵
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
            attn_logits += torch.log(industry_decay)
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


class TemporalAttention(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)
        self.context_vector = nn.Parameter(torch.Tensor(d_model, 1))
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.context_vector)

    def forward(self, z):
        # z: [N, T, D]
        h = torch.tanh(self.trans(z))
        scores = torch.matmul(h, self.context_vector).squeeze(-1)  # [N, T]
        attn_weights = torch.softmax(scores, dim=1).unsqueeze(1)  # [N, 1, T]
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, z).squeeze(1)  # [N, D]
        return output


class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()

        self.d_output = d_output
        self.t = beta

        self.encoder = nn.Sequential(
            nn.Linear(d_input, d_output),
            nn.SiLU(),
            nn.Linear(d_output, d_output),
        )

    def forward(self, x):
        x_enc = self.encoder(x)

        # 特征缩放因子
        x_scale = torch.softmax(x_enc / self.t, dim=-1)
        x_scale = self.d_output * x_scale

        return x_scale


class PPNet(nn.Module):
    def __init__(
        self,
        d_ts_feat,
        d_cs_feat,
        d_fund_feat,
        d_market,
        d_emb,
        d_model,
        t_nhead,
        s_nhead,
        dropout,
        beta,
    ):
        super(PPNet, self).__init__()

        # Feature hidden dimensions
        self.cs_hidden_dim = d_model
        self.temporal_hidden_dim = d_model // 4
        self.fund_hidden_dim = d_model // 16

        # Temporal layers
        self.temporal_attn = TAttention(
            d_in=d_ts_feat,
            d_model=self.temporal_hidden_dim,
            nhead=t_nhead,
            dropout=dropout,
        )
        self.temporal_aggregator = TemporalAttention(
            d_model=self.temporal_hidden_dim, dropout=dropout
        )

        # Market layers
        self.market_gate = Gate(d_market, d_cs_feat, beta=beta)

        # Feature encoders
        self.ts_proj = nn.Sequential(
            nn.Linear(self.temporal_hidden_dim, self.temporal_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.temporal_hidden_dim),
            nn.Dropout(dropout),
        )
        self.cs_proj = nn.Sequential(
            nn.Linear(d_cs_feat, self.cs_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.cs_hidden_dim),
            nn.Dropout(dropout),
        )
        self.fund_proj = nn.Sequential(
            nn.Linear(d_fund_feat, self.fund_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.fund_hidden_dim),
            nn.Dropout(dropout),
        )

        # Fusion layers
        self.fusion_proj = nn.Sequential(
            nn.Linear(
                self.temporal_hidden_dim + self.cs_hidden_dim + self.fund_hidden_dim,
                2 * d_model,
            ),
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
        self.prediction_head = nn.Linear(d_model, 1)

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
        # Temporal Feature Extraction
        temporal_features = self.temporal_attn(
            stock_ts_features
        )  # Intra-stock temporal attention
        temporal_states = self.temporal_aggregator(
            temporal_features
        )  # (N, temporal_hidden_dim), aggregate over time

        # Modulate stock cross-sectional features based on market context
        feature_gated_weights = self.market_gate(market_features)
        gated_cs_features = stock_cs_features * feature_gated_weights

        # Map heterogeneous features into a unified latent space
        temporal_states = self.ts_proj(temporal_states)
        cs_states = self.cs_proj(gated_cs_features)
        fund_states = self.fund_proj(stock_fund_features)

        # Fuse temporal，cross-sectional and fundamental representations
        fused_states = torch.cat([temporal_states, cs_states, fund_states], dim=-1)
        fused_states = self.fusion_proj(fused_states)  # (N, d_model)

        # Industry-aware attention to capture inter-stock correlations
        industry_decay = self._build_industry_decay(industry_indices)
        spatial_states = self.spatial_attn(
            fused_states, industry_decay=industry_decay
        )  # (N, d_model)

        # Map refined representations to final predictions
        predictions = self.prediction_head(spatial_states)  # (N, 1)
        return predictions
