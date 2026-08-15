import math

import torch
from torch import nn

from .embed import DataEmbedding


class TemporalAttention(nn.Module):
    def __init__(self, d_model, d_regime):
        super().__init__()
        self.q_trans = nn.Linear(d_model, d_model, bias=False)
        self.k_trans = nn.Linear(d_model, d_model, bias=False)
        self.scale = math.sqrt(d_model)

        # regime 条件偏置生成器
        self.regime_bias_proj = nn.Linear(d_regime, d_model, bias=False)

    def forward(self, z, regime_embedding):
        # z shape: [N, T, D]
        N, T, D = z.shape

        # 1. 最后一天作为 Query
        last_day_feat = z[:, -1, :]  # [N, D]
        query = self.q_trans(last_day_feat).unsqueeze(-1)  # [N, D, 1]

        # 2. 历史序列作为 Key
        keys = self.k_trans(z)  # [N, T, D]

        # 3. 计算内容相似度 logits
        scores = torch.matmul(keys, query).squeeze(-1)  # [N, T]
        scores = scores / self.scale

        # 4. 加入 regime-conditioned bias（逐时间步）
        #    regime_embedding -> [N, d_regime] -> [N, D]
        regime_query = self.regime_bias_proj(regime_embedding)  # [N, D]
        bias = (
            torch.matmul(keys, regime_query.unsqueeze(-1)).squeeze(-1) / self.scale
        )  # [N, T]

        scores = scores + bias

        # 5. 归一化并加权求和
        lam = torch.softmax(scores, dim=1).unsqueeze(1)  # [N, 1, T]
        output = torch.matmul(lam, z).squeeze(1)  # [N, D]

        return output


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


class RegimeEncoder(nn.Module):
    """
    市场状态特征映射为连续的Regime Embedding
    """

    def __init__(
        self,
        d_mkt: int,
        d_regime: int = 32,
        d_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d_mkt, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_regime),
        )

        self.skip_proj = nn.Linear(d_mkt, d_regime, bias=False)

        # ReZero初始化：初期以线性映射分支(skip_proj)为主，利于稳定收敛
        self.regime_scale = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, market_features):
        # market_features: [..., d_mkt]
        residual = self.skip_proj(market_features)
        nonlinear = self.encoder(market_features)

        return residual + self.regime_scale * nonlinear


class MarketFiLM(nn.Module):
    """
    使用统一 Regime Embedding 对特征流进行 FiLM 调制。

    h' = (1 + gamma) * h + beta
    """

    def __init__(
        self,
        d_regime: int,
        d_feat: int,
        d_hidden: int = 64,
    ):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Linear(d_regime, d_hidden),
            nn.SiLU(),
        )

        self.gamma = nn.Linear(d_hidden, d_feat)
        self.beta = nn.Linear(d_hidden, d_feat)

        # 初始保持恒等映射
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)

        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, regime_embedding):
        z = self.adapter(regime_embedding)

        gamma = self.gamma(z)
        beta = self.beta(z)

        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return (1.0 + gamma) * x + beta


class FusionBlock(nn.Module):
    def __init__(self, d_in, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = MLP(hidden_size=d_model, intermediate_size=d_model * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)
        return x + self.dropout(self.mlp(self.norm(x)))


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(p=dropout)

        self.o_proj = nn.Linear(d_model, d_model)
        self.mlp = MLP(d_model, 2 * d_model)
        self.input_layernorm = nn.LayerNorm(d_model, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x):
        residual = x
        hidden_states = self.input_layernorm(x)
        B, S, D = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        hidden_states = self.o_proj(out)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SAttention(nn.Module):
    def __init__(self, d_model, d_emb, nhead, dropout):
        super().__init__()
        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.o_proj = nn.Linear(d_model, d_model)

        self.mlp = MLP(d_model, 2 * d_model)

        self.input_layernorm = nn.LayerNorm(d_model, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x):
        """
        x: (N, D)
        """

        residual = x
        x = self.input_layernorm(x)

        # (N, D)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # -> (H, N, Dh)
        q = q.view(-1, self.nhead, self.head_dim).transpose(0, 1)
        k = k.view(-1, self.nhead, self.head_dim).transpose(0, 1)
        v = v.view(-1, self.nhead, self.head_dim).transpose(0, 1)

        # (H, N, N)
        attn_logits = torch.matmul(q, k.transpose(-1, -2))
        attn_logits *= self.scale

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # (H, N, Dh)
        attn_out = torch.matmul(attn_weights, v)

        # -> (N, D)
        hidden_states = attn_out.transpose(0, 1).contiguous().view(-1, self.d_model)

        hidden_states = self.o_proj(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        return residual + hidden_states


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
        num_labels,
        use_intraday=False,
    ):
        super().__init__()

        self.use_intraday = use_intraday

        # Feature Dimensions
        self.d_temporal_hidden = d_model // 4
        self.d_fusion_input = self.d_temporal_hidden + d_cs_feat
        if self.use_intraday:
            self.d_fusion_input += self.d_temporal_hidden

        # Market Regime Encoder
        d_regime = 32

        self.regime_encoder = RegimeEncoder(
            d_mkt=d_mkt_feat,
            d_regime=d_regime,
            d_hidden=64,
            dropout=0.1,
        )

        # Regime-Conditioned Feature Modulation
        self.ts_film = MarketFiLM(
            d_regime=d_regime,
            d_feat=self.d_temporal_hidden,
            d_hidden=64,
        )

        self.cs_film = MarketFiLM(
            d_regime=d_regime,
            d_feat=d_cs_feat,
            d_hidden=64,
        )

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
        self.temporal_attn = TemporalAttention(self.d_temporal_hidden, d_regime)

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

        # Fusion Encoder (Temporal + Cross-Sectional Feature Integration)
        self.fusion_block = FusionBlock(
            d_in=self.d_fusion_input,
            d_model=d_model,
            dropout=dropout,
        )

        # Spatial Encoder (Inter-stock)
        self.spatial_encoder = SAttention(
            d_model=d_model,
            d_emb=d_emb,
            nhead=s_nhead,
            dropout=dropout,
        )

        # Prediction heads
        self.prediction_heads = nn.ModuleList(
            [nn.Linear(d_model, 1, bias=False) for _ in range(num_labels)]
        )

    def forward(
        self,
        stock_ts_features,
        stock_cs_features,
        market_state_features,
        stock_industry_ids=None,
        stock_fund_features=None,
        stock_intraday_ts_features=None,
    ):
        # Encode Market Regime
        regime_embedding = self.regime_encoder(market_state_features)

        # Regime-conditioned CS Features
        stock_cs_features = self.cs_film(stock_cs_features, regime_embedding)

        # Intra-Stock Temporal Modeling (Daily)
        stock_temporal_embeds = self.data_embedding(stock_ts_features)
        stock_temporal_embeds = self.ts_film(stock_temporal_embeds, regime_embedding)
        stock_temporal_states = self.temporal_layers(stock_temporal_embeds)
        stock_temporal_features = self.temporal_attn(
            stock_temporal_states, regime_embedding
        )

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

        # Feature Fusion
        if self.use_intraday:
            stock_features = torch.cat(
                [
                    stock_temporal_features,
                    stock_intraday_temporal_features,
                    stock_cs_features,
                ],
                dim=-1,
            )
        else:
            stock_features = torch.cat(
                [
                    stock_temporal_features,
                    stock_cs_features,
                ],
                dim=-1,
            )

        fused_states = self.fusion_block(stock_features)

        # Industry-aware Inter-Stock Attention
        spatial_states = self.spatial_encoder(fused_states)

        # Final Prediction on Spatial Representations
        predictions = torch.cat(
            [head(spatial_states) for head in self.prediction_heads], dim=-1
        )
        return predictions
