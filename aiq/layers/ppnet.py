import math

import torch
from torch import nn

from .embed import DataEmbedding


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 分离 Query 和 Key/Value 的映射空间
        self.q_trans = nn.Linear(d_model, d_model, bias=False)
        self.k_trans = nn.Linear(d_model, d_model, bias=False)
        self.scale = math.sqrt(d_model)

    def forward(self, z):
        # z shape: [N, T, D]

        # 1. 提取最后一天作为 Query，并映射到 Q 空间
        last_day_feat = z[:, -1, :]  # [N, D]
        query = self.q_trans(last_day_feat).unsqueeze(-1)  # [N, D, 1]

        # 2. 将整个历史序列映射到 K 空间
        keys = self.k_trans(z)  # [N, T, D]

        # 3. 计算注意力得分，并加入缩放因子防梯度消失
        scores = torch.matmul(keys, query).squeeze(-1)  # [N, T]
        scores = scores / self.scale  # 关键缩放！

        # 4. 归一化并加权求和
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


class MarketFiLM(nn.Module):
    """用 market state 对单个特征流做 FiLM 仿射调制。

    h' = (1 + gamma) * h + beta

    - 自带独立 market 编码器，不与其他特征流共享；
    - gamma/beta 零初始化 → 训练初始为恒等映射，不破坏原特征分布；
    - 自动适配 ts (3D: [N, T, d_feat]) 与 cs (2D: [N, d_feat]) 两种输入。
    """

    def __init__(self, d_mkt, d_feat, d_hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_mkt, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
        )
        self.gamma = nn.Linear(d_hidden, d_feat)
        self.beta = nn.Linear(d_hidden, d_feat)

        # 零初始化：gamma=0 → (1+0)=1，beta=0 → 恒等
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, cond):
        z = self.encoder(cond)
        gamma = self.gamma(z)
        beta = self.beta(z)
        if x.dim() == 3:
            # ts: [N, T, d_feat] → gamma/beta 广播到 T 轴
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
        self.temporal_attn = TemporalAttention(self.d_temporal_hidden)

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
        self.ts_film = MarketFiLM(d_mkt=d_mkt_feat, d_feat=d_ts_feat, d_hidden=64)
        self.cs_film = MarketFiLM(d_mkt=d_mkt_feat, d_feat=d_cs_feat, d_hidden=64)

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
        # Intra-Stock Temporal Modeling (Daily)
        stock_ts_features = self.ts_film(stock_ts_features, market_state_features)
        stock_temporal_embeds = self.data_embedding(stock_ts_features)
        stock_temporal_states = self.temporal_layers(stock_temporal_embeds)
        stock_temporal_features = self.temporal_attn(stock_temporal_states)

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
        gated_stock_cs_features = self.cs_film(stock_cs_features, market_state_features)

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
        spatial_states = self.spatial_encoder(fused_states)

        # Final Prediction on Spatial Representations
        predictions = torch.cat(
            [head(spatial_states) for head in self.prediction_heads], dim=-1
        )
        return predictions
