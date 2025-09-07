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

        self.out_proj = nn.Linear(d_model, d_model)

        self.norm_x = nn.LayerNorm(d_model, eps=1e-5)
        self.norm_ind = nn.LayerNorm(d_emb, eps=1e-5)
        self.norm_ffn = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x, industry_embeds):
        # x: (N, D)  — 股票特征
        # industry_embeds: (N, d_emb) — 行业 embedding
        x_states = self.norm_x(x)
        ind_states = self.norm_ind(industry_embeds)

        # Q / K 用行业信息引导，V 只用股票特征
        qk_input = torch.cat([x_states, ind_states], dim=-1)
        q = self.qtrans(qk_input)
        k = self.ktrans(qk_input)
        v = self.vtrans(x_states)

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

        attn_output = torch.cat(attn_outputs, dim=-1)  # (N, D)
        attn_output = self.out_proj(attn_output)

        xt = x + attn_output
        out = xt + self.ffn(self.norm_ffn(xt))
        return out


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(nhead)])

        self.out_proj = nn.Linear(d_model, d_model)

        # Input LayerNorm
        self.norm_x = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm_ffn = LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout),
        )

    def forward(self, x):
        x_states = self.norm_x(x)
        q = self.qtrans(x_states)
        k = self.ktrans(x_states)
        v = self.vtrans(x_states)

        dim = int(self.d_model / self.nhead)
        attn_outputs = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim :]
                kh = k[:, :, i * dim :]
                vh = v[:, :, i * dim :]
            else:
                qh = q[:, :, i * dim : (i + 1) * dim]
                kh = k[:, :, i * dim : (i + 1) * dim]
                vh = v[:, :, i * dim : (i + 1) * dim]
            attn_weights = torch.softmax(
                torch.matmul(qh, kh.transpose(1, 2)) / math.sqrt(dim), dim=-1
            )
            attn_weights = self.attn_dropout[i](attn_weights)
            attn_outputs.append(torch.matmul(attn_weights, vh))

        attn_output = torch.concat(attn_outputs, dim=-1)
        attn_output = self.out_proj(attn_output)

        # FFN
        xt = x + attn_output
        output = xt + self.ffn(self.norm_ffn(xt))

        return output


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


class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x, context):
        """
        Args:
            x: (N, d_model) stock representations
            context: (N, d_model) market representations
        Returns:
            out: (N, d_model) stock updated with market context
        """
        # (N, d_model) -> (N, 1, d_model)
        q = x.unsqueeze(1)
        k = v = context.unsqueeze(1)

        attn_out, _ = self.cross_attn(q, k, v)  # (N, 1, d_model)
        x = x + attn_out.squeeze(1)  # residual
        x = self.norm(x)

        # Feed-forward
        out = self.ffn(x) + x
        return out


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):
        """
        参数:
        - input_dim: 输入特征维度
        - hidden_dims: list, 每一层的隐藏层维度, e.g. [128, 64]
        - output_dim: 输出维度 (分类任务一般是类别数)
        - dropout: dropout比例
        """
        super(MLP, self).__init__()

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())  # 可换成 GELU, LeakyReLU 等
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))  # 输出层
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
    ):
        super(PPNet, self).__init__()

        # Temporal processing layers
        self.temporal_proj = nn.Linear(d_ts_feat, d_model)
        self.temporal_pos_embed = PositionalEncoding(d_model)
        self.temporal_self_attn = TAttention(
            d_model=d_model, nhead=t_nhead, dropout=dropout
        )
        self.temporal_aggregator = TemporalAttention(d_model=d_model)

        # Cross-sectional processing layers
        self.cross_sectional_mlp = MLP(
            input_dim=d_cs_feat,
            hidden_dims=[2 * d_model],
            output_dim=d_model,
            dropout=dropout,
        )

        # Fusion layer
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        # Industry embedding
        self.industry_embed = nn.Embedding(256, d_emb)

        # Spatial attention
        self.spatial_attn = SAttention(
            d_model=d_model,
            d_emb=d_emb,
            nhead=s_nhead,
            dropout=dropout,
        )

        # Market processing
        self.market_proj = nn.Linear(d_market, d_model)

        # Cross attention (stock ↔ market)
        self.cross_attn = CrossAttention(
            d_model=d_model, nhead=s_nhead, dropout=dropout
        )

        # Prediction head
        self.prediction_head = nn.Linear(d_model, 1)

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
            stock_ts_features: (N, T, ts_feature_dim) temporal features of stocks
            stock_cs_features: (N, cs_feature_dim) cross-sectional features of stocks
            market_features: (N, market_feature_dim) market features
        Returns:
            predictions: (N, 1) prediction for each stock
        """

        # Process temporal stock features
        temporal_states = self.temporal_proj(stock_ts_features)  # (N, T, model_dim)
        temporal_embed = self.temporal_pos_embed(temporal_states)
        temporal_attn_output = self.temporal_self_attn(
            temporal_embed
        )  # Intra-stock temporal attention
        temporal_aggregated = self.temporal_aggregator(
            temporal_attn_output
        )  # (N, model_dim), aggregate over time

        # Process cross-sectional stock features
        cs_states = self.cross_sectional_mlp(stock_cs_features)  # (N, model_dim)

        # Fuse temporal and cross-sectional representations
        fused_states = self.fusion_proj(
            torch.cat([temporal_aggregated, cs_states], dim=-1)
        )  # (N, model_dim)

        # Embed industries and apply spatial attention
        industry_embeds = self.industry_embed(industry_indices)  # (N, embed_dim)
        stock_states = self.spatial_attn(
            fused_states, industry_embeds=industry_embeds
        )  # (N, model_dim)

        # Process market features and apply cross attention
        market_states = self.market_proj(market_features)  # (N, model_dim)
        attn_output = self.cross_attn(stock_states, market_states)  # (N, model_dim)

        # Generate final prediction
        predictions = self.prediction_head(attn_output)  # (N, 1)
        return predictions
