# import math

# import torch
# from torch import nn
# from torch.nn.modules.linear import Linear
# from torch.nn.modules.dropout import Dropout
# from torch.nn.modules.normalization import LayerNorm

# from aiq.layers.dlinear import DLinear, DLinear_Init
# from aiq.layers.rwkv import Block, RWKV_Init


# class SAttention(nn.Module):
#     def __init__(self, d_model, nhead, dropout):
#         super().__init__()

#         self.d_model = d_model
#         self.nhead = nhead
#         self.temperature = math.sqrt(self.d_model / nhead)

#         self.qtrans = nn.Linear(d_model, d_model, bias=False)
#         self.ktrans = nn.Linear(d_model, d_model, bias=False)
#         self.vtrans = nn.Linear(d_model, d_model, bias=False)

#         attn_dropout_layer = []
#         for i in range(nhead):
#             attn_dropout_layer.append(Dropout(p=dropout))
#         self.attn_dropout = nn.ModuleList(attn_dropout_layer)

#         # input LayerNorm
#         self.norm1 = LayerNorm(d_model, eps=1e-5)

#         # FFN layerNorm
#         self.norm2 = LayerNorm(d_model, eps=1e-5)
#         self.ffn = nn.Sequential(
#             Linear(d_model, d_model),
#             nn.ReLU(),
#             Dropout(p=dropout),
#             Linear(d_model, d_model),
#             Dropout(p=dropout),
#         )

#     def forward(self, x):
#         x = self.norm1(x)
#         q = self.qtrans(x).transpose(0, 1)
#         k = self.ktrans(x).transpose(0, 1)
#         v = self.vtrans(x).transpose(0, 1)

#         dim = int(self.d_model / self.nhead)
#         att_output = []
#         for i in range(self.nhead):
#             if i == self.nhead - 1:
#                 qh = q[:, :, i * dim :]
#                 kh = k[:, :, i * dim :]
#                 vh = v[:, :, i * dim :]
#             else:
#                 qh = q[:, :, i * dim : (i + 1) * dim]
#                 kh = k[:, :, i * dim : (i + 1) * dim]
#                 vh = v[:, :, i * dim : (i + 1) * dim]

#             atten_ave_matrixh = torch.softmax(
#                 torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1
#             )
#             if self.attn_dropout:
#                 atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
#             att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
#         att_output = torch.concat(att_output, dim=-1)

#         # FFN
#         xt = x + att_output
#         xt = self.norm2(xt)
#         att_output = xt + self.ffn(xt)

#         return att_output


# class Filter(nn.Module):
#     def __init__(self, d_input, d_output, seq_len, kernel=5, stride=5):
#         super().__init__()
#         self.d_input = d_input
#         self.d_output = d_output
#         self.seq_len = seq_len

#         self.trans = nn.Linear(d_input, d_output)

#         self.aggregate = nn.Conv1d(
#             d_output, d_output, kernel_size=kernel, stride=stride, groups=d_output
#         )

#         # 输入是[N, T, d_feat]
#         conv_feat = math.floor((self.seq_len - kernel) / stride + 1)

#         self.proj_out = nn.Linear(conv_feat, 1)

#     def forward(self, x):
#         x = self.trans.forward(x)  # [N, T, d_feat]
#         x_trans = x.transpose(-1, -2)  # [N, d_feat, T]
#         x_agg = self.aggregate.forward(x_trans)  # [N, d_feat, conv_feat]
#         out = self.proj_out.forward(x_agg)  # [N, d_feat, 1]
#         return out.transpose(-1, -2)  # [N, 1, d_feat]


# class TemporalAttention(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.trans = nn.Linear(d_model, d_model, bias=False)

#     def forward(self, z):
#         h = self.trans(z)  # [N, T, D]
#         query = h[:, -1, :].unsqueeze(-1)
#         lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
#         lam = torch.softmax(lam, dim=1).unsqueeze(1)
#         # [N, 1, T], [N, T, D] --> [N, 1, D]
#         output = torch.matmul(lam, z).squeeze(1)
#         return output


# class PPNet(nn.Module):
#     def __init__(
#         self,
#         d_feat=158,
#         d_model=256,
#         t_nhead=4,
#         s_nhead=2,
#         seq_len=8,
#         pred_len=1,
#         dropout=0.5,
#         gate_input_start_index=158,
#         gate_input_end_index=221,
#     ):
#         super().__init__()

#         self.d_feat = d_feat
#         self.d_model = d_model
#         self.n_attn = d_model
#         self.n_head = t_nhead

#         # market
#         self.gate_input_start_index = gate_input_start_index
#         self.gate_input_end_index = gate_input_end_index
#         self.d_gate_input = gate_input_end_index - gate_input_start_index  # F'
#         self.feature_gate = Filter(self.d_gate_input, self.d_feat, seq_len)

#         self.market_linear = nn.Linear(d_feat, d_model)

#         # instrument fundamental modules
#         self.num_industries = 192
#         self.ind_embedding_dim = 16
#         self.ind_embedding = nn.Embedding(self.num_industries, self.ind_embedding_dim)

#         self.num_fund_features = 20
#         self.fund_encoder = nn.Sequential(
#             nn.Linear(self.num_fund_features, 4 * self.num_fund_features),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(4 * self.num_fund_features, self.num_fund_features),
#             nn.Dropout(dropout),
#         )

#         # instrument price and volume modules
#         self.feat_to_model = nn.Linear(d_feat, d_model)

#         self.rwkv_trend = Block(
#             layer_id=0,
#             n_embd=self.d_model,
#             n_attn=self.n_attn,
#             n_head=self.n_head,
#             ctx_len=300,
#             n_ffn=self.d_model,
#             hidden_sz=self.d_model,
#         )
#         RWKV_Init(
#             self.rwkv_trend,
#             vocab_size=self.d_model,
#             n_embd=self.d_model,
#             rwkv_emb_scale=1.0,
#         )
#         self.rwkv_season = Block(
#             layer_id=0,
#             n_embd=self.d_model,
#             n_attn=self.n_attn,
#             n_head=self.n_head,
#             ctx_len=300,
#             n_ffn=self.d_model,
#             hidden_sz=self.d_model,
#         )
#         RWKV_Init(
#             self.rwkv_season,
#             vocab_size=self.d_model,
#             n_embd=self.d_model,
#             rwkv_emb_scale=1.0,
#         )

#         self.dlinear = DLinear(
#             seq_len=seq_len,
#             pred_len=seq_len,
#             enc_in=self.d_model,
#             kernel_size=3,
#             individual=False,
#             merge_outputs=False,
#         )
#         DLinear_Init(self.dlinear, min_val=-5e-2, max_val=8e-2)

#         self.trend_TC = nn.Sequential(
#             SAttention(
#                 d_model=d_model, nhead=s_nhead, dropout=dropout
#             ),  # Stock correlation
#             self.rwkv_trend,  # Time correlation
#         )
#         self.season_TC = nn.Sequential(
#             self.rwkv_season,  # Time correlation
#             SAttention(
#                 d_model=d_model, nhead=s_nhead, dropout=dropout
#             ),  # Stock correlation
#         )

#         self.temporal_attn = TemporalAttention(d_model=d_model)

#         hidden_dim = d_model // 2
#         self.regression_head = nn.Sequential(
#             # nn.Linear(d_model + self.num_fund_features, hidden_dim),
#             nn.Linear(d_model, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, pred_len)
#         )

#     def forward(self, x):
#         # x: [N, T, D]
#         ind_class = x[:, -1, 0].long()
#         cat_feats = self.ind_embedding(ind_class)

#         fund_feats = x[:, :, 1:5].mean(dim=1)
#         fund_feats = torch.cat([cat_feats, fund_feats], dim=1)
#         fund_feats = self.fund_encoder(fund_feats)

#         cont_feats = x[:, :, 5 : self.gate_input_start_index]
#         cont_feats = self.feat_to_model(cont_feats)
#         trend_feat, season_feat = self.dlinear(cont_feats)

#         gate_input = x[:, :, self.gate_input_start_index : self.gate_input_end_index]
#         market_feat = self.feature_gate(gate_input)

#         trend_out = self.trend_TC(trend_feat) + self.market_linear(market_feat)
#         season_out = self.season_TC(season_feat)
#         temporal_out = self.temporal_attn(trend_out + season_out)

#         # fused_out = torch.cat([temporal_out, fund_feats], dim=-1)
#         # output = self.regression_head(fused_out)
#         output = self.regression_head(temporal_out)
#         return output
import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math


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
        d_feat,
        d_model,
        t_nhead,
        s_nhead,
        dropout,
        gate_input_start_index,
        gate_input_end_index,
        seq_len,
        pred_len,
        beta=5,
    ):
        super(PPNet, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = gate_input_end_index - gate_input_start_index  # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=dropout),
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=dropout),
            TemporalAttention(d_model=d_model),
            # decoder
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        src = x[:, :, 5 : self.gate_input_start_index]  # N, T, D
        gate_input = x[:, -1, self.gate_input_start_index : self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)

        output = self.layers(src)

        return output