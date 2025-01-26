########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################


def RWKV_Init(
    module, vocab_size, n_embd, rwkv_emb_scale
):  # fancy initialization of all lin & emb layer in the module
    for m in module.modules():
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        with torch.no_grad():
            name = "[unknown weight]"
            for (
                name,
                parameter,
            ) in module.named_parameters():  # find the name of the weight
                if id(m.weight) == id(parameter):
                    break

            shape = m.weight.data.shape
            gain = 1.0  # positive: gain for orthogonal, negative: std for normal
            scale = 1.0  # extra scale for gain

            if isinstance(m, nn.Linear):

                if m.bias is not None:
                    m.bias.data.zero_()
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == vocab_size and shape[1] == n_embd:  # final projection?
                    scale = rwkv_emb_scale

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == vocab_size and shape[1] == n_embd:  # token emb?
                    scale = rwkv_emb_scale

            if hasattr(m, "scale_init"):
                scale = m.scale_init

            # print(str(shape[0]).ljust(5), str(shape[1]).ljust(5), f'{round(scale, 2):g}'.ljust(4), name)

            gain *= scale
            if gain == 0:
                nn.init.zeros_(m.weight)  # zero init is great for some RWKV matrices
            elif gain > 0:
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                nn.init.normal_(m.weight, mean=0, std=-gain)


class RWKV_TimeMix(nn.Module):
    def __init__(self, layer_id, n_embd, n_attn, n_head, ctx_len):
        super().__init__()
        assert n_attn % n_head == 0
        self.layer_id = layer_id
        self.ctx_len = ctx_len
        self.n_head = n_head
        self.head_size = n_attn // n_head

        with torch.no_grad():  # initial time_w curves for better convergence
            ww = torch.ones(n_head, ctx_len)
            curve = torch.tensor(
                [-(ctx_len - 1 - i) for i in range(ctx_len)]
            )  # the distance
            for h in range(n_head):
                if h < n_head - 1:
                    decay_speed = math.pow(ctx_len, -(h + 1) / (n_head - 1))
                else:
                    decay_speed = 0.0
                ww[h] = torch.exp(curve * decay_speed)
                # print('layer', layer_id, 'head', h, 'decay_speed', round(decay_speed, 4), ww[h][:5].numpy(), '...', ww[h][-5:].numpy())
        self.time_w = nn.Parameter(ww)

        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(ctx_len, 1))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(n_embd, n_attn)
        self.value = nn.Linear(n_embd, n_attn)
        self.receptance = nn.Linear(n_embd, n_attn)

        # if .rwkv_tiny_attn > 0:
        #     self.tiny_att = RWKV_TinyAttn()

        self.output = nn.Linear(n_attn, n_embd)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT - 1 :]  # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat([self.time_shift(x[:, :, : C // 2]), x[:, :, C // 2 :]], dim=-1)
        # if hasattr(self, 'tiny_att'):
        #     tiny_att = self.tiny_att(x, self.mask)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        k = torch.clamp(k, max=30, min=-60)  # clamp extreme values. e^30 = 10^13
        k = torch.exp(k)
        sum_k = torch.cumsum(k, dim=1)

        kv = (k * v).view(B, T, self.n_head, self.head_size)

        wkv = (torch.einsum("htu,buhc->bthc", w, kv)).contiguous().view(B, T, -1)

        rwkv = torch.sigmoid(r) * wkv / sum_k

        rwkv = self.output(rwkv)
        # if hasattr(self, 'tiny_att'):
        #     rwkv += tiny_att

        return rwkv * self.time_gamma[:T, :]


class RWKV_ChannelMix(nn.Module):
    def __init__(self, layer_id, n_embd, n_ffn, hidden_sz, n_attn, n_head, ctx_len):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        hidden_sz = (
            5 * n_ffn // 2
        )  # can use smaller hidden_sz because of receptance gating
        self.key = nn.Linear(n_embd, hidden_sz)
        self.value = nn.Linear(n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, n_embd)
        self.receptance = nn.Linear(n_embd, n_embd)

        self.receptance.scale_init = 0
        self.weight.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()

        x = torch.cat([self.time_shift(x[:, :, : C // 2]), x[:, :, C // 2 :]], dim=-1)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        wkv = self.weight(F.mish(k) * v)  # i find mish is a bit better than gelu

        rwkv = torch.sigmoid(r) * wkv

        return rwkv


class RWKV_TinyAttn(nn.Module):  # extra tiny attention
    def __init__(self, n_embd, rwkv_tiny_attn, rwkv_tiny_head):
        super().__init__()
        self.d_attn = rwkv_tiny_attn
        self.n_head = rwkv_tiny_head
        self.head_size = self.d_attn // self.n_head

        self.qkv = nn.Linear(n_embd, self.d_attn * 3)
        self.out = nn.Linear(self.d_attn, n_embd)

    def forward(self, x, mask):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        if self.n_head > 1:
            q = q.view(B, T, self.n_head, self.head_size).transpose(
                1, 2
            )  # (B, T, C) -> (B, nh, T, hs)
            k = k.view(B, T, self.n_head, self.head_size).transpose(
                1, 2
            )  # (B, T, C) -> (B, nh, T, hs)
            v = v.view(B, T, self.n_head, self.head_size).transpose(
                1, 2
            )  # (B, T, C) -> (B, nh, T, hs)

        qk = (q @ k.transpose(-2, -1)) * (
            1.0 / math.sqrt(self.head_size)
        )  # (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        qk = qk.masked_fill(mask == 0, float("-inf"))
        qk = F.softmax(qk, dim=-1)
        qkv = qk @ v  # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)

        if self.n_head > 1:
            qkv = (
                qkv.transpose(1, 2).contiguous().view(B, T, -1)
            )  # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        return self.out(qkv)


########################################################################################################
# MHA_rotary: Multi-head Attention + Rotary Encoding + GeGLU FFN
########################################################################################################


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_len=None):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), -1)


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos[..., : q.shape[-2], :], sin[..., : q.shape[-2], :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class MHA_rotary(nn.Module):
    def __init__(self, n_embd, n_attn, n_head, ctx_len, layer_id, time_shift=False):
        super().__init__()
        self.layer_id = layer_id
        assert n_attn % n_head == 0
        self.n_head = n_head
        self.ctx_len = ctx_len
        self.head_size = n_attn // n_head

        if time_shift:
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.query = nn.Linear(n_embd, n_attn)
        self.key = nn.Linear(n_embd, n_attn)
        self.value = nn.Linear(n_embd, n_attn)

        self.register_buffer("mask", torch.tril(torch.ones(ctx_len, ctx_len)))

        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.output = nn.Linear(n_attn, n_embd)

    def forward(self, x):
        B, T, C = x.size()

        if hasattr(self, "time_shift"):
            x = torch.cat(
                [self.time_shift(x[:, :, : C // 2]), x[:, :, C // 2 :]], dim=-1
            )

        q = (
            self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        )  # (B, T, C) -> (B, nh, T, hs)
        k = (
            self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        )  # (B, T, C) -> (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        )  # (B, T, C) -> (B, nh, T, hs)

        q, query_pass = q[..., : self.rotary_ndims], q[..., self.rotary_ndims :]
        k, key_pass = k[..., : self.rotary_ndims], k[..., self.rotary_ndims :]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # rotary encoding
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)

        att = (q @ k.transpose(-2, -1)) * (
            1.0 / math.sqrt(k.size(-1))
        )  # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # causal mask
        att = F.softmax(att, dim=-1)  # softmax

        x = att @ v  # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = (
            x.transpose(1, 2).contiguous().view(B, T, -1)
        )  # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x)
        return x


class GeGLU(torch.nn.Module):
    def __init__(self, layer_id, n_embd, n_ffn, time_shift=False):
        super().__init__()
        self.layer_id = layer_id

        if time_shift:
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        hidden_sz = 3 * n_ffn
        self.key = nn.Linear(n_embd, hidden_sz)
        self.value = nn.Linear(n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        if hasattr(self, "time_shift"):
            x = torch.cat(
                [self.time_shift(x[:, :, : C // 2]), x[:, :, C // 2 :]], dim=-1
            )

        k = self.key(x)
        v = self.value(x)
        y = self.weight(F.gelu(k) * v)
        return y


########################################################################################################
# MHA_pro: with more tricks
########################################################################################################


class MHA_pro(nn.Module):
    def __init__(
        self, n_head, ctx_len, n_attn, n_embd, rotary_ndims, head_size, layer_id
    ):
        super().__init__()
        self.layer_id = layer_id
        assert n_attn % n_head == 0
        self.n_head = n_head
        self.ctx_len = ctx_len
        self.head_size = n_attn // n_head

        self.time_w = nn.Parameter(torch.ones(self.n_head, ctx_len))
        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(ctx_len, 1))
        self.register_buffer("mask", torch.tril(torch.ones(ctx_len, ctx_len)))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.query = nn.Linear(n_embd, n_attn)
        self.key = nn.Linear(n_embd, n_attn)
        self.value = nn.Linear(n_embd, n_attn)

        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.head_mix = nn.Conv2d(
            self.n_head, self.n_head, kernel_size=1, bias=False
        )  # talking heads

        self.output = nn.Linear(n_attn, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT - 1 :]  # w is now a circulant matrix
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat(
            [self.time_shift(x[:, :, : C // 2]), x[:, :, C // 2 :]], dim=-1
        )  # time-shift mixing
        q = (
            self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        )  # (B, T, C) -> (B, nh, T, hs)
        k = (
            self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        )  # (B, T, C) -> (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        )  # (B, T, C) -> (B, nh, T, hs)

        q, query_pass = q[..., : self.rotary_ndims], q[..., self.rotary_ndims :]
        k, key_pass = k[..., : self.rotary_ndims], k[..., self.rotary_ndims :]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # rotary encoding
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)

        att = (q @ k.transpose(-2, -1)) * (
            1.0 / math.sqrt(k.size(-1))
        )  # self-attention: (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # causal mask
        att = F.softmax(att, dim=-1)  # softmax
        att = att * w  # time-weighting
        att = self.head_mix(att)  # talking heads

        x = att @ v  # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = (
            x.transpose(1, 2).contiguous().view(B, T, -1)
        )  # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x) * self.time_gamma[:T, :]
        return x


########################################################################################################
# The GPT Model with our blocks
########################################################################################################


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1.0 / 2)
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (norm_x * self.dd + 1e-12)
        return self.weight * x_normed


class FixedNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1.0 / 2)

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        x_normed = x / (norm_x * self.dd + 1e-12)
        return x_normed


########################################################################################################


class Block(nn.Module):
    def __init__(
        self,
        n_embd,
        n_attn,
        n_head,
        ctx_len,
        n_ffn,
        hidden_sz,
        model_type="RWKV",
        layer_id=1,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if model_type == "RWKV":
            # self.ln1 = FixedNorm(.n_embd)
            # self.ln2 = FixedNorm(.n_embd)
            self.attn = RWKV_TimeMix(layer_id, n_embd, n_attn, n_head, ctx_len)
            self.mlp = RWKV_ChannelMix(
                layer_id, n_embd, n_ffn, hidden_sz, n_attn, n_head, ctx_len
            )

        elif model_type == "MHA_rotary":
            self.attn = MHA_rotary(n_embd, n_attn, n_head, ctx_len, layer_id)
            self.mlp = GeGLU(layer_id, n_embd, n_ffn, time_shift=True)

        elif model_type == "MHA_shift":
            self.attn = MHA_rotary(
                n_embd, n_attn, n_head, ctx_len, layer_id, time_shift=True
            )
            self.mlp = GeGLU(layer_id, n_embd, n_ffn, time_shift=True)

        elif model_type == "MHA_pro":
            self.attn = MHA_pro(
                n_head,
                ctx_len,
                n_attn,
                n_embd,
                rotary_ndims=-1,
                head_size=n_attn,
                layer_id=layer_id,
            )
            self.mlp = RWKV_ChannelMix(
                layer_id, n_embd, n_ffn, hidden_sz, n_attn, n_head, ctx_len
            )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


if __name__ == "__main__":
    x = torch.randn((16, 8, 256))
    rwkv = Block(256, 256, 4, 300, 256, 256)
    y = rwkv.forward(x)
    print(y.shape)
