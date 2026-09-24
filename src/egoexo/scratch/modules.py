"""手撸层：把 ``functional.py`` 里的公式包成可训练模块。

设计原则（很重要，决定了「手撸」的边界在哪）
--------------------------------------------
我们**继续用 ``nn.Module`` 当脚手架**，但只用它做四件与数学无关的事：

    参数注册（self.weight = nn.Parameter(...)）
    递归收集参数（model.parameters()）
    设备搬运（model.to(device)）
    存档/读档（state_dict / load_state_dict）

「数学」（矩阵乘、softmax、归一化、反向）全部由我们在 ``forward`` 里自己写。

为什么不连 ``nn.Module`` 也一起换掉：因为那样会**失去最强的验证手段**。
现在这些手撸层的**参数名和官方逐字一致**，于是可以

    scratch_model.load_state_dict(lib_model.state_dict())      # 直接吃官方 checkpoint

然后比较两者输出是否逐元素相等。这是比「看起来对」强得多的证据。
真要体验「连 autograd 都没有」的世界，看 ``scratch/micrograd.py``（纯 Python，零 torch）。

参数名必须与官方对齐的清单（改名字就会 load_state_dict 失败）
-------------------------------------------------------------
    ScratchLinear                    weight, bias
    ScratchLayerNorm                 weight, bias
    ScratchEmbedding                 weight
    ScratchMultiheadAttention        in_proj_weight, in_proj_bias, out_proj.{weight,bias}
    ScratchTransformerEncoderLayer   self_attn.*, linear1.*, linear2.*, norm1.*, norm2.*
    ScratchTransformerEncoder        layers.{i}.*, norm.{weight,bias}
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from . import functional as Fs
from . import init as I

__all__ = [
    "ScratchLinear",
    "ScratchLayerNorm",
    "ScratchGELU",
    "ScratchReLU",
    "ScratchDropout",
    "ScratchEmbedding",
    "ScratchSinusoidalPosEmb",
    "ScratchMultiheadAttention",
    "ScratchTransformerEncoderLayer",
    "ScratchTransformerEncoder",
    "ScratchAttentionPool",
]


# ---------------------------------------------------------------- 基础层


class ScratchLinear(nn.Module):
    """``y = x @ Wᵀ + b``。官方的 ``nn.Linear`` 就是这两行加一个参数容器。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, manual_bwd: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.manual_bwd = manual_bwd
        I.reset_linear(self.weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.manual_bwd:
            from .autograd_ops import linear_fn

            return linear_fn(x, self.weight, self.bias)
        return Fs.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class ScratchLayerNorm(nn.Module):
    """对最后一维归一化。``normalized_shape`` 只用来校验，不参与计算。"""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, manual_bwd: bool = False):
        super().__init__()
        if isinstance(normalized_shape, (tuple, list)):
            assert len(normalized_shape) == 1, "本实现只支持对最后一维归一化"
            normalized_shape = int(normalized_shape[0])
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.manual_bwd = manual_bwd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.normalized_shape, f"最后一维 {x.size(-1)} != {self.normalized_shape}"
        if self.manual_bwd:
            from .autograd_ops import layer_norm_fn

            return layer_norm_fn(x, self.weight, self.bias, self.eps)
        return Fs.layer_norm(x, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"


class ScratchGELU(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return Fs.gelu(x, self.approximate)


class ScratchReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return Fs.relu(x)


class ScratchDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 训练/评估靠 nn.Module 的 self.training 位切换（model.train()/eval() 会递归设置）
        return Fs.dropout(x, self.p, training=self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class ScratchEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        I.normal_(self.weight, 0.0, 1.0)  # 对齐 nn.Embedding 默认初始化 N(0,1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return Fs.embedding(ids, self.weight)


class ScratchSinusoidalPosEmb(nn.Module):
    """位置编码。参数为零，不参与训练，也不会有梯度 —— 纯查表加法。"""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = Fs.sinusoidal_pe(max_len, dim)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype)


# ---------------------------------------------------------------- 多头注意力


class ScratchMultiheadAttention(nn.Module):
    """对应 ``nn.MultiheadAttention``。

    参数布局刻意和官方**完全一致**，这样才能直接 load_state_dict 对拍。
    官方把 q/k/v 三个投影拼成一个大矩阵 ``in_proj_weight (3E, E)``，
    好处是 self-attention 时一次矩阵乘搞定三次投影。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
        manual_bwd: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} 必须能被 num_heads {num_heads} 整除"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = float(dropout)
        self.batch_first = batch_first
        self.manual_bwd = manual_bwd

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            # 官方把两个投影的 bias 合成一个 (3E,)；out_proj 的 bias 单独初始化成 0
            self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = ScratchLinear(embed_dim, embed_dim, bias=bias)
        I.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            I.constant_(self.in_proj_bias, 0.0)
            I.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """签名与 ``nn.MultiheadAttention.forward`` 对齐（只实现项目用到的分支）。

        key_padding_mask (B, Lk) **bool，True = 屏蔽**。注意这个语义和本项目
        ``kp_mask`` / ``frame_mask``（1 = 有效）**相反**，调用处要取反，
        ``models.py`` 里写的就是 ``~visual_mask``。
        """
        if not self.batch_first:
            # 官方默认 (L, B, E)；我们内部统一按 (B, L, E) 算，前后转置一下
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        if attn_mask is not None or is_causal:
            raise NotImplementedError("本实现只支持 key_padding_mask（项目里不需要 attn_mask / causal）")

        if self.manual_bwd:
            from .autograd_ops import multi_head_attention_fn

            out, weights = multi_head_attention_fn(
                query, key, value,
                self.in_proj_weight, self.in_proj_bias,
                self.out_proj.weight, self.out_proj.bias,
                self.num_heads, key_padding_mask, need_weights,
                self.dropout, self.training,
            )
        else:
            out, weights = Fs.multi_head_attention(
                query, key, value,
                self.in_proj_weight, self.in_proj_bias,
                self.out_proj.weight, self.out_proj.bias,
                self.num_heads, key_padding_mask, need_weights,
                self.dropout, self.training,
            )

        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, weights


# ---------------------------------------------------------------- Transformer 编码器


class ScratchTransformerEncoderLayer(nn.Module):
    """对应 ``nn.TransformerEncoderLayer``（只支持 norm_first，也就是 Pre-LN）。

    Pre-LN 的结构（``norm_first=True``）：

        x = x + Dropout(Attn(LN(x)))            ← 残差在**归一化之后**介入
        x = x + Dropout(FFN(LN(x)))

    Post-LN（原始 Transformer）是把 LN 放在残差**之后**：

        x = LN(x + Attn(x))
        x = LN(x + FFN(x))

    这个顺序不是细节：Post-LN 在深层需要 warmup 才能训起来，Pre-LN 不需要。
    本项目用 Pre-LN。
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = False,
        norm_first: bool = False,
        manual_bwd: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.norm_first = norm_first
        self.self_attn = ScratchMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                   manual_bwd=manual_bwd)
        self.linear1 = ScratchLinear(d_model, dim_feedforward, manual_bwd=manual_bwd)
        self.dropout = ScratchDropout(dropout)
        self.linear2 = ScratchLinear(dim_feedforward, d_model, manual_bwd=manual_bwd)
        self.norm1 = ScratchLayerNorm(d_model, manual_bwd=manual_bwd)
        self.norm2 = ScratchLayerNorm(d_model, manual_bwd=manual_bwd)
        self.dropout1 = ScratchDropout(dropout)
        self.dropout2 = ScratchDropout(dropout)
        self.activation = ScratchGELU() if activation == "gelu" else ScratchReLU()

    def _sa_block(self, x, key_padding_mask):
        out = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(out)

    def _ff_block(self, x):
        # 顺序是 linear1 -> activation -> dropout -> linear2
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, is_causal: bool = False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x


class ScratchTransformerEncoder(nn.Module):
    """对应 ``nn.TransformerEncoder``：把 N 层串起来（可选的最后再归一化一次）。

    ⚠️ 实测：当前 torch（2.12）里 ``nn.TransformerEncoder(..., norm=None)`` 的
    ``self.norm`` 就是 ``None``，不会自动补一个 LayerNorm。
    所以本项目 ``TemporalEncoder`` 才在自己外面套了 ``self.norm = nn.LayerNorm(dim)``。
    """

    def __init__(self, encoder_layer: nn.Module, num_layers: int, enable_nested_tensor: bool = False, norm=None):
        super().__init__()
        # deepcopy：每一层是同一个模板的**独立副本**（各自有各自的权重）
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor

    def forward(self, src: torch.Tensor, mask=None, src_key_padding_mask=None, is_causal=None):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            out = self.norm(out)
        return out


class ScratchAttentionPool(nn.Module):
    """注意力池化（可学习 query 版），对应 models.py 里的 ``AttentionPool``。"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = ScratchLayerNorm(dim)
        self.drop = ScratchDropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x (B, T, D), mask (B, T) 1=有效 → (B, D)。"""
        q = self.query.expand(x.size(0), -1, -1)
        d = x.size(-1)
        scores = (q * (1.0 / math.sqrt(d))) @ x.transpose(1, 2)  # (B,1,T)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        w = Fs.masked_softmax(scores, dim=-1)
        # 注意：官方 models.py 的版本是 dropout 加在 attention 权重上，这里保持一致
        w = self.drop(w)
        return self.norm((w @ x).squeeze(1))
