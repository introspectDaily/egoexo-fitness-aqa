"""实现切换开关：同一份模型代码，可以在「官方封装」和「手撸实现」之间切换。

为什么用**工厂 + 开关**而不是复制两份 models.py
------------------------------------------------
1. 复制两份迟早会漂移（改了一边忘了另一边），这是最隐蔽的 bug 来源；
2. 一份代码两种实现，才能做**同一 checkpoint 双向加载**的对拍 —— 这是最强的正确性证据；
3. 训练默认仍走官方实现，历史实验数字（README §7）不受影响，可复现。

CLI
---
    --impl lib           默认。全用官方 nn.*
    --impl scratch       全手撸
    --scratch-parts LIST 覆盖 --impl，精确指定手撸哪些部分

可选手撸的部分（``--scratch-parts``）
------------------------------------
    linear   nn.Linear              → ScratchLinear
    norm     nn.LayerNorm           → ScratchLayerNorm
    act      nn.GELU                → ScratchGELU
    dropout  nn.Dropout             → ScratchDropout
    embed    nn.Embedding           → ScratchEmbedding
    attn     nn.MultiheadAttention  → ScratchMultiheadAttention
    encoder  nn.TransformerEncoder* → ScratchTransformerEncoder*
    pool     注意力池化             → ScratchAttentionPool
    loss     F.binary_cross_entropy_with_logits / F.cross_entropy → 手撸版
    optim    torch.optim.AdamW + LambdaLR → ScratchAdamW + 手写调度
    clip     F 的梯度裁剪           → 手写全局范数裁剪
    bwd      以上全部改用**手写 backward**（torch.autograd.Function）

``bwd`` 会自动带上它依赖的 ``linear,norm,attn,loss``（手写 backward 就实现在这几个里）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["ALL_PARTS", "Impl", "LIB_IMPL", "SCRATCH_IMPL", "resolve_impl", "describe_impl"]

# 手撸的粒度。顺序 == 建议的学习顺序（先线性/归一化，再注意力，最后反向）
ALL_PARTS = (
    "linear",
    "norm",
    "act",
    "dropout",
    "embed",
    "attn",
    "encoder",
    "pool",
    "loss",
    "clip",
    "optim",
    "bwd",
)

# bwd 只是一个「模式开关」，它必须依附在这些层上才能真正生效
_BWD_IMPLIES = ("linear", "norm", "attn", "loss")


@dataclass(frozen=True)
class Impl:
    """描述「哪些部分用手撸实现」。不可变，可安全地塞进 dataclass 配置里。"""

    parts: frozenset[str] = frozenset()
    gelu_approximate: str = "none"

    # ------------------------------------------------------------ 查询
    def on(self, part: str) -> bool:
        return part in self.parts

    @property
    def is_scratch(self) -> bool:
        return bool(self.parts)

    @property
    def manual_bwd(self) -> bool:
        return self.on("bwd")

    def with_gelu(self, approximate: str) -> "Impl":
        """切 GELU 的精确版 / tanh 近似版（``--gelu tanh`` 用，纯为了看差异）。"""
        return replace(self, gelu_approximate=approximate)

    # ------------------------------------------------------------ 层工厂
    def linear(self, in_features: int, out_features: int, bias: bool = True):
        if self.on("linear"):
            from .modules import ScratchLinear

            return ScratchLinear(in_features, out_features, bias, manual_bwd=self.manual_bwd)
        import torch.nn as nn

        return nn.Linear(in_features, out_features, bias=bias)

    def layernorm(self, normalized_shape: int, eps: float = 1e-5):
        if self.on("norm"):
            from .modules import ScratchLayerNorm

            return ScratchLayerNorm(normalized_shape, eps, manual_bwd=self.manual_bwd)
        import torch.nn as nn

        return nn.LayerNorm(normalized_shape, eps=eps)

    def gelu(self):
        if self.on("act"):
            from .modules import ScratchGELU

            return ScratchGELU(self.gelu_approximate)
        import torch.nn as nn

        return nn.GELU(approximate=self.gelu_approximate)

    def dropout(self, p: float):
        if self.on("dropout"):
            from .modules import ScratchDropout

            return ScratchDropout(p)
        import torch.nn as nn

        return nn.Dropout(p)

    def embedding(self, num_embeddings: int, embedding_dim: int):
        if self.on("embed"):
            from .modules import ScratchEmbedding

            return ScratchEmbedding(num_embeddings, embedding_dim)
        import torch.nn as nn

        return nn.Embedding(num_embeddings, embedding_dim)

    def mha(self, embed_dim: int, num_heads: int, dropout: float = 0.0, batch_first: bool = False):
        if self.on("attn"):
            from .modules import ScratchMultiheadAttention

            return ScratchMultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first,
                                            manual_bwd=self.manual_bwd)
        import torch.nn as nn

        return nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)

    def encoder_layer(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "gelu",
        batch_first: bool = True,
        norm_first: bool = True,
    ):
        if self.on("encoder"):
            from .modules import ScratchTransformerEncoderLayer

            return ScratchTransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation,
                batch_first=batch_first, norm_first=norm_first, manual_bwd=self.manual_bwd,
            )
        import torch.nn as nn

        return nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, batch_first=batch_first, norm_first=norm_first,
        )

    def encoder(self, layer, num_layers: int, enable_nested_tensor: bool = False):
        if self.on("encoder"):
            from .modules import ScratchTransformerEncoder

            return ScratchTransformerEncoder(layer, num_layers, enable_nested_tensor=enable_nested_tensor)
        import torch.nn as nn

        # ⚠️ 注意 torch 2.12 里 norm 默认是 None（不会自动补 LayerNorm），
        #    所以 TemporalEncoder 自己外面又套了一层 norm。两版行为一致。
        return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=enable_nested_tensor)

    def attention_pool(self, dim: int, dropout: float):
        if self.on("pool"):
            from .modules import ScratchAttentionPool

            return ScratchAttentionPool(dim, dropout)
        from ..models import AttentionPool  # 项目原有的手写实现（同样只用张量运算）

        return AttentionPool(dim, dropout)


LIB_IMPL = Impl(frozenset())
SCRATCH_IMPL = Impl(frozenset(ALL_PARTS))


def resolve_impl(impl: str = "lib", parts: str | None = None, gelu: str = "none") -> Impl:
    """把 CLI 字符串解析成 :class:`Impl`。

    parts 支持 ``all`` / ``none`` / 逗号分隔的部件名。传了 parts 就以它为准，忽略 impl。
    """
    if parts:
        names = {p.strip() for p in parts.split(",") if p.strip()}
        if "all" in names:
            names = set(ALL_PARTS)
        if "none" in names:
            names = set()
        unknown = names - set(ALL_PARTS)
        if unknown:
            raise SystemExit(
                f"--scratch-parts 里有未知部件 {sorted(unknown)}。\n可选：{', '.join(ALL_PARTS)}（或 all / none）"
            )
        if "bwd" in names:
            names |= set(_BWD_IMPLIES)
        return Impl(frozenset(names), gelu_approximate=gelu)

    if impl == "scratch":
        return SCRATCH_IMPL.with_gelu(gelu)
    return LIB_IMPL.with_gelu(gelu)


def describe_impl(impl: Impl) -> str:
    """人读的一行描述，训练日志里打出来。"""
    if not impl.parts:
        return "lib (全部使用官方 nn.* 实现)"
    ordered = [p for p in ALL_PARTS if impl.on(p)]
    extra = f", gelu={impl.gelu_approximate}" if impl.on("act") and impl.gelu_approximate != "none" else ""
    return f"scratch [{', '.join(ordered)}{extra}]"
