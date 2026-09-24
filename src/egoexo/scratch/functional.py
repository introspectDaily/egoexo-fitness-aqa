"""手撸算子：只用 torch.Tensor 的矩阵乘 / 逐元素函数，不用任何 ``nn.*`` 或 ``F.*``。

为什么单独开一个文件
--------------------
``nn.Linear`` / ``nn.MultiheadAttention`` / ``F.gelu`` 这些封装把**数学**藏在了 C++ 里。
想知道 Transformer 到底在算什么，就得自己把下面这几行写一遍：

    y    = x @ Wᵀ + b                                  # Linear
    y    = (x - μ) / √(σ² + ε) * γ + β                  # LayerNorm
    a    = softmax(Q Kᵀ / √d_head) V                    # Attention
    y    = 0.5x(1 + erf(x/√2))                          # GELU

然后**逐元素对拍**官方实现（``python -m egoexo.cli parity``）。
不写对拍的手撸只能加深错觉，不能加深理解 —— 详见 docs/手撸对照手册.md。

约定（不遵守就对拍不过）
------------------------
1. 全是**纯函数**：无状态、不持有参数，权重由调用方显式传入。
2. 反向传播交给 ``torch.autograd``（想手写 backward 走 ``scratch/autograd_ops.py``）。
3. 形状约定与 PyTorch 官方**逐字对齐**，每个函数的 docstring 里都写清楚。
4. 这里只借 ``torch.Tensor`` 做「数组 + 自动微分」，不借 ``nn.Module`` 做「层」。
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "linear",
    "layer_norm",
    "gelu",
    "relu",
    "softmax",
    "log_softmax",
    "dropout",
    "embedding",
    "sinusoidal_pe",
    "masked_softmax",
    "scaled_dot_product_attention",
    "multi_head_attention",
    "attention_pool",
    "binary_cross_entropy_with_logits",
    "cross_entropy",
    "logsumexp",
]


# ---------------------------------------------------------------- 基本线性/归一化


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``y = x @ Wᵀ + b``。

    x      (... , in_features)
    weight (out_features, in_features)      ← 注意是「输出在前」，和 nn.Linear 一致
    bias   (out_features,) 或 None
    →      (... , out_features)

    为什么要转置：nn.Linear 存的是 W[out, in]，因为这样 forward 是 `x @ W.T`
    （行向量 × 列向量），比存 W[in, out] 再做 `W.T @ x` 对缓存更友好。
    反向时 dL/dW = dL/dyᵀ @ x 也天然是 [out, in] 的形状，不需要转置。
    """
    y = x @ weight.t()
    if bias is not None:
        y = y + bias
    return y


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """对**最后一维**做 LayerNorm（和 ``nn.LayerNorm(D)`` 等价）。

    μ  = mean(x, -1)
    σ² = var(x, -1, unbiased=False)     ← 除以 N，不是 N-1！官方也是「有偏」方差
    y  = (x - μ) / √(σ² + ε) * γ + β

    三个容易写错的点：
    1. 方差要用**有偏**（除以 N）。用 ``unbiased=True`` 会让小 N 时有可见偏差，
       对拍立刻暴露（相对误差 ~1/N）。
    2. ε 加在**开根号里面**：``sqrt(var + eps)``，不是 ``sqrt(var) + eps``。
    3. 归一化维度是最后一维，这里显式写 ``dim=-1``，不用 ``normalized_shape`` 那套。
    """
    # 用 float32 计算统计量再转回原 dtype：官方 layer_norm 在 fp16/bf16 下也是
    # 用 float 累加，low-precision 下直接算 var 会掉精度。
    orig_dtype = x.dtype
    x32 = x.float()
    mu = x32.mean(dim=-1, keepdim=True)
    var = x32.var(dim=-1, unbiased=False, keepdim=True)
    xhat = (x32 - mu) / torch.sqrt(var + eps)
    out = xhat
    if weight is not None:
        out = out * weight.float()
    if bias is not None:
        out = out + bias.float()
    return out.to(orig_dtype)


# ---------------------------------------------------------------- 激活


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """GELU。

    approximate="none"  : y = 0.5x(1 + erf(x/√2))            ← nn.GELU() 的默认，精确版
    approximate="tanh"  : y = 0.5x(1 + tanh(√(2/π)(x+0.044715x³)))

    ⚠️ nn.TransformerEncoderLayer(activation="gelu") 用的是**精确版**（``F.gelu``），
       不是 tanh 近似。想对拍就选 none；tanh 版留着自己看「近似到什么程度」。
    """
    if approximate == "tanh":
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp_min(x, 0.0)


# ---------------------------------------------------------------- softmax 家族


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """数值稳定的 softmax：先减每行的最大值再取指数。

    不减 max 会怎样：exp(100) = inf，整行变 nan。而 softmax 对「整体平移不变」，
    所以减 max 是**恒等变换**，只解决数值问题、不改数学。

    额外处理「整行都是 -inf」（key padding 全被 mask 掉）的情况：
    官方 SDPA 会输出 0，这里也输出 0，避免 0/0=nan 把整个 loss 污染。
    """
    m = x.amax(dim=dim, keepdim=True)
    # 全 -inf 行：amax 也是 -inf，减它会得到 nan。换成 0，后面 exp(-inf)=0，和为 0。
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    e = torch.exp(x - m)
    s = e.sum(dim=dim, keepdim=True)
    tiny = torch.finfo(e.dtype).tiny
    return e / s.clamp_min(tiny)


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """log(softmax(x))，同样先减 max：``x - m - log Σ exp(x - m)``。

    直接在 ``log(softmax(x))`` 上算会在概率极小时下溢成 -inf，
    而它正是交叉熵要用的量，所以必须单独实现。
    """
    m = x.amax(dim=dim, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    shifted = x - m
    return shifted - torch.logsumexp(shifted, dim=dim, keepdim=True)


def masked_softmax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """对「可能整行都是 -inf」的 scores 做 softmax；被整行 mask 掉的行输出**全 0**。

    这是个必须显式处理的边界：padding 位置会把某一行所有 key 都填成 -inf，
    此时 ``softmax`` 是 0/0 = nan。官方的 ``F.scaled_dot_product_attention``
    以及 MHA 的融合路径都输出 0，这里对齐它。

    ⚠️ 不能直接把 -inf 换成 0 再 softmax —— 那样整行会变成**均匀分布 1/Lk**，
       是「所有 key 等权」的意思，和「没有 key」完全不同。必须 softmax 之后再置 0。
    """
    all_masked = torch.isneginf(scores.amax(dim=dim, keepdim=True))
    if not bool(all_masked.any()):
        return softmax(scores, dim=dim)
    safe = scores.masked_fill(all_masked, 0.0)
    return softmax(safe, dim=dim).masked_fill(all_masked, 0.0)


def logsumexp(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """log Σ exp(x)，同样用减 max 的技巧。"""
    m = x.amax(dim=dim, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    out = torch.log(torch.exp(x - m).sum(dim=dim, keepdim=True)) + m
    return out if keepdim else out.squeeze(dim)


def dropout(x: torch.Tensor, p: float = 0.5, training: bool = True, generator: torch.Generator | None = None) -> torch.Tensor:
    """倒置 dropout（inverted dropout）：训练时按 1-p 概率保留，再除以 (1-p)。

    除以 (1-p) 是关键：它保证**期望不变**，所以推理时直接原样输出即可，
    不需要在推理时再乘 (1-p)。这就是「inverted」的含义。
    """
    if not training or p == 0.0:
        return x
    if p == 1.0:
        return torch.zeros_like(x)
    keep = 1.0 - p
    mask = torch.bernoulli(torch.full_like(x, keep), generator=generator)
    return x * mask / keep


def embedding(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """查表：``weight[ids]``。

    ids    (...)   int64
    weight (num_embeddings, embedding_dim)
    →      (..., embedding_dim)

    一行就能写完，但值得单独列出来：所谓 Embedding 层就是个查表，
    没有任何魔法（反向时是对被查到的那几行做 scatter-add，
    没被查到的行梯度为 0 —— 这就是「稀疏梯度」的由来）。
    """
    return weight[ids]


# ---------------------------------------------------------------- 位置编码


def sinusoidal_pe(length: int, dim: int, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """正弦位置编码（Attention Is All You Need 原式），返回 (length, dim)。

        PE[p, 2i]   = sin(p / 10000^(2i/dim))
        PE[p, 2i+1] = cos(p / 10000^(2i/dim))

    实现上把 ``1/10000^(2i/dim)`` 写成 ``exp(2i/dim * (-log 10000))``：
    指数里做乘法比反复求幂更稳，也和论文/常见实现一致。
    要求 dim 为偶数（奇偶两两成对）。
    """
    assert dim % 2 == 0, f"dim 必须是偶数，实际 {dim}"
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)  # (L,1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))  # (dim/2,)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype)


# ---------------------------------------------------------------- 注意力


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_padding_mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """单头注意力：``softmax(Q Kᵀ/√d) V``。

    q (B, H, Lq, Dh)
    k (B, H, Lk, Dh)
    v (B, H, Lk, Dv)
    key_padding_mask (B, Lk) bool，True = 该位置是 padding，要屏蔽
    →  (B, H, Lq, Dv)

    两个关键点：
    1. 缩放放在 **Q 上**而不是结果上（``q/√d`` 而不是 ``attn/√d``）。
       为什么除以 √d：q·k 是 Dh 个 ~N(0,1) 项的乘积之和，方差≈Dh，
       不缩放的 logits 量级 ~√Dh，softmax 会饱和成 one-hot、梯度消失。
    2. mask 必须在 **softmax 之前**填 -inf。填 0 是错的（softmax(0)=1 而不是 0），
       填一个大负数也只是近似。
    """
    dh = q.size(-1)
    if scale is None:
        scale = 1.0 / math.sqrt(dh)
    scores = (q * scale) @ k.transpose(-1, -2)  # (B,H,Lq,Lk)
    if key_padding_mask is not None:
        # (B,Lk) -> (B,1,1,Lk) 广播到所有 head 和所有 query
        scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
    attn = masked_softmax(scores, dim=-1)
    return attn @ v


def multi_head_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor | None,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor | None,
    num_heads: int,
    key_padding_mask: torch.Tensor | None = None,
    need_weights: bool = False,
    dropout_p: float = 0.0,
    training: bool = False,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """多头注意力，参数布局与 ``nn.MultiheadAttention`` **完全一致**。

    这是整个项目里最值得手撸的一段。官方实现里 q/k/v 投影是「拼成一个大矩阵一次乘」
    的（``in_proj_weight`` 形状 ``(3E, E)``），拆开看就是三次 ``linear``：

        W = [W_q ; W_k ; W_v]          每一块 (E, E)
        Q = q @ W_qᵀ + b_q
        K = k @ W_kᵀ + b_k
        V = v @ W_vᵀ + b_v

    然后

        head_i = attention(Q_i, K_i, V_i)                    i = 1..H
        out    = concat(head_1..head_H) @ W_oᵀ + b_o

    「多头」不是一个玄学结构，就是**把 E 维切成 H 段各自做注意力，再拼回来**。
    唯一的额外自由度是每个头有自己的投影子矩阵，于是不同的头可以关注不同的模式。

    query (B, Lq, E)，key/value (B, Lk, E)（self-attention 时三者同一张量）
    in_proj_weight (3E, E)，in_proj_bias (3E,)
    out_proj_weight (E, E)，out_proj_bias (E,)
    → ((B, Lq, E), attn_weights or None)

    一个必须知道的坑：``nn.MultiheadAttention`` 对 query 和 key/value **共享权重**
    的两种情况走不同代码路径（打包一次乘 vs 拆成 3 次乘），数学上等价但浮点结果
    会差 ~1e-7。对拍时按这个量级设阈值。
    """
    E = query.size(-1)
    head_dim = E // num_heads
    assert E % num_heads == 0, f"embed_dim {E} 不能被 num_heads {num_heads} 整除"

    if key is value and query is key:
        # self-attention：一次大矩阵乘，再切三段
        proj = linear(query, in_proj_weight, in_proj_bias)
        q, k, v = proj.split(E, dim=-1)
    elif key is value:
        # cross-attention（q 与 k/v 不同）：分块乘，避免重复投影 k/v
        w_q, w_kv = in_proj_weight.split([E, 2 * E], dim=0)
        b_q, b_kv = (in_proj_bias.split([E, 2 * E], dim=0) if in_proj_bias is not None else (None, None))
        q = linear(query, w_q, b_q)
        kv = linear(key, w_kv, b_kv)
        k, v = kv.split(E, dim=-1)
    else:
        w_q, w_k, w_v = in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = (in_proj_bias.chunk(3, dim=0) if in_proj_bias is not None else (None, None, None))
        q = linear(query, w_q, b_q)
        k = linear(key, w_k, b_k)
        v = linear(value, w_v, b_v)

    B, Lq = q.shape[0], q.shape[1]
    Lk = k.size(1)
    # (B, L, E) -> (B, L, H, Dh) -> (B, H, L, Dh)
    # 这一步是「多头」的全部实现：把 E 拆成 H×Dh，把 head 提到 batch 维旁边。
    q = q.reshape(B, Lq, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(B, Lk, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, Lk, num_heads, head_dim).transpose(1, 2)

    dh = head_dim
    if scale is None:
        scale = 1.0 / math.sqrt(dh)
    scores = (q * scale) @ k.transpose(-1, -2)  # (B,H,Lq,Lk)
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
    attn = masked_softmax(scores, dim=-1)
    if training and dropout_p > 0:
        attn = dropout(attn, dropout_p, training=True)
    ctx = attn @ v  # (B,H,Lq,Dh)

    ctx = ctx.transpose(1, 2).reshape(B, Lq, E)  # 拼回头
    out = linear(ctx, out_proj_weight, out_proj_bias)
    if not need_weights:
        return out, None
    # 官方默认 average_attn_weights=True：在 head 维取平均
    return out, attn.mean(dim=1)


def attention_pool(x: torch.Tensor, query: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """注意力池化：用一个**可学习的 query 向量**去给每帧打权重，再加权求和。

    x     (B, T, D)
    query (B, 1, D)  或 (1, 1, D)
    mask  (B, T) bool，True = 有效帧（注意和 MHA 的 key_padding_mask 语义**相反**）
    →     (B, D)

    和 mean pooling 的区别：mean 是「假设每帧一样重要」，
    这里让模型自己学「哪一帧最说明问题」——对「某一帧姿势就错了」这类静态错误是关键。
    数学上就是一次单头、单 query 的 cross-attention，最后不做投影。
    """
    d = x.size(-1)
    scores = (query * (1.0 / math.sqrt(d))) @ x.transpose(1, 2)  # (B,1,T)
    if mask is not None:
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
    w = masked_softmax(scores, dim=-1)
    return (w @ x).squeeze(1)


# ---------------------------------------------------------------- 损失用的底层算子


def binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """带 logits 的（可选加权的）二元交叉熵，逐元素，不做 reduction。

    BCE 的朴素写法是 ``-(y log σ(x) + (1-y) log(1-σ(x)))``，
    当 σ(x) 下溢到 0 时 ``log 0 = -inf``，直接爆。

    所以用 **log-sum-exp 恒等式**改写（``max(x,0) - x*y + log(1+exp(-|x|))``）：

        BCE(x, y) = max(x, 0) - x·y + log(1 + exp(-|x|))

    它和朴素式在数学上完全相等，但 ``exp(-|x|)`` 永远 ≤1，不会溢出。

    pos_weight 作用在正类的 log 项上（PyTorch 的定义）：
        BCE_w(x, y) = (1 + (w-1)y) · [max(x,0) - x·y + log(1+exp(-|x|))]
    等价于「正样本的 loss 乘 w」。这就是缓解 22% 正类不平衡的那个旋钮。
    """
    # clamp_min 是为了数值安全：|x| 很小时 log1p(exp(-|x|)) ≈ log(2)，无精度问题
    loss = torch.clamp_min(logits, 0.0) - logits * target + torch.log1p(torch.exp(-logits.abs()))
    if pos_weight is not None:
        loss = loss * (1.0 + (pos_weight - 1.0) * target)
    return loss


def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """多分类交叉熵（标量，取 mean）。

    logits (B, C)
    target (B,)  int64
    → scalar

    用 ``log_softmax`` + gather，而不是 ``softmax`` 后再取 log：
    前者数值稳定，后者在 logit 差距大时会得到 log(0) = -inf（再乘 -1 变 inf）。
    这是「为什么框架都提供 F.cross_entropy 而不是 softmax+log」的标准答案。
    """
    logp = log_softmax(logits, dim=-1)
    return -logp.gather(dim=-1, index=target.unsqueeze(-1)).mean()
