"""手写反向传播：用 ``torch.autograd.Function`` 自己推 forward 和 backward。

到这一步才真正碰到「深度学习」本身。前面 ``functional.py`` 里虽然没调 ``F.*``，
但 backward 还是 autograd 替我们算的。这里把这四个最常见算子的**导数手推一遍**，
再用 ``torch.autograd.gradcheck``（数值微分）验证。

    LinearFn                y = xWᵀ + b
    LayerNormFn             y = (x-μ)/√(σ²+ε)·γ + β
    BCEWithLogitsFn         softplus 形式的 BCE（含 pos_weight）
    SoftmaxCrossEntropyFn   log_softmax + NLL
    SDPAWithMaskFn          softmax(QKᵀ/√d + mask) V     ← 注意力的核心

启用方式：``--scratch-parts bwd``（或 ``--impl scratch --scratch-parts ...,bwd``）。

为什么要靠 gradcheck 而不是「看起来对」
----------------------------------------
手推的公式里最容易错的是：
  - 归一化类算子忘了「减均值」那一项（σ 和 μ 都依赖 x，`dμ/dx` 不是 0）；
  - 沿哪一维求和搞错（batch 维求和 vs 特征维求和）；
  - 缩放系数（1/√d）该乘在 Q 上还是结果上，反向时漏乘。

这些错误 forward 完全看不出来，只有和数值微分（或 autograd）逐元素对比才会暴露。
``cli parity`` 会把「手写 backward 的梯度」和「autograd 的梯度」放在一起比。
"""

from __future__ import annotations

import math

import torch

from . import functional as Fs

__all__ = [
    "LinearFn",
    "LayerNormFn",
    "BCEWithLogitsFn",
    "SoftmaxCrossEntropyFn",
    "SDPAWithMaskFn",
    "linear_fn",
    "layer_norm_fn",
    "bce_with_logits_fn",
    "cross_entropy_fn",
    "sdpa_fn",
    "multi_head_attention_fn",
]


# ---------------------------------------------------------------- Linear


class LinearFn(torch.autograd.Function):
    """``y = x @ Wᵀ + b``。

    形状：x (..., in)，W (out, in)，b (out,)，y (..., out)

    反向推导（设上游梯度 g = ∂L/∂y，形状 (..., out)）：

        ∂L/∂W = Σ_batch gᵀ ⊗ x        → 把所有前导维摊平成 batch 后：`g2ᵀ @ x2`，形状 (out, in)
        ∂L/∂b = Σ_batch g            → 沿除最后一维以外的所有维求和
        ∂L/∂x = g @ W                → (..., out) @ (out, in) = (..., in)

    ⚠️ 三个常见错误：
      1. 忘了转置：``dW = x2ᵀ @ g2`` 形状是 (in, out)，和 W 不匹配。
      2. dW 忘了对 batch 求和（不然形状多一维，或者 batch=1 时看不出来，batch>1 就崩）。
      3. db 对**最后一维**求和（应该是对前面的所有维求和）。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
        ctx.save_for_backward(x, weight)
        return Fs.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        x2 = x.reshape(-1, x.size(-1))              # (N, in)
        g2 = grad_out.reshape(-1, grad_out.size(-1))  # (N, out)
        grad_w = g2.t() @ x2                        # (out, in)
        grad_x = grad_out @ weight                  # (..., in)
        grad_b = g2.sum(dim=0) if ctx.needs_input_grad[2] else None
        return grad_x, grad_w, grad_b


def linear_fn(x, weight, bias=None):
    return LinearFn.apply(x, weight, bias)


# ---------------------------------------------------------------- LayerNorm


class LayerNormFn(torch.autograd.Function):
    """``y = (x - μ)/√(σ² + ε) · γ + β``，对最后一维做。

    反向推导（这是最值得推一遍的算子，因为 μ 和 σ 都依赖 x）：

        记 inv = 1/√(σ²+ε)，x̂ = (x-μ)·inv，y = x̂·γ + β，上游梯度 g = ∂L/∂y

        ∂L/∂γ = Σ_batch g·x̂
        ∂L/∂β = Σ_batch g
        d x̂   = g·γ
        ∂L/∂σ² = Σ dx̂·(x-μ)·(-1/2)·inv³
        ∂L/∂μ  = Σ dx̂·(-inv)  +  ∂L/∂σ² · mean(-2(x-μ))
                = -inv·Σ dx̂                        ← 第二项恒为 0，因为 Σ(x-μ)=0
        ∂L/∂x  = dx̂·inv  +  ∂L/∂σ²·2(x-μ)/N  +  ∂L/∂μ/N

    那个「第二项恒为 0」就是 LayerNorm 反向里最常见的陷阱来源：
    有人因为 Σ(x-μ)=0 就以为 μ 那一项也能省，**不能** —— μ 的梯度通过
    x̂ = (x-μ)inv 已经传进来了，`∂L/∂μ` 那一项还得老老实实除以 N 广播回去。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float):
        orig_dtype = x.dtype
        x32 = x.float()
        mu = x32.mean(dim=-1, keepdim=True)
        var = x32.var(dim=-1, unbiased=False, keepdim=True)
        inv = torch.rsqrt(var + eps)
        xhat = (x32 - mu) * inv
        ctx.save_for_backward(xhat, inv, weight)
        out = xhat
        if weight is not None:
            out = out * weight.float()
        if bias is not None:
            out = out + bias.float()
        return out.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        xhat, inv, weight = ctx.saved_tensors
        g = grad_out.float()

        grad_w = grad_b = None
        if weight is not None:
            if ctx.needs_input_grad[1]:
                grad_w = (g * xhat).sum(dim=tuple(range(g.dim() - 1)))
            dxhat = g * weight.float()
        else:
            dxhat = g

        n = xhat.size(-1)
        # x = xhat / inv + mu  →  (x - μ) = xhat / inv
        centered = xhat / inv
        dvar = (dxhat * centered * (-0.5) * inv.pow(3)).sum(dim=-1, keepdim=True)
        dmu = (-inv * dxhat).sum(dim=-1, keepdim=True)  # 第二项（mean(-2(x-μ))）恒为 0
        grad_x = dxhat * inv + dvar * 2.0 * centered / n + dmu / n

        if ctx.needs_input_grad[2] and ctx.saved_tensors[2] is not None:
            grad_b = g.sum(dim=tuple(range(g.dim() - 1)))
        return grad_x.to(grad_out.dtype), grad_w, grad_b, None


def layer_norm_fn(x, weight=None, bias=None, eps: float = 1e-5):
    return LayerNormFn.apply(x, weight, bias, eps)


# ---------------------------------------------------------------- BCE


class BCEWithLogitsFn(torch.autograd.Function):
    """逐元素、无 reduction 的带 logits BCE：

        L = max(x,0) - x·y + log(1 + exp(-|x|))
        L_w = (1 + (w-1)·y) · L                  # pos_weight 只在正类上放大

    反向推导：softplus(x) 的导数是 σ(x)，所以

        ∂L/∂x = σ(x) - y
        ∂L_w/∂x = (1 + (w-1)·y) · (σ(x) - y)

    这个式子漂亮得值得记住：**BCE 对 logit 的梯度就是「预测概率 − 标签」**。
    线性回归对输出的梯度也是「预测 − 标签」，所以两者在梯度形式上是一回事。
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor | None):
        sig = torch.sigmoid(logits)
        ctx.save_for_backward(sig, target, pos_weight)
        return Fs.binary_cross_entropy_with_logits(logits, target, pos_weight)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        sig, target, pos_weight = ctx.saved_tensors
        grad_logits = (sig - target) * grad_out
        if pos_weight is not None:
            grad_logits = grad_logits * (1.0 + (pos_weight - 1.0) * target)
        return grad_logits, None, None


def bce_with_logits_fn(logits, target, pos_weight=None):
    return BCEWithLogitsFn.apply(logits, target, pos_weight)


# ---------------------------------------------------------------- 交叉熵


class SoftmaxCrossEntropyFn(torch.autograd.Function):
    """``mean(-log softmax(x)[y])``。反向同样只有一行：``(softmax(x) - onehot) / B``。

    这就是「为什么分类头不用手写 backward」—— 它和 BCE 的梯度形状完全一样。
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, target: torch.Tensor):
        logp = Fs.log_softmax(logits, dim=-1)
        ctx.save_for_backward(logits, target)
        return -logp.gather(dim=-1, index=target.unsqueeze(-1)).mean()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        logits, target = ctx.saved_tensors
        p = Fs.softmax(logits, dim=-1)
        grad = p.clone()
        grad.scatter_add_(-1, target.unsqueeze(-1), torch.full_like(target.unsqueeze(-1), -1.0, dtype=grad.dtype))
        grad = grad / logits.size(0) * grad_out   # 除以 B 是因为 forward 里 .mean()
        return grad, None


def cross_entropy_fn(logits, target):
    return SoftmaxCrossEntropyFn.apply(logits, target)


# ---------------------------------------------------------------- 注意力核心


class SDPAWithMaskFn(torch.autograd.Function):
    """单头注意力 ``O = masked_softmax(QKᵀ/√d) V``，含手写 backward。

    q (B,H,Lq,Dh)，k (B,H,Lk,Dh)，v (B,H,Lk,Dh)，key_padding_mask (B,Lk) bool（True=屏蔽）

    反向推导（设 g = ∂L/∂O）：

        dA = g @ Vᵀ                                  # (B,H,Lq,Lk)
        dZ = A ⊙ (dA − Σ_last(dA ⊙ A))               ← softmax 的雅可比：对角 − 外积
        dQ = (dZ @ K) · scale                        # scale 乘在 Q 上，所以 dQ 也要乘
        dK = dZᵀ @ (Q·scale)
        dV = Aᵀ @ g

    最值得记的是 softmax 那一行：设 a = softmax(z)，则

        ∂a_i/∂z_j = a_i(δ_ij − a_j)   ⟹   ∂L/∂z = a ⊙ (∂L/∂a − ⟨∂L/∂a, a⟩)

    也就是「先像乘出来，再减去自己在 a 上的加权平均」。写不出来这一行，
    就没真正理解注意力。

    另外注意 `dQ = (dZ @ K) · scale`：因为 forward 里缩放加在 Q 上。
    如果当初写成 `attn/√d`，反向就得改在这里 —— 这就是「实现细节决定反向公式」的例子。
    """

    @staticmethod
    def forward(ctx, q, k, v, key_padding_mask, scale):
        dh = q.size(-1)
        if scale is None:
            scale = 1.0 / math.sqrt(dh)
        scores = (q * scale) @ k.transpose(-1, -2)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        a = Fs.masked_softmax(scores, dim=-1)
        ctx.save_for_backward(q, k, v, a)
        ctx.scale = scale
        return a @ v

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, a = ctx.saved_tensors
        scale = ctx.scale
        grad_a = grad_out @ v.transpose(-1, -2)
        tmp = (grad_a * a).sum(dim=-1, keepdim=True)
        grad_z = a * (grad_a - tmp)
        grad_q = (grad_z @ k) * scale
        grad_k = grad_z.transpose(-1, -2) @ (q * scale)
        grad_v = a.transpose(-1, -2) @ grad_out
        return grad_q, grad_k, grad_v, None, None


def sdpa_fn(q, k, v, key_padding_mask=None, scale=None):
    return SDPAWithMaskFn.apply(q, k, v, key_padding_mask, scale)


def multi_head_attention_fn(
    query,
    key,
    value,
    in_proj_weight,
    in_proj_bias,
    out_proj_weight,
    out_proj_bias,
    num_heads,
    key_padding_mask=None,
    need_weights: bool = True,
    dropout_p: float = 0.0,
    training: bool = False,
    scale=None,
):
    """多头注意力 = 「线性投影（手写反向）→ 拆头 → SDPA（手写反向）→ 拼回 → 线性投影」。

    这个函数本身是**普通函数**，不是 ``autograd.Function``：因为它的每一块
    （``linear_fn``、``sdpa_fn``）都已经各自处理好了反向，autograd 负责把它们串起来。
    硬把整个 MHA 塞进一个 Function 反而不利于理解 —— 真实框架也是这么分块的。
    """
    E = query.size(-1)
    head_dim = E // num_heads

    if key is value and query is key:
        proj = linear_fn(query, in_proj_weight, in_proj_bias)
        q, k, v = proj.split(E, dim=-1)
    elif key is value:
        w_q, w_kv = in_proj_weight.split([E, 2 * E], dim=0)
        b_q, b_kv = (in_proj_bias.split([E, 2 * E], dim=0) if in_proj_bias is not None else (None, None))
        q = linear_fn(query, w_q, b_q)
        kv = linear_fn(key, w_kv, b_kv)
        k, v = kv.split(E, dim=-1)
    else:
        w_q, w_k, w_v = in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = (in_proj_bias.chunk(3, dim=0) if in_proj_bias is not None else (None, None, None))
        q = linear_fn(query, w_q, b_q)
        k = linear_fn(key, w_k, b_k)
        v = linear_fn(value, w_v, b_v)

    B, Lq = q.shape[0], q.shape[1]
    Lk = k.size(1)
    q = q.reshape(B, Lq, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(B, Lk, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, Lk, num_heads, head_dim).transpose(1, 2)

    ctx_out = sdpa_fn(q, k, v, key_padding_mask, scale)
    if training and dropout_p > 0:
        ctx_out = Fs.dropout(ctx_out, dropout_p, training=True)
    ctx_out = ctx_out.transpose(1, 2).reshape(B, Lq, E)
    out = linear_fn(ctx_out, out_proj_weight, out_proj_bias)
    if not need_weights:
        return out, None
    # 需要权重时重算一次（省内存，代价是可忽略的）
    dh = head_dim
    s = 1.0 / math.sqrt(dh) if scale is None else scale
    scores = (q * s) @ k.transpose(-1, -2)
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
    a = Fs.masked_softmax(scores, dim=-1)
    return out, a.mean(dim=1)
