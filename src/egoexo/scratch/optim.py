"""手撸优化器与学习率调度（不依赖 ``torch.optim``）。

AdamW 的全部内容就是下面这些加减乘除。值得手写一遍的理由：
``torch.optim.AdamW(..., weight_decay=0.05)`` 看起来只是一个参数，
但它到底是「加进梯度」还是「直接衰减权重」，直接决定训练行为 ——
Adam+L2 和 AdamW 是**两个不同的算法**，论文里混用会复现不出来。

        m_t = β₁ m_{t-1} + (1-β₁) g_t                          # 一阶动量：梯度的 EMA
        v_t = β₂ v_{t-1} + (1-β₂) g_t²                          # 二阶动量：梯度平方的 EMA
        m̂ = m_t / (1-β₁ᵗ),  v̂ = v_t / (1-β₂ᵗ)                  # 偏差修正（初始化是 0，早期偏小）
        θ = θ·(1 - lr·wd) - lr·m̂ / (√v̂ + ε)                    # ← 解耦权重衰减（AdamW）
                                                              #   Adam+L2 是 θ -= lr(m̂/(√v̂+ε) + wd·θ)

偏差修正为什么必要：m 从 0 开始，第 1 步 m = 0.1g（β₁=0.9），远小于真实梯度；
除以 (1-β₁ᵗ) = 0.1 就把它拉回来了。少了这一步，前期学习率等于被缩小 10 倍。
"""

from __future__ import annotations

import math

import torch

__all__ = ["ScratchAdamW", "WarmupCosineLR", "cosine_with_warmup", "clip_grad_norm_"]


class ScratchAdamW:
    """对齐 ``torch.optim.AdamW`` 的更新公式（含解耦权重衰减）。"""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        self.param_groups = [{"params": [p for p in params if p.requires_grad], "lr": lr,
                              "betas": betas, "eps": eps, "weight_decay": weight_decay}]
        self.state: dict[int, dict] = {}
        self._step = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if set_to_none:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self._step += 1
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            # 偏差修正的幂次：所有参数共享同一个 step 计数
            bc1 = 1.0 - beta1 ** self._step
            bc2 = 1.0 - beta2 ** self._step
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state.setdefault(id(p), {"m": torch.zeros_like(p), "v": torch.zeros_like(p)})
                m, v = st["m"], st["v"]
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                # 解耦权重衰减：先衰减权重，再走 Adam 的那一步
                p.mul_(1.0 - lr * wd)
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    """线性 warmup + 余弦退火，返回 **lr 的乘子**（∈[0,1]）。

        step < warmup:   m = (step+1) / warmup      ← +1：让第一步不为 0
        else:            m = 0.5(1 + cos(π · progress))

    为什么需要 warmup：Adam 的二阶动量 v 在最初几步估计极不可靠
    （只有 1~2 个样本的梯度平方），此时用全量 lr 容易一步走飞。
    Transformer 类模型基本都靠 warmup 才训得稳。
    """
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


class WarmupCosineLR:
    """把 :func:`cosine_with_warmup` 包成带 ``.step()`` 的调度器（对齐 torch 的调用方式）。

    注意：**必须在 optimizer.step() 之后调用**。torch 的 LambdaLR 也是这个语义，
    顺序反了会跳过第一个 lr 值（经典的「第一个 epoch 没 warmup」bug）。
    """

    def __init__(self, optimizer, total_steps: int, warmup_steps: int):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self._step = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        for g, base in zip(optimizer.param_groups, self.base_lrs):
            g.setdefault("initial_lr", base)

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for g in self.optimizer.param_groups]

    def step(self) -> None:
        self._step += 1
        mult = cosine_with_warmup(self._step - 1, self.total_steps, self.warmup_steps)
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * mult


def clip_grad_norm_(params, max_norm: float, eps: float = 1e-6) -> torch.Tensor:
    """全局范数梯度裁剪，语义与 ``torch.nn.utils.clip_grad_norm_`` 一致。

        total = √ Σ_p ‖g_p‖²                    ← **对所有参数一起**求范数，不是逐参数
        scale = max_norm / (total + ε)
        g_p  ← g_p · min(1, scale)

    为什么要「一起」求范数：逐参数裁剪会改变各参数之间的**相对**更新比例，
    等于偷偷改了优化问题的几何形状。全局范数裁剪保持方向不变，只缩长度。

    ε=1e-6 是为了 total=0（全部梯度为 0）时不出现除零。
    """
    total = 0.0
    grads = []
    for p in params:
        if p.grad is None:
            continue
        g = p.grad
        grads.append(g)
        total += float(g.detach().pow(2).sum())
    total = math.sqrt(total)
    if total == 0.0:
        return torch.tensor(0.0)
    clip_coef = max_norm / (total + eps)
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)
    return torch.tensor(total)
