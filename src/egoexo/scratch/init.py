"""手撸参数初始化：和 ``torch.nn.init`` 的默认行为对齐。

为什么必须对齐：初始化的量级决定了训练能不能起来（太小=梯度消失，太大=激活饱和），
而框架的默认初始化**不是显然的**。把公式写出来才能知道 ``nn.Linear`` 默认在做什么。

    nn.Linear.reset_parameters()
        kaiming_uniform_(weight, a=√5)   →  gain = √(2/(1+a²)) = 1/√3
                                          bound = √3 · gain/√fan_in = 1/√fan_in
                                        →  weight ~ U(-1/√fan_in, 1/√fan_in)
        bias ~ U(-1/√fan_in, 1/√fan_in)     （和 weight 同一个 bound，不是 0！）

    nn.MultiheadAttention._reset_parameters()
        xavier_uniform_(in_proj_weight)  →  bound = √(6/(fan_in+fan_out))
        in_proj_bias = 0,  out_proj.bias = 0

    nn.LayerNorm  → weight = 1, bias = 0（identity 起步）
    nn.Embedding  → weight ~ N(0, 1)
"""

from __future__ import annotations

import math

import torch

__all__ = ["kaiming_uniform_", "xavier_uniform_", "constant_", "normal_", "reset_linear", "fan_in_fan_out"]


def fan_in_fan_out(t: torch.Tensor) -> tuple[int, int]:
    """按 torch 的约定算 fan_in / fan_out。

    torch 对 fan 的定义（``_calculate_fan_in_and_fan_out``）：
        ndim == 2:  fan_in = size(1), fan_out = size(0)
        ndim > 2:   fan_in = size(1) * receptive_field, fan_out = size(0) * receptive_field
    注意 2D 时是 **fan_in = 第 1 维**（因为权重是 (out, in) 存的）。
    """
    if t.dim() < 2:
        raise ValueError(f"fan 要求至少 2 维张量，实际 {t.shape}")
    num_input = t.size(1)
    num_output = t.size(0)
    receptive = 1
    for s in t.shape[2:]:
        receptive *= s
    return num_input * receptive, num_output * receptive


def kaiming_uniform_(t: torch.Tensor, a: float = math.sqrt(5)) -> torch.Tensor:
    """Kaiming 均匀初始化，原地。``a`` 是负斜率（ReLU 用 √5 时 gain 退化成 1/√3）。"""
    fan_in, _ = fan_in_fan_out(t)
    gain = math.sqrt(2.0 / (1.0 + a * a))
    std = gain / math.sqrt(fan_in) if fan_in > 0 else 0.0
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        t.uniform_(-bound, bound)
    return t


def xavier_uniform_(t: torch.Tensor) -> torch.Tensor:
    """Xavier/Glorot 均匀初始化：让前向和反向的方差都守恒。"""
    fan_in, fan_out = fan_in_fan_out(t)
    gain = 1.0
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        t.uniform_(-bound, bound)
    return t


def constant_(t: torch.Tensor, val: float) -> torch.Tensor:
    with torch.no_grad():
        t.fill_(val)
    return t


def normal_(t: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    with torch.no_grad():
        t.normal_(mean, std)
    return t


def reset_linear(weight: torch.Tensor, bias: torch.Tensor | None) -> None:
    """复刻 ``nn.Linear.reset_parameters``。"""
    kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = fan_in_fan_out(weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        with torch.no_grad():
            bias.uniform_(-bound, bound)
