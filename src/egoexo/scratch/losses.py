"""手撸损失函数（与 ``egoexo/losses.py`` 一一对应）。

四个损失各自回答项目里的一个问题：

    keypoint_loss    加权 BCE —— 为什么必须加权（22% 正类，不加权退化成全预测多数类）
    coral_loss       有序回归 —— 为什么 1~5 分不能用 MSE（序信息）
    focal            难例挖掘 —— 加权 BCE 的进阶版
    infonce_alignment 跨视角对齐 —— 为什么必须显式对齐（只混训不涨点）

对拍：``cli parity`` 会把这里的实现和 ``egoexo/losses.py`` + ``F.*`` 逐个比数值。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import functional as Fs

__all__ = ["coral_loss", "keypoint_loss", "cross_entropy_loss", "infonce_alignment", "resolve_pos_weight"]


def coral_loss(logits: torch.Tensor, ord_target: torch.Tensor, impl=None) -> torch.Tensor:
    """CORAL：K-1 个累积二分类的 BCE（取 mean）。

    logits     (B, K-1)   累积 logit，第 k 列表示 P(score > k+1)
    ord_target (B, K-1)   标签 1[score > k+1]

    共享权重 + 独立偏置的结构在模型那头（``CoralHead``），这里只是 BCE。
    有意思的地方在**标签构造**：``1[score>1], 1[score>2], 1[score>3], 1[score>4]``
    必然是 ``1,1,0,0`` 这种「前面全 1、后面全 0」的形状。这个单调性就是「序信息」，
    它不会被 MSE 利用，也不会被普通多分类利用。
    """
    if impl is not None and impl.on("loss"):
        from .autograd_ops import bce_with_logits_fn

        if impl.manual_bwd:
            return bce_with_logits_fn(logits, ord_target, None).mean()
        return Fs.binary_cross_entropy_with_logits(logits, ord_target, None).mean()
    return F.binary_cross_entropy_with_logits(logits, ord_target)


def keypoint_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | float | None = None,
    focal_gamma: float = 0.0,
    impl=None,
) -> torch.Tensor:
    """掩码 + 加权的 BCE，可选 Focal。

    Step 1 掩码：只有该动作真实存在的关键点位置参与（padding 不算 loss）。
    Step 2 加权：``pos_weight ≈ 3.54``（负/正 = 78/22）。不加权会怎样 ——
                 全预测「达标」就能拿 0.78 accuracy 但 **F1=0**，而 F1 是主指标。
    Step 3 Focal：``(1-p_t)^γ`` 给「已经学好的样本」降权，等价于自动做难例挖掘。
                 注意它和 pos_weight 是**两种不同的**不平衡处理，可叠加。

    ⚠️ 一个必须注意的语义：本项目的正类是 **unsatisfies（不达标）**，
    不是「达标」。正类搞反的话 F1 会算得完全不同（而且看起来还挺高）。
    """
    valid = mask > 0
    if not valid.any():
        return logits.new_zeros(())

    logit = logits[valid]
    target = labels[valid]

    if pos_weight is not None and not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)

    if impl is not None and impl.on("loss"):
        if impl.manual_bwd:
            from .autograd_ops import bce_with_logits_fn

            bce = bce_with_logits_fn(logit, target, pos_weight)
        else:
            bce = Fs.binary_cross_entropy_with_logits(logit, target, pos_weight)
    else:
        bce = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight, reduction="none")

    if focal_gamma > 0:
        p = torch.sigmoid(logit)
        # p_t = 「预测成正确类别的概率」，无论正负样本都落在 [0,1]
        p_t = p * target + (1 - p) * (1 - target)
        bce = bce * (1 - p_t).pow(focal_gamma)
    return bce.mean()


def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor, impl=None) -> torch.Tensor:
    if impl is not None and impl.on("loss"):
        if impl.manual_bwd:
            from .autograd_ops import cross_entropy_fn

            return cross_entropy_fn(logits, target)
        return Fs.cross_entropy(logits, target)
    return F.cross_entropy(logits, target)


def infonce_alignment(
    z: torch.Tensor,
    action_key: torch.Tensor,
    view_type: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE：同一物理动作的**不同视角**互为正样本。

        L = -1/|P(i)| · Σ_{j∈P(i)} log  exp(s_ij/τ) / Σ_{k≠i} exp(s_ik/τ)

    这个损失的作用是「把跨视角的表示拉到一起」，用来对抗论文里那个
    「exo 训 / ego 测，Top-1 从 0.93 掉到 0.09」的崩塌。

    ⚠️ 有个**静默 NaN 陷阱**，这是本文件最值得记住的一行：
        `log_prob` 在对角线位置是 -inf（自己和自己被 mask 成 -inf 之后
        logsumexp 减出来的）。如果用 `log_prob * pos`，那么 `-inf * 0 = NaN`，
        整个 loss 直接 NaN，而且**不报错**，只是训练慢慢崩。
        必须用 ``masked_fill`` 而不是乘法。
    """
    if z.size(0) < 2:
        return z.new_zeros(())

    # 先 L2 归一化，于是内积 == 余弦相似度，落在 [-1,1]，logits 量级可控
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    logits = z @ z.t() / temperature

    same_action = action_key[:, None] == action_key[None, :]
    cross_view = view_type[:, None] != view_type[None, :]
    pos = same_action & cross_view

    eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    pos = pos & ~eye

    valid = pos.any(dim=1)
    if not valid.any():
        return z.new_zeros(())

    logits = logits.masked_fill(eye, float("-inf"))
    log_prob = logits - Fs.logsumexp(logits, dim=-1, keepdim=True)
    log_prob = log_prob.masked_fill(~pos, 0.0)  # ← 不能用 ``log_prob * pos``（-inf*0=NaN）

    pos_count = pos.sum(dim=-1).clamp(min=1)
    loss = -log_prob.sum(dim=-1)[valid] / pos_count[valid]
    return loss.mean()


def resolve_pos_weight(pos: float, neg: float, clip: float = 20.0) -> float:
    return float(min(neg / max(pos, 1e-12), clip))
