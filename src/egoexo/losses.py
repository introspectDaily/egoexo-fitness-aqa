"""损失函数。

    L = w_score · L_coral + w_kp · L_kp + w_act · L_action + λ · L_align

权重默认值来自论文与项目文档的交叉验证：
- λ=0.7 是论文 GEVFormer 用的跨视角对齐权重，且论文证明了它有效
  （ego F1 +0.0087 / exo −0.0022，净收益为正）。
- w_act=0.3：动作分类只当辅助正则，不让它主导 —— 12 类分类论文已到 0.93，
  对主任务的信息增益有限，但共享 backbone 的梯度有助于表征。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    score: float = 1.0
    keypoint: float = 1.0
    action: float = 0.3
    align: float = 0.7

    def as_dict(self) -> dict[str, float]:
        return {"score": self.score, "keypoint": self.keypoint, "action": self.action, "align": self.align}


def coral_loss(logits: torch.Tensor, ord_target: torch.Tensor) -> torch.Tensor:
    """有序回归损失：K-1 个累积二分类的 BCE。

    ord_target[b, k] = 1[score_b > k+1]，天然满足单调（一旦某档为 0，更高档必为 0），
    所以共享权重 + 独立偏置的结构能学出正确的序关系。
    """
    return F.binary_cross_entropy_with_logits(logits, ord_target)


def keypoint_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | float | None = None,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """加权的掩码 BCE。

    必须加权：正类（unsatisfies）只占 22%，不加权模型会退化成全预测 satisfied
    而拿到 0.78 的 accuracy、F1=0。pos_weight ≈ 3.54（实测）。

    focal_gamma > 0 时改用 Focal Loss，对"已经学好的多数类"降权，进一步缓解不平衡。
    """
    valid = mask > 0
    if not valid.any():
        return logits.new_zeros(())

    logit = logits[valid]
    target = labels[valid]

    if pos_weight is not None:
        if not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
        pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)

    bce = F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos_weight, reduction="none")

    if focal_gamma > 0:
        p = torch.sigmoid(logit)
        p_t = p * target + (1 - p) * (1 - target)
        bce = bce * (1 - p_t).pow(focal_gamma)

    return bce.mean()


def total_loss(
    out,
    batch: dict,
    weights: LossWeights,
    pos_weight: torch.Tensor | float | None = None,
    focal_gamma: float = 0.0,
    align_fn=None,
) -> tuple[torch.Tensor, dict[str, float]]:
    parts: dict[str, float] = {}

    l_score = coral_loss(out.score_logits, batch["score_ord"])
    parts["score"] = float(l_score.detach())

    l_kp = keypoint_loss(out.keypoint_logits, batch["keypoint"], batch["kp_mask"], pos_weight, focal_gamma)
    parts["keypoint"] = float(l_kp.detach())

    total = weights.score * l_score + weights.keypoint * l_kp

    if weights.action > 0 and out.action_logits.numel() > 0:
        l_act = F.cross_entropy(out.action_logits, batch["action_id"])
        parts["action"] = float(l_act.detach())
        total = total + weights.action * l_act

    if weights.align > 0 and align_fn is not None:
        l_align = align_fn(out.embedding, batch)
        parts["align"] = float(l_align.detach())
        total = total + weights.align * l_align

    parts["total"] = float(total.detach())
    return total, parts
