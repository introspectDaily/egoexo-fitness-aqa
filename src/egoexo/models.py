"""模型：时序编码器 + 三个头（有序分数 / 文本条件关键点 / 动作分类）+ 跨视角对齐。

为什么是这个结构（而不是直接上 VideoMAE）：

1. 数据集 release 的是**逐帧 CLIP ViT-B/32 特征**，不是原始视频。所以最省、最稳的
   路线是在冻结特征上训一个轻量时序模型 —— 论文自己的 GEVFormer 也是这么做的
   （CLIP 视觉/文本编码器全程冻结，只训 TCM + CMV）。
2. 可训样本只有 913 个单动作 × 视角 = 5371 条，且同 record 内高度相关。
   在这个量级上，**参数量越小越不容易过拟合**。本模型可训参数约 2~4M。
3. 关键点是「文本条件」任务：同一段视频要针对不同的关键点句子给出不同结论。
   所以用 cross-attention 让文本 query 去视觉 token 里找证据，而不是把所有关键点
   当成无差别的多标签输出。这样 12 个动作可以共享一套头。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- 基础块


class SinusoidalPosEmb(nn.Module):
    """时间位置编码。参数为零，不会加剧过拟合。"""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype)


class AttentionPool(nn.Module):
    """可学习的注意力池化，比 mean pooling 更能挑出「关键那一帧」。"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, D)
        q = self.query.expand(x.size(0), -1, -1)
        attn = (q @ x.transpose(1, 2)) / math.sqrt(x.size(-1))  # (B, 1, T)
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, :], float("-inf"))
        attn = self.drop(attn.softmax(dim=-1))
        return self.norm((attn @ x).squeeze(1))


class TemporalEncoder(nn.Module):
    """冻结 CLIP 特征 -> 时序 token。

    输入维度固定 512（CLIP ViT-B/32）。先降维再送 Transformer：
    既省算力，也起到瓶颈正则的作用。
    """

    def __init__(
        self,
        in_dim: int = 512,
        dim: int = 256,
        depth: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos = SinusoidalPosEmb(dim, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, feat: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.proj(feat)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=(~mask if mask is not None else None))
        return self.norm(x)


# ---------------------------------------------------------------- 头


class CoralHead(nn.Module):
    """有序回归（CORAL）。

    1~5 分是有序标签，MSE 隐含「1→2 的差距 == 4→5 的差距」，这不成立。
    CORAL 把它拆成 K-1 个**共享权重、各自偏置**的累积二分类：
        P(y > k) = sigmoid(w·z + b_k)
    预测值 = 1 + Σ_k P(y > k)，且天然保证单调（b_k 有序性由训练自行学出）。
    对照项目文档：主观评分类任务该用 ordinal，而不是纯回归。
    """

    def __init__(self, dim: int, levels: int = 5, dropout: float = 0.1):
        super().__init__()
        self.levels = levels
        self.drop = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.randn(dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(levels - 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z (B,D) @ weight (D,1) -> (B,1)，再广播加 bias (K-1,) -> (B,K-1)
        # 共享权重 + 独立偏置：这是 CORAL 的定义，保证 K-1 个累积 logit 只差一个偏置
        return self.drop(z) @ self.weight.unsqueeze(-1) + self.bias

    @staticmethod
    def predict(logits: torch.Tensor, score_min: int = 1) -> torch.Tensor:
        return score_min + torch.sigmoid(logits).sum(dim=-1)


class KeypointHead(nn.Module):
    """文本条件的关键点头（GEV 的核心）。

    query = 该动作的技术关键点文本嵌入（CLIP 文本编码器，冻结）
    key/value = 视觉时序 token
    -> 每条关键点得到一个 logit

    这样 12 个动作、102 条关键点句子共享同一套参数，且对没见过的措辞有一定泛化。
    """

    def __init__(self, dim: int, text_dim: int = 512, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, dim))
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(
        self,
        visual: torch.Tensor,
        text_emb: torch.Tensor,
        kp_mask: torch.Tensor,
        visual_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """visual (B,T,D), text_emb (B,N,text_dim), kp_mask (B,N) -> logits (B,N)"""
        q = self.text_proj(text_emb)  # (B, N, D)
        ctx, _ = self.attn(
            q,
            visual,
            visual,
            key_padding_mask=(~visual_mask if visual_mask is not None else None),
            need_weights=False,
        )
        h = self.norm(q + self.drop(ctx))
        logits = self.out(h).squeeze(-1)  # (B, N)
        return logits.masked_fill(kp_mask <= 0, 0.0)


# ---------------------------------------------------------------- 主模型


@dataclass
class ModelOutput:
    score: torch.Tensor  # (B,) 连续分
    score_logits: torch.Tensor  # (B, K-1) 累积 logits
    keypoint_logits: torch.Tensor  # (B, N)
    action_logits: torch.Tensor  # (B, C)
    embedding: torch.Tensor  # (B, D)  用于跨视角对齐


class AqaModel(nn.Module):
    def __init__(
        self,
        num_actions: int,
        kp_text_emb: torch.Tensor,  # (num_actions, max_kp, text_dim)
        in_dim: int = 512,
        dim: int = 256,
        depth: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        num_frames: int = 32,
        score_levels: int = 5,
        use_action_head: bool = True,
        use_view_embed: bool = True,
    ):
        super().__init__()
        self.encoder = TemporalEncoder(in_dim, dim, depth, heads, dropout, max_len=max(64, num_frames * 2))
        self.pool = AttentionPool(dim, dropout)
        self.use_view_embed = use_view_embed
        if use_view_embed:
            self.view_embed = nn.Embedding(6, dim)
            nn.init.normal_(self.view_embed.weight, std=0.02)

        self.score_head = CoralHead(dim, score_levels, dropout)
        self.keypoint_head = KeypointHead(dim, kp_text_emb.size(-1), heads, dropout)
        self.use_action_head = use_action_head
        self.action_head = nn.Linear(dim, num_actions) if use_action_head else None

        # 关键点文本嵌入作为 buffer：不训练、随模型一起搬设备、随 checkpoint 一起存
        self.register_buffer("kp_text_emb", kp_text_emb, persistent=True)
        self.dim = dim
        self.score_levels = score_levels

    def encode(self, feat: torch.Tensor, view_id: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.encoder(feat, mask)
        z = self.pool(tokens, mask)
        if self.use_view_embed and view_id is not None:
            z = z + self.view_embed(view_id)
        return z

    def forward(
        self,
        feat: torch.Tensor,
        action_id: torch.Tensor,
        view_id: torch.Tensor | None = None,
        kp_mask: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> ModelOutput:
        tokens = self.encoder(feat, frame_mask)
        z = self.pool(tokens, frame_mask)
        if self.use_view_embed and view_id is not None:
            z = z + self.view_embed(view_id)

        text_emb = self.kp_text_emb[action_id]  # (B, N, text_dim)
        if kp_mask is None:
            kp_mask = torch.ones(text_emb.size(0), text_emb.size(1), device=feat.device)

        score_logits = self.score_head(z)
        kp_logits = self.keypoint_head(tokens, text_emb, kp_mask, frame_mask)
        action_logits = self.action_head(z) if self.action_head is not None else z.new_zeros((z.size(0), 0))

        return ModelOutput(
            score=CoralHead.predict(score_logits),
            score_logits=score_logits,
            keypoint_logits=kp_logits,
            action_logits=action_logits,
            embedding=z,
        )


# ---------------------------------------------------------------- 跨视角对齐


def infonce_alignment(
    z: torch.Tensor,
    action_key: torch.Tensor,
    view_type: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE：让**同一动作的不同视角**在表征空间里互相靠近。

    论文的关键发现：简单把 ego+exo 混在一起训**不会**稳定涨点，甚至掉点；
    真正有效的是这个显式对齐损失（论文 λ=0.7）。所以这个不是可选装饰。

    action_key 用来标识"同一个物理动作"（同一 record 的同一 action_idx），
    view_type 用来排除同视角之间的配对（同视角本来就一样，对齐它没有信息量）。
    """
    if z.size(0) < 2:
        return z.new_zeros(())

    z = F.normalize(z, dim=-1)
    logits = z @ z.t() / temperature  # (B, B)

    same_action = action_key[:, None] == action_key[None, :]
    cross_view = view_type[:, None] != view_type[None, :]
    pos = same_action & cross_view

    # 排除自身
    eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    pos = pos & ~eye

    valid = pos.any(dim=1)
    if not valid.any():
        return z.new_zeros(())

    logits = logits.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    # ⚠️ 必须用 masked_fill 而不是 `log_prob * pos`：对角线位置的 log_prob 是 -inf，
    #    而 `-inf * 0` 是 **NaN**（不是 0），会把整个 loss 污染成 NaN。
    log_prob = log_prob.masked_fill(~pos, 0.0)

    pos_count = pos.sum(dim=-1).clamp(min=1)
    loss = -log_prob.sum(dim=-1)[valid] / pos_count[valid]
    return loss.mean()
