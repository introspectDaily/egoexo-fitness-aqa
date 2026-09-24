"""torch Dataset / collate。

每个样本产出：
    feat        (T, D)   定长帧序列（已由 features.extract 预抽取好）
    score       float    标注者平均分（1~5）
    score_ord   (K-1,)   有序回归的累积目标: y_k = 1[score > k]
    keypoint    (N,)     关键点标签，1 = unsatisfies（论文口径的正类）
    kp_mask     (N,)     padding 掩码（该动作实际有几条关键点）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import FeatureBundle

SCORE_MIN, SCORE_MAX = 1, 5
N_SCORE_LEVELS = SCORE_MAX - SCORE_MIN + 1  # 5
VIEW_IDS = {v: i for i, v in enumerate(["ego_l", "ego_m", "ego_r", "exo_l", "exo_m", "exo_r"])}


@dataclass
class DatasetBundle:
    """训练集/验证集共用的打包信息（词表、维度）。"""

    action_to_id: dict[str, int]
    kp_vocab: dict[str, list[str]]  # action_name -> 关键点文本
    max_keypoints: int
    dim: int
    num_frames: int

    def save_meta(self) -> dict:
        return {
            "action_to_id": self.action_to_id,
            "kp_vocab": self.kp_vocab,
            "max_keypoints": self.max_keypoints,
            "dim": self.dim,
            "num_frames": self.num_frames,
        }


def build_bundle(samples, kp_vocab: dict[str, list[str]], dim: int, num_frames: int) -> DatasetBundle:
    action_to_id = {name: i for i, name in enumerate(sorted(kp_vocab))}
    return DatasetBundle(
        action_to_id=action_to_id,
        kp_vocab=kp_vocab,
        max_keypoints=max(len(v) for v in kp_vocab.values()),
        dim=dim,
        num_frames=num_frames,
    )


class AqaDataset(Dataset):
    def __init__(
        self,
        samples,
        features: FeatureBundle,
        bundle: DatasetBundle,
        indices: np.ndarray | None = None,
        l2norm: bool = True,
        augment: bool = False,
        temporal_jitter: int = 0,
        feat_dropout: float = 0.0,
        score_noise: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        self.samples = samples
        self.features = features
        self.bundle = bundle
        self.indices = np.arange(len(samples)) if indices is None else np.asarray(indices)
        self.l2norm = l2norm
        self.augment = augment
        self.temporal_jitter = temporal_jitter
        self.feat_dropout = feat_dropout
        # 分数是 >=1 位标注者的平均，本身带噪。训练时加一点高斯噪声相当于
        # 软标签 / 标签平滑，能防止回归器去拟合标注者之间的分歧。
        self.score_noise = score_noise
        self.rng = rng or np.random.default_rng(0)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[self.indices[i]]
        feat = np.asarray(self.features.get(s.sample_id), dtype=np.float32)

        if self.l2norm:
            norm = np.linalg.norm(feat, axis=-1, keepdims=True)
            feat = feat / np.maximum(norm, 1e-6)

        if self.augment:
            if self.temporal_jitter > 0 and len(feat) > self.temporal_jitter:
                j = int(self.rng.integers(-self.temporal_jitter, self.temporal_jitter + 1))
                if j > 0:
                    feat = np.concatenate([feat[j:], np.repeat(feat[-1:], j, axis=0)], axis=0)
                elif j < 0:
                    feat = np.concatenate([np.repeat(feat[:1], -j, axis=0), feat[:j]], axis=0)
            if self.feat_dropout > 0:
                keep = self.rng.random(feat.shape[:-1]) >= self.feat_dropout
                feat = feat * keep[..., None]

        score = float(s.score)
        if self.augment and self.score_noise > 0:
            score = float(np.clip(score + self.rng.normal(0, self.score_noise), SCORE_MIN, SCORE_MAX))

        # 有序回归目标: 对 k=1..K-1，y_k = 1 当且仅当 score > k
        ord_target = np.array([1.0 if score > (SCORE_MIN + k) else 0.0 for k in range(N_SCORE_LEVELS - 1)], dtype=np.float32)

        n_kp = len(s.keypoints)
        kp = np.zeros(self.bundle.max_keypoints, dtype=np.float32)
        mask = np.zeros(self.bundle.max_keypoints, dtype=np.float32)
        for j, k in enumerate(s.keypoints):
            kp[j] = float(k.label)
            mask[j] = 1.0

        return {
            "feat": torch.from_numpy(np.ascontiguousarray(feat)),
            "score": torch.tensor(score, dtype=torch.float32),
            "score_ord": torch.from_numpy(ord_target),
            "keypoint": torch.from_numpy(kp),
            "kp_mask": torch.from_numpy(mask),
            "action_id": torch.tensor(self.bundle.action_to_id[s.action_name], dtype=torch.long),
            "view_id": torch.tensor(VIEW_IDS[s.view], dtype=torch.long),
            "is_ego": torch.tensor(1 if s.view_type == "ego" else 0, dtype=torch.float32),
            "n_kp": torch.tensor(n_kp, dtype=torch.long),
            "index": torch.tensor(int(self.indices[i]), dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    out: dict[str, torch.Tensor] = {}
    for key in ("feat", "score", "score_ord", "keypoint", "kp_mask", "action_id", "view_id", "is_ego", "n_kp", "index"):
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out
