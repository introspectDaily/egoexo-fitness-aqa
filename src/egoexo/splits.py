"""数据划分。

核心约束（这个数据集上最容易踩的坑）：
    一个 record 内的多个单动作、以及同一个单动作的 6 路视角，**必须同进同出**。
    因为它们共享被试、机位、衣着、光照、体型 —— 打散了就是指标虚高。

实测佐证：官方 subaction 标注里的 train/test subset 是**按 (record, view, sequence) 给的**，
68 个 record **全部**是 train/test 混排（0 个纯 train、0 个纯 test）。
→ 直接套用官方 subset 训 AQA 会泄漏。所以本模块用 GroupKFold(by record_id)，
   这是唯一无泄漏的选择，而且因为 76 个 record 各有唯一 original_actor，
   「按 record 划分」等价于「按人划分」。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class Fold:
    train_idx: np.ndarray
    val_idx: np.ndarray
    fold: int
    train_records: list[str]
    val_records: list[str]


def group_kfold(samples, n_splits: int = 5, seed: int = 0) -> list[Fold]:
    """按 record_id 分组的 K 折。

    record 数量少（76），所以折数不宜大：5 折每折约 15 个 record 验证。
    划分目标不是“严格分层”，而是“各折样本量接近 + record 不跨折”——
    后者才是防泄漏的关键，前者只影响指标稳定性。
    """
    rng = random.Random(seed)

    # record -> 该 record 的样本下标
    by_record: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        by_record.setdefault(s.record_id, []).append(i)

    records = sorted(by_record)
    rng.shuffle(records)

    # 贪心分配：优先把大 record 分给当前样本数最少的折，让各折样本量接近
    fold_records: list[list[str]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for rid in sorted(records, key=lambda r: -len(by_record[r])):
        target = int(np.argmin(fold_sizes))
        fold_records[target].append(rid)
        fold_sizes[target] += len(by_record[rid])

    all_idx = set(range(len(samples)))
    folds = []
    for k in range(n_splits):
        val_records = sorted(fold_records[k])
        val_idx = np.array(sorted(i for r in val_records for i in by_record[r]), dtype=np.int64)
        train_idx = np.array(sorted(all_idx - set(val_idx.tolist())), dtype=np.int64)
        folds.append(
            Fold(
                train_idx=train_idx,
                val_idx=val_idx,
                fold=k,
                train_records=sorted(set(records) - set(val_records)),
                val_records=val_records,
            )
        )
    return folds


def holdout(samples, val_frac: float = 0.2, seed: int = 0) -> Fold:
    """单次划分。快速迭代用；正式结果请用 group_kfold。"""
    folds = group_kfold(samples, n_splits=max(2, round(1 / val_frac)), seed=seed)
    return folds[0]


def assert_no_leakage(samples, split: Fold, name: str = "split") -> None:
    """断言 train/val 之间没有 record 重叠。任何一次训练前都应该调用。"""
    tr = {samples[i].record_id for i in split.train_idx}
    va = {samples[i].record_id for i in split.val_idx}
    overlap = tr & va
    if overlap:
        raise AssertionError(f"[{name}] record 泄漏! {len(overlap)} 个 record 同时出现在 train/val: {sorted(overlap)[:10]}")
    assert len(tr) + len(va) == len({s.record_id for s in samples}), f"[{name}] 有 record 掉出了划分"
