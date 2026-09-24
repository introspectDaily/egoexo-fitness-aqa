"""评估指标。

口径对齐论文：
- 分数：SROCC 为主（AQA 领域标准，跨样本尺度免疫），同时报 PLCC / MAE / RMSE
- 关键点：**F1，且正类 = "unsatisfies"**（少数类）。论文 Table 7 就是这么算的，
  因为 "satisfies" 占 78%，报 accuracy 会让"全预测 satisfied"拿到 0.78 的假高分。
- 全部指标按 ego / exo / 总体 三列分别报（跨视角是这个数据集的核心变量）。
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(stats.pearsonr(a, b).statistic)


def score_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if len(pred) == 0:
        return {"srocc": float("nan"), "plcc": float("nan"), "mae": float("nan"), "rmse": float("nan"), "acc1": float("nan"), "n": 0}
    err = pred - target
    return {
        "srocc": _safe_spearman(pred, target),
        "plcc": _safe_pearson(pred, target),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        # ±1 分内算对：有序标签上比 MAE 直观得多
        "acc1": float((np.abs(err) <= 1.0).mean()),
        "n": int(len(pred)),
    }


def keypoint_metrics(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """正类 = unsatisfies (label==1)。

    阈值需要在验证集上扫（见 :func:`best_threshold`），因为正类只占 22%，
    默认 0.5 通常不是最优 —— 但论文没调阈值，所以两边都报。
    """
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    y = np.asarray(labels)
    m = np.asarray(mask).astype(bool)
    p, y = p[m], y[m]
    if len(p) == 0:
        return {"f1": float("nan"), "precision": float("nan"), "recall": float("nan"), "accuracy": float("nan"), "n": 0}

    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "accuracy": float((pred == y).mean()),
        "pos_rate": float(y.mean()),
        "n": int(len(p)),
    }


def best_threshold(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray, grid: int = 41) -> tuple[float, float]:
    """在验证集上扫 F1 最优阈值。返回 (threshold, f1)。"""
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    y = np.asarray(labels)
    m = np.asarray(mask).astype(bool)
    p, y = p[m], y[m]
    if len(p) == 0:
        return 0.5, float("nan")
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, grid):
        f1 = keypoint_metrics(p, y, np.ones_like(y, dtype=bool), threshold=float(t))["f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, float(best_f1)


def per_action_breakdown(samples, indices, pred, target) -> dict[str, dict[str, float]]:
    """按动作类型拆开看。12 类里哪几类学不会，一眼能看出来。"""
    out: dict[str, dict[str, float]] = {}
    names = [samples[i].action_name for i in indices]
    for name in sorted(set(names)):
        sel = np.array([n == name for n in names])
        if sel.sum() < 3:
            out[name] = {"srocc": float("nan"), "mae": float("nan"), "n": int(sel.sum())}
            continue
        m = score_metrics(np.asarray(pred)[sel], np.asarray(target)[sel])
        out[name] = {"srocc": m["srocc"], "mae": m["mae"], "n": m["n"]}
    return out


# ---------------------------------------------------------------- 标注者一致性


def krippendorff_alpha_ordinal(units: list[list[int]], max_value: int = 5) -> float:
    """有序尺度的 Krippendorff's alpha（分数标注的标注者间一致性）。

    用差异函数 d(c,k) = (Σ_{g=c..k} n_g - (n_c+n_k)/2)^2，即 ordinal metric。
    units = [[标注者1的分, 标注者2的分, ...], ...]，只统计 >=2 位标注者的动作。
    """
    return _krippendorff(units, max_value, ordinal=True)


def krippendorff_alpha_nominal(units: list[list[int]], max_value: int = 1) -> float:
    """名义尺度的 alpha（关键点 True/False 的一致性）。"""
    return _krippendorff(units, max_value, ordinal=False)


def _krippendorff(units, max_value: int, ordinal: bool) -> float:
    """Krippendorff's alpha 的实际实现。

    ⚠️ units 里的值会被强制转成 int。原因：如果传进来的是 bool，
    `coincidence[a, b]` 会变成 **布尔掩码索引**（numpy 会插一个新轴返回副本），
    而不是取 (1,0) 元素 —— 不报错、静默算错，最后表现为 alpha=nan。
    这个坑真的踩过。
    """
    pairs = [[int(v) for v in u] for u in units if len(u) >= 2]
    if len(pairs) < 2:
        return float("nan")

    values = list(range(1, max_value + 1))
    coincidence = np.zeros((max_value + 1, max_value + 1), dtype=np.float64)
    for u in pairs:
        m = len(u)
        for a in u:
            for b in u:
                if a == b:
                    continue
                coincidence[a, b] += 1.0 / (m - 1)

    n_c = coincidence.sum(axis=1)
    n = coincidence.sum()
    if n == 0:
        return float("nan")

    if ordinal:

        def delta(c: int, k: int) -> float:
            lo, hi = min(c, k), max(c, k)
            return float((sum(n_c[lo : hi + 1]) - (n_c[c] + n_c[k]) / 2.0) ** 2)
    else:

        def delta(c: int, k: int) -> float:
            return 0.0 if c == k else 1.0

    do = sum(coincidence[c, k] * delta(c, k) for c in values for k in values) / n
    de = sum(n_c[c] * n_c[k] * delta(c, k) for c in values for k in values) / (n * (n - 1))
    if de <= 0:
        # 边际分布退化到单一取值，期望分歧为 0，alpha 无定义
        return float("nan")
    return float(1 - do / de)


# ---------------------------------------------------------------- 标注分歧诊断


def agreement_diagnostics(score_units: list[list[int]]) -> dict[str, float]:
    """比 alpha 更好读的一致性诊断。

    alpha 低到 0.17 时，光看一个数字不容易判断是「真低」还是「算法错」。
    同时报这几个直接可核验的量：
      - exact    : 两两完全相等的比例
      - within1  : 两两差距 <= 1 的比例
      - mean_abs : 两两平均绝对差
      - spearman : 只取恰好 2 位标注者的动作，把两人分数当成两列向量做秩相关
                   （这是独立于 alpha 的交叉验证：如果 alpha≈0.17 而 spearman≈0.5，
                    说明 alpha 算错了；两者都低才是标注本身真的不一致）
    """
    import itertools

    diffs: list[int] = []
    for u in score_units:
        for a, b in itertools.combinations(u, 2):
            diffs.append(abs(int(a) - int(b)))

    pairs2 = [(int(u[0]), int(u[1])) for u in score_units if len(u) == 2]
    spearman = float("nan")
    if len(pairs2) >= 10:
        x = np.array([p[0] for p in pairs2], dtype=np.float64)
        y = np.array([p[1] for p in pairs2], dtype=np.float64)
        spearman = _safe_spearman(x, y)

    if not diffs:
        return {"n_pairs": 0, "exact": float("nan"), "within1": float("nan"), "mean_abs": float("nan"), "spearman": spearman}

    d = np.array(diffs, dtype=np.float64)
    return {
        "n_pairs": int(len(d)),
        "exact": float((d == 0).mean()),
        "within1": float((d <= 1).mean()),
        "mean_abs": float(d.mean()),
        "spearman": spearman,
    }
