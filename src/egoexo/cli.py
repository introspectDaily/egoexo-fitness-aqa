"""命令行入口。

    python -m egoexo.cli stats   --raw-dir data/raw_annotations
    python -m egoexo.cli probe   --feat-root data/features_open
    python -m egoexo.cli extract --raw-dir ... --feat-root ... --out data/precomputed
    python -m egoexo.cli smoke                      # 不需要任何数据，先验证代码
    python -m egoexo.cli train   --precomputed data/precomputed --folds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- stats


def cmd_stats(args) -> None:
    import collections
    import statistics

    from .annotations import action_subsets, build_keypoint_vocab, load_dataset
    from .metrics import agreement_diagnostics, krippendorff_alpha_nominal, krippendorff_alpha_ordinal

    ds = load_dataset(args.raw_dir)
    vocab = build_keypoint_vocab(ds)

    print("=" * 72)
    print("EgoExo-Fitness 数据概览")
    print("=" * 72)
    print(f"record 数            : {len(ds.records)}")
    print(f"唯一 original_actor  : {len({r.actor for r in ds.records.values()})}  (1 record = 1 人 → 按人划分 == 按 record 划分)")
    print(f"有 IAJ 标注的单动作   : {len(ds.actions)}")
    print(f"样本数 (单动作 × 视角): {len(ds.samples)}")
    print(f"动作类别              : {len(ds.action_names)}")
    print(f"唯一关键点句子        : {sum(len(v) for v in vocab.values())}")

    print("\n-- 视角分布 --")
    for v, c in sorted(collections.Counter(s.view for s in ds.samples).items()):
        print(f"  {v}: {c}")
    print("  view_type:", dict(collections.Counter(s.view_type for s in ds.samples)))

    print("\n-- 标注者人数分布（每个单动作） --")
    for k, c in sorted(collections.Counter(s.n_annotators for s in ds.samples).items()):
        print(f"  {k} 位: {c}")

    scores = [s.score for s in ds.samples]
    print(f"\n-- 分数分布 -- (mean={statistics.mean(scores):.3f}, std={statistics.pstdev(scores):.3f})")
    for k, c in sorted(collections.Counter(round(s, 1) for s in scores).items()):
        bar = "█" * int(c / max(1, len(scores)) * 200)
        print(f"  {k:>4}: {c:>5}  {bar}")

    pos = np.mean([k.label for s in ds.samples for k in s.keypoints])
    print(f"\n-- 关键点 --")
    print(f"  unsatisfies 占比: {pos:.4f}  → pos_weight≈{(1-pos)/pos:.3f}")
    for name, kps in sorted(vocab.items()):
        print(f"  {name}: {len(kps)} 条")

    # 标注者间一致性：这份 release 里几乎没人做，做了就是加分项
    print("\n-- 标注者间一致性 --")
    score_units = [a.scores for a in ds.actions.values()]
    alpha_s = krippendorff_alpha_ordinal(score_units, max_value=5)
    print(f"  分数(有序) Krippendorff's α = {alpha_s:.4f}   [n={sum(1 for u in score_units if len(u) >= 2)} 个多标注动作]")

    # 逐关键点位置收集各标注者的 0/1。必须用**未聚合**的原始值：
    # 用多数表决后的 keypoints 会让 α 恒等于 1，等于什么都没测。
    kp_units: list[list[int]] = []
    for a in ds.actions.values():
        kp_units.extend(a.keypoints_per_annotator)
    alpha_k = krippendorff_alpha_nominal(kp_units, max_value=1)
    print(f"  关键点(名义) Krippendorff's α = {alpha_k:.4f}   [n={len(kp_units)} 个关键点位置]")

    diag = agreement_diagnostics(score_units)
    from .annotations import annotator_bias_table

    bias = annotator_bias_table(ds.actions)
    used = {k: v for k, v in bias.items() if v["used"]}
    print(f"\n  标注者个人均值（共 {len(bias)} 人，{len(used)} 人标注量 >=30）:")
    if used:
        lo = min(v["mean"] for v in used.values())
        hi = max(v["mean"] for v in used.values())
        print(f"    区间 {lo:.3f} ~ {hi:.3f}  (跨度 {hi-lo:.3f} 分 / 满分 5 分)")
        for k, v in sorted(used.items(), key=lambda kv: -kv[1]["mean"]):
            print(f"      {k:<14} n={v['n']:>4}  mean={v['mean']:.3f}  std={v['std']:.3f}")
    print(f"\n  分数分歧诊断 [n={diag['n_pairs']} 对]:")
    print(f"    完全相等      : {diag['exact']:.4f}")
    print(f"    差距 ≤1 分    : {diag['within1']:.4f}")
    print(f"    平均绝对差    : {diag['mean_abs']:.4f}")
    print(f"    两位标注者间 Spearman: {diag['spearman']:.4f}   ← 独立于 α 的交叉校验")

    if not np.isnan(alpha_s) and alpha_s < 0.6:
        print("\n  ⚠️  α < 0.6：分数标注本身分歧很大。直接后果：")
        print("      (a) SROCC 的上限被标注噪声压住，别指望 0.9；")
        print("      (b) 训练必须当软标签处理（本仓库默认 --score-noise 0.15）；")
        print("      (c) 报结果时应同时报这个 α，否则数字无法被正确解读。")

    # 官方划分是否可用
    subs = action_subsets(args.raw_dir)
    pure = sum(1 for v in subs.values() if len(v) == 1)
    print(f"\n-- 官方 train/test subset --")
    print(f"  覆盖 record: {len(subs)} | 纯 train/test: {pure} | 混合: {len(subs) - pure}")
    if pure == 0 and subs:
        print("  ⚠️  全部 record 内部 train/test 混排 → 官方 subset 不能直接用于 AQA（会泄漏）")
        print("      本仓库用 GroupKFold(by record_id) 作为无泄漏协议。")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "num_records": len(ds.records),
                    "num_actions": len(ds.actions),
                    "num_samples": len(ds.samples),
                    "action_names": ds.action_names,
                    "keypoint_vocab": vocab,
                    "score_mean": statistics.mean(scores),
                    "score_std": statistics.pstdev(scores),
                    "unsatisfies_rate": float(pos),
                    "krippendorff_alpha_score": alpha_s,
                    "krippendorff_alpha_keypoint": alpha_k,
                    "score_agreement_diagnostics": diag,
                    "annotator_bias": {k: v for k, v in ds.debias_info.get("table", {}).items()},
                },
                indent=1,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\n已写出 {args.out}")


# ---------------------------------------------------------------- probe / extract


def cmd_probe(args) -> None:
    from .features import probe

    probe(args.feat_root, max_files=args.max_files)


def cmd_extract(args) -> None:
    from .annotations import build_keypoint_vocab, load_dataset
    from .features import extract

    ds = load_dataset(args.raw_dir)
    vocab = build_keypoint_vocab(ds)
    if args.only_views:
        keep = set(args.only_views.split(","))
        samples = [s for s in ds.samples if s.view in keep]
        print(f"[extract] 只保留视角 {sorted(keep)}: {len(samples)}/{len(ds.samples)} 个样本")
    else:
        samples = ds.samples

    extract(
        samples,
        args.feat_root,
        args.out,
        num_frames=args.num_frames,
        crop_ratio=args.crop_ratio,
        dtype=args.dtype,
    )
    (Path(args.out) / "kp_vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- smoke


def cmd_smoke(args) -> None:
    """用合成数据跑通「建表 -> 前向 -> 反向 -> 评估」全链路。

    在花时间下 6.4GB 特征之前先跑这个：如果这里过了，说明代码没问题，
    后面出问题就一定是数据/环境问题，能省掉大量瞎猜。
    """
    import torch

    from .annotations import Action, Dataset, Keypoint, Record, Sample
    from .data import build_bundle, collate
    from .engine import TrainConfig, evaluate, train_fold
    from .features import FeatureBundle
    from .splits import assert_no_leakage, group_kfold

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    T, D, N_ACTIONS, N_REC = args.frames, 512, 4, 12
    names = [f"Action{i}" for i in range(N_ACTIONS)]
    kp_vocab = {n: [f"keypoint {n} #{j}" for j in range(7 + i % 3)] for i, n in enumerate(names)}

    samples, actions, records = [], {}, {}
    for r in range(N_REC):
        rid = f"rec{r:02d}"
        views = ("ego_l", "ego_m", "exo_l", "exo_m")
        records[rid] = Record(rid, f"actor{r}", views, {v: 3000 for v in views}, 8)
        for a in range(8):
            name = names[(r + a) % N_ACTIONS]
            kps = [Keypoint(t, bool(rng.integers(0, 2))) for t in kp_vocab[name]]
            act = Action(rid, a + 1, name, 100 + a * 100, 100 + a * 100 + 300, "", kps, [3, 4], ["x", "y"], [None, None])
            actions[act.key] = act
            for v in views:
                samples.append(
                    Sample(f"{act.key}_{v}", rid, f"actor{r}", a + 1, name, v, "ego" if v.startswith("ego") else "exo",
                           100, 400, 3000, act.score, [3, 4], kps, "", 2)
                )
    samples.sort(key=lambda s: s.sample_id)
    ds = Dataset(records=records, actions=actions, samples=samples)

    arr = rng.standard_normal((len(samples), T, D)).astype(np.float16)
    fb = FeatureBundle(array=arr, sample_ids=[s.sample_id for s in samples], index={s.sample_id: i for i, s in enumerate(samples)})
    bundle = build_bundle(samples, kp_vocab, D, T)
    kp_text = torch.randn(len(bundle.action_to_id), bundle.max_keypoints, 512)

    folds = group_kfold(samples, n_splits=3, seed=0)
    assert_no_leakage(samples, folds[0], "smoke")
    print(f"[smoke] 合成样本 {len(samples)} 条, {len(folds)} 折, 每折 val {len(folds[0].val_idx)} 条")

    cfg = TrainConfig(epochs=args.epochs, batch_size=8, num_workers=0, device="cuda" if torch.cuda.is_available() else "cpu", dim=64, depth=2, heads=2)
    out = Path(args.out)
    summary = train_fold(samples, fb, bundle, kp_text, folds[0], cfg, out)
    print(f"[smoke] ✅ 全链路通过: {summary}")


# ---------------------------------------------------------------- train


def cmd_train(args) -> None:
    import torch

    from .annotations import build_keypoint_vocab, load_dataset
    from .data import build_bundle
    from .engine import TrainConfig, train_fold
    from .features import FeatureBundle
    from .losses import LossWeights
    from .splits import assert_no_leakage, group_kfold, holdout
    from .text import build_kp_text_table

    ds = load_dataset(args.raw_dir, debias_scores=getattr(args, "score_debias", False))
    vocab = build_keypoint_vocab(ds)
    samples = ds.samples
    if args.only_views:
        keep = set(args.only_views.split(","))
        samples = [s for s in samples if s.view in keep]

    fb = FeatureBundle.load(args.precomputed)
    # 只保留被预抽取覆盖的样本，并按预抽取的顺序对齐
    samples = [s for s in samples if s.sample_id in fb.index]
    print(f"[train] 样本 {len(samples)} | 预抽取 {len(fb.sample_ids)} | 帧数 {fb.array.shape[1]} | 维度 {fb.array.shape[2]}")

    bundle = build_bundle(samples, vocab, fb.array.shape[2], fb.array.shape[1])
    kp_text = build_kp_text_table(vocab, bundle.action_to_id, out_dir=args.cache_dir)

    folds = group_kfold(samples, n_splits=args.folds, seed=args.seed) if args.folds > 1 else [holdout(samples, seed=args.seed)]

    cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        dropout=args.dropout, dim=args.dim, depth=args.depth, heads=args.heads,
        temporal_jitter=args.temporal_jitter, feat_dropout=args.feat_dropout, score_noise=args.score_noise,
        focal_gamma=args.focal_gamma, ema_decay=args.ema_decay, seed=args.seed, num_workers=args.num_workers,
        amp=not args.no_amp,
        weights=LossWeights(score=args.w_score, keypoint=args.w_keypoint, action=args.w_action, align=args.w_align),
    )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.json").write_text(
        # vars(args) 里含 argparse 塞进去的 func 回调，不是 JSON 可序列化的
        json.dumps({k: v for k, v in vars(args).items() if not callable(v)}, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )

    results = []
    for fold in folds:
        assert_no_leakage(samples, fold, f"fold{fold.fold}")
        results.append(train_fold(samples, fb, bundle, kp_text, fold, cfg, out_root, log=print))

    keys = ["srocc", "plcc", "mae", "acc1", "kp_f1", "kp_f1_best", "srocc_ego", "srocc_exo", "kp_f1_ego", "kp_f1_exo"]
    print("\n" + "=" * 78)
    print(f"{'K 折汇总':<12}" + "".join(f"{k:>12}" for k in keys))
    for i, r in enumerate(results):
        print(f"{'fold ' + str(i):<12}" + "".join(f"{r.get(k, float('nan')):>12.4f}" for k in keys))
    mean = {k: float(np.nanmean([r.get(k, np.nan) for r in results])) for k in keys}
    std = {k: float(np.nanstd([r.get(k, np.nan) for r in results])) for k in keys}
    print(f"{'MEAN':<12}" + "".join(f"{mean[k]:>12.4f}" for k in keys))
    print(f"{'STD':<12}" + "".join(f"{std[k]:>12.4f}" for k in keys))
    print("=" * 78)
    (out_root / "cv_summary.json").write_text(
        json.dumps({"per_fold": results, "mean": mean, "std": std}, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n结果已写入 {out_root}")


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="egoexo", description="EgoExo-Fitness 微调工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stats", help="数据统计 + 标注者一致性（不需要特征）")
    s.add_argument("--raw-dir", default="data/raw_annotations")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("probe", help="探测特征文件结构")
    s.add_argument("--feat-root", default="data/features_open")
    s.add_argument("--max-files", type=int, default=3)
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("extract", help="预抽取定长帧特征")
    s.add_argument("--raw-dir", default="data/raw_annotations")
    s.add_argument("--feat-root", default="data/features_open")
    s.add_argument("--out", default="data/precomputed")
    s.add_argument("--num-frames", type=int, default=32)
    s.add_argument("--crop-ratio", type=float, default=1.0, help="<1 时取时间窗居中部分，裁掉动作首尾噪声")
    s.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    s.add_argument("--only-views", default=None, help="逗号分隔，如 ego_l,ego_m,ego_r")
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("smoke", help="合成数据全链路自检（不需要下载任何数据）")
    s.add_argument("--out", default="runs/smoke")
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--frames", type=int, default=16)
    s.set_defaults(func=cmd_smoke)

    s = sub.add_parser("train", help="K 折训练 + 评估")
    s.add_argument("--raw-dir", default="data/raw_annotations")
    s.add_argument("--precomputed", default="data/precomputed")
    s.add_argument("--cache-dir", default="data/cache")
    s.add_argument("--out", default="runs/exp1")
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--weight-decay", type=float, default=0.05)
    s.add_argument("--dropout", type=float, default=0.2)
    s.add_argument("--dim", type=int, default=256)
    s.add_argument("--depth", type=int, default=3)
    s.add_argument("--heads", type=int, default=4)
    s.add_argument("--temporal-jitter", type=int, default=2)
    s.add_argument("--feat-dropout", type=float, default=0.1)
    s.add_argument("--score-noise", type=float, default=0.15)
    s.add_argument("--focal-gamma", type=float, default=0.0)
    s.add_argument("--ema-decay", type=float, default=0.0)
    s.add_argument("--w-score", type=float, default=1.0)
    s.add_argument("--w-keypoint", type=float, default=1.0)
    s.add_argument("--w-action", type=float, default=0.3)
    s.add_argument("--w-align", type=float, default=0.7, help="跨视角 InfoNCE 权重，论文用 0.7")
    s.add_argument("--only-views", default=None)
    s.add_argument("--score-debias", action="store_true",
                   help="按标注者个人均值去偏（实测逐对完全相等率 31%%->40%%）。"
                        "标签在训练时现算，所以只在这里设就够了")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--num-workers", type=int, default=2)
    s.add_argument("--no-amp", action="store_true")
    s.set_defaults(func=cmd_train)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
