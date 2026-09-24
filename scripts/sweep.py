#!/usr/bin/env python3
"""并行消融 / 扫参调度器。

## 为什么并行

实测这个任务的 **GPU 利用率只有 15%**：模型 <5M 参数、输入是预抽取好的 176MB
特征矩阵，瓶颈在 Python/DataLoader 而不是算力。实测并发 10 个任务后利用率到 100%，
显存只用 4.6G/20G。所以「串行跑再加长训练」是纯浪费。

## 为什么可以横比

所有任务用 `--fold N --folds M` 固定同一折，验证集完全相同，指标可直接对比。
（`group_kfold` 的折划分只依赖 seed 和 record 顺序，所以不同配置的同一折是同一批
验证 record。）

## 用法

    python scripts/sweep.py --epochs 60 --concurrency 10
    python scripts/sweep.py --only base,no_worst        # 只跑某几个
    python scripts/sweep.py --status                    # 只看已有结果，不跑

结果增量写入 <out>/results.json，每个任务的日志在 <out>/logs/<job>.log。

## ⚠️ 一定要看每折的结果，不能只看均值

实测折间方差（KP F1 0.047~0.124）**远大于**配置间差异（最大 0.039）。
2 折不足以给配置排名；`--report` 会把按折拆开的结果一起打出来。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

PROJ = pathlib.Path(__file__).resolve().parents[1]

# 配置名 -> 额外 CLI 参数。默认 --precomputed 由 BASE 统一给，避免和 CLI 默认值漂移
# （踩过一次：抽取目录名和 CLI 默认值不一致，8 个任务全部 FileNotFoundError）
BASE_FEATURES = "data/precomputed"

CONFIGS: dict[str, list[str]] = {
    "base": [],
    "no_worst": ["--no-worst-frame"],
    "crop10": ["--precomputed", "data/pc_v32c10"],
    "f64": ["--precomputed", "data/pc_v64c10"],
    "big": ["--dim", "384", "--depth", "4", "--heads", "6"],
    "debias": ["--score-debias"],
    "noalign": ["--w-align", "0.0"],
    "focal": ["--focal-gamma", "2.0"],
    "highdrop": ["--dropout", "0.4", "--feat-dropout", "0.2"],
    "long150": ["--epochs", "150"],
    "bigbatch": ["--batch-size", "256", "--lr", "8e-4"],
}

METRIC_KEYS = ["srocc", "plcc", "mae", "acc1", "kp_f1", "kp_f1_best",
               "srocc_ego", "srocc_exo", "kp_f1_ego", "kp_f1_exo",
               "fused_srocc", "fused_kp_f1", "fused_kp_best"]


def collect(out: pathlib.Path) -> dict[str, dict]:
    """扫描 <out>/<job>/cv_summary.json，聚合成 {config: 按折均值}。"""
    per: dict[str, list[tuple[int, dict]]] = {}
    for d in sorted(out.glob("*_f*")):
        f = d / "cv_summary.json"
        if not f.exists():
            continue
        name = d.name
        if "_f" not in name:
            continue
        config, _, fold = name.rpartition("_f")
        if not fold.isdigit():
            continue
        s = json.loads(f.read_text())["per_fold"][0]
        per.setdefault(config, []).append((int(fold), s))

    out_agg: dict[str, dict] = {}
    for config, rows in per.items():
        n = len(rows)
        agg = {k: sum(r[1].get(k, float("nan")) for r in rows) / n for k in METRIC_KEYS}
        agg["n_folds"] = n
        agg["folds"] = sorted(r[0] for r in rows)
        # 折间极差：判断配置差异是否可信的关键数字
        agg["kp_f1_spread"] = (max(r[1].get("kp_f1", 0) for r in rows)
                               - min(r[1].get("kp_f1", 0) for r in rows)) if n > 1 else 0.0
        out_agg[config] = agg
    return out_agg


def report(out: pathlib.Path) -> None:
    agg = collect(out)
    if not agg:
        print("(没有已完成的结果)")
        return
    hdr = ["kp_f1", "kp_f1_best", "srocc", "mae", "srocc_ego", "srocc_exo", "fused_kp_best", "spread", "n"]
    print(f"{'config':<12}" + "".join(f"{h:>13}" for h in hdr))
    print("-" * (12 + 13 * len(hdr)))
    for c, m in sorted(agg.items(), key=lambda kv: -kv[1]["kp_f1_best"]):
        vals = [m["kp_f1"], m["kp_f1_best"], m["srocc"], m["mae"], m["srocc_ego"], m["srocc_exo"],
                m.get("fused_kp_best", float("nan")), m["kp_f1_spread"], m["n_folds"]]
        print(f"{c:<12}" + "".join(f"{v:>13.4f}" if isinstance(v, float) else f"{v:>13}" for v in vals))
    print()
    print("对照：论文 GEVFormer 0.5439 | 论文 CLIP-GEV 朴素基线 0.4881 | Random 0.3178 | 多数类 0.3265")
    spread = max(m["kp_f1_spread"] for m in agg.values())
    diff = max(m["kp_f1_best"] for m in agg.values()) - min(m["kp_f1_best"] for m in agg.values())
    print(f"折间极差最大 {spread:.4f}  vs  配置间差异最大 {diff:.4f}", end="  -> ")
    print("⚠️ 噪声主导，不要排名" if spread > diff else "配置差异可信")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/sweep")
    ap.add_argument("--folds", type=int, default=5, help="总折数")
    ap.add_argument("--fold-list", default="0,1", help="要跑哪些折，逗号分隔")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3, help="每个任务的数据加载进程数")
    ap.add_argument("--amp-dtype", default="bf16")
    ap.add_argument("--configs", default=None, help="逗号分隔的配置名，默认全部")
    ap.add_argument("--only", default=None, help="只跑这些配置（与 --configs 同义）")
    ap.add_argument("--status", action="store_true", help="只打印已有结果，不启动任务")
    args = ap.parse_args()

    out = (PROJ / args.out) if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    if args.status:
        report(out)
        return

    names = (args.only or args.configs)
    names = [n.strip() for n in names.split(",")] if names else list(CONFIGS)
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        sys.exit(f"未知配置: {unknown}。可选: {list(CONFIGS)}")

    folds = [int(f) for f in args.fold_list.split(",")]
    jobs = [(f"{c}_f{f}", CONFIGS[c], f) for c in names for f in folds]

    running: dict[str, tuple] = {}
    queue = list(jobs)
    status = {j: "pending" for j, _, _ in jobs}
    print(f"[sweep] {len(jobs)} 个任务, 并发 {args.concurrency}, {args.epochs} epoch", flush=True)

    def snapshot() -> None:
        rows = [{"job": j, "config": c, "fold": f, "status": status.get(j, "pending")}
                for j, c, f in jobs]
        (out / "results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))

    while queue or running:
        while queue and len(running) < args.concurrency:
            job, extra, fold = queue.pop(0)
            if (out / job / "cv_summary.json").exists():
                status[job] = "cached"
                continue
            lf = open(out / "logs" / f"{job}.log", "w")
            cmd = [sys.executable, "-u", "-m", "egoexo.cli", "train",
                   "--out", str(out / job), "--precomputed", BASE_FEATURES,
                   "--amp-dtype", args.amp_dtype, "--folds", str(args.folds), "--fold", str(fold),
                   "--epochs", str(args.epochs), "--num-workers", str(args.workers)] + extra
            running[job] = (subprocess.Popen(cmd, cwd=PROJ, stdout=lf, stderr=subprocess.STDOUT), lf, time.time())
            status[job] = "running"
            print(f"[sweep] START {job}", flush=True)
        time.sleep(8)
        for job in list(running):
            p, lf, t0 = running[job]
            if p.poll() is not None:
                lf.close()
                running.pop(job)
                f = out / job / "cv_summary.json"
                m = json.loads(f.read_text())["per_fold"][0] if f.exists() else None
                status[job] = "done" if m else f"FAIL(rc={p.returncode})"
                info = (f"srocc={m['srocc']:.4f} kp_f1={m['kp_f1']:.4f} best={m['kp_f1_best']:.4f}"
                        if m else "")
                print(f"[sweep] {status[job]:<14} {job:<14} {info}  ({time.time()-t0:.0f}s)", flush=True)
                snapshot()
    snapshot()
    print("\n[sweep] 全部结束\n", flush=True)
    report(out)


if __name__ == "__main__":
    main()
