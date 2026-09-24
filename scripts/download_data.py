#!/usr/bin/env python3
"""下载 EgoExo-Fitness 的标注与视觉特征。

数据是 gated 的，必须先：
  1. 在 https://huggingface.co/datasets/Lymann/EgoExo-Fitness 点 "Agree and access repository"
  2. 用**同一个账号**的 token（见 README 的权限说明）

用法：
    python scripts/download_data.py                    # 标注 + 特征全下
    python scripts/download_data.py --annotations-only # 只下标注（6MB，秒下）
    python scripts/download_data.py --skip-extract     # 下完不解压

下载量：标注 ~6MB；视觉特征 6.4GB（2 个分片）；可视帧另有 ~67GB，本脚本不下。

⚠️ 特征分片是 `split` 切的，必须**按序合并**再解压，漏一片会报 unexpected end of file。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ID = "Lymann/EgoExo-Fitness"
REPO_TYPE = "dataset"

ANNOTATIONS = [
    "meta_records.json",
    "action_level_annotations.json",
    "interpretable_action_judgement.json",
    "subaction_level_annotations_ant13_style_v1.json",
]

FEATURE_SHARDS = [
    "features_open/visual/EgoExo_Fitness_CLIP_Vid_Feat_w_Rotate.tar.gz.aa",
    "features_open/visual/EgoExo_Fitness_CLIP_Vid_Feat_w_Rotate.tar.gz.ab",
]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download(repo_files: list[str], local_dir: Path) -> list[Path]:
    """走 hf_hub_download（自带断点续传）。返回落地后的实际路径。"""
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[download] 没有 huggingface_hub，先 pip install huggingface_hub", file=sys.stderr)
        raise

    # 让下载快点；hf_transfer 没装也不影响
    import os

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    out = []
    for f in repo_files:
        t0 = time.time()
        try:
            p = hf_hub_download(
                repo_id=REPO_ID,
                filename=f,
                repo_type=REPO_TYPE,
                local_dir=str(local_dir),
            )
        except Exception as e:  # noqa: BLE001
            hint = ""
            msg = str(e)
            if "401" in msg or "403" in msg or "restricted" in msg.lower() or "gated" in msg.lower():
                hint = (
                    "\n  → 这是权限问题，按顺序排查：\n"
                    "    1. 你是不是**同意了条款的那个账号**？（换个账号的 token 一样被拒）\n"
                    "    2. Fine-grained token 除了 'public repos' 还必须勾上\n"
                    "       'Read access to contents of all public GATED repos you can access'\n"
                    "       —— 这两项在权限页面上是分开的，漏勾会 401。\n"
                    "    3. shell 里是不是残留了旧账号的 HF_TOKEN 环境变量（它的优先级高于本地登录）？\n"
                    "       检查: env | grep HF_TOKEN"
                )
            raise RuntimeError(f"下载 {f} 失败: {e}{hint}") from e
        sz = Path(p).stat().st_size
        print(f"[download] {f}  {human(sz)}  {time.time()-t0:.0f}s")
        out.append(Path(p))
    return out


def merge_and_extract(shard_paths: list[Path], dest: Path) -> None:
    """按 .aa/.ab/... 顺序合并成一个 tar.gz，然后解压，最后删掉中间产物。"""
    shard_paths = sorted(shard_paths, key=lambda p: p.name)
    dest.mkdir(parents=True, exist_ok=True)
    merged = dest / "features.tar.gz"

    if not merged.exists() or merged.stat().st_size < sum(p.stat().st_size for p in shard_paths):
        print(f"[merge] 合并 {len(shard_paths)} 个分片 -> {merged}")
        with open(merged, "wb") as out:
            for p in shard_paths:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out, length=8 << 20)
    print(f"[merge] 大小 {human(merged.stat().st_size)}")

    print(f"[extract] 解压到 {dest}")
    r = subprocess.run(["tar", "-xzf", str(merged), "-C", str(dest)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"解压失败（多半是分片没下全或顺序不对）:\n{r.stderr[:500]}\n"
            f"  分片大小: {[(p.name, human(p.stat().st_size)) for p in shard_paths]}"
        )

    # 分片和合并包加起来 ~9.6GB，解压完就没用了
    for p in [*shard_paths, merged]:
        p.unlink(missing_ok=True)
    print("[extract] 已清理分片与合并包")


def main() -> None:
    ap = argparse.ArgumentParser(description="下载 EgoExo-Fitness 标注与特征")
    ap.add_argument("--root", default="data", help="数据根目录")
    ap.add_argument("--annotations-only", action="store_true")
    ap.add_argument("--skip-extract", action="store_true", help="下载分片但不解压")
    args = ap.parse_args()

    root = Path(args.root)
    ann_dir = root / "raw_annotations"
    feat_dir = root / "features_open"

    print("=" * 70)
    print("EgoExo-Fitness 下载")
    print("=" * 70)

    # 先把凭证链走通，后面的报错才不会含糊
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from egoexo.secrets import apply_hf_endpoint, describe, hf_login

    ep = apply_hf_endpoint()
    print(f"[auth] 凭证状态: {describe()}")
    print(f"[auth] HF_ENDPOINT: {ep or '<默认 huggingface.co>（国内网络会连不上）'}")
    token = hf_login(quiet=False)
    if token is None:
        print("[auth] ⚠️ 未找到 HF token，只能下公开文件；gated 文件会 401")

    # hf_hub_download 带 local_dir 时会把文件落到 local_dir 下的相对路径，
    # 所以 raw_annotations/*.json 已经在最终位置了，不需要再搬
    ann = download([f"raw_annotations/{f}" for f in ANNOTATIONS], root)
    ann_dir.mkdir(parents=True, exist_ok=True)
    for p in ann:
        p = Path(p)
        if p.parent.resolve() != ann_dir.resolve():
            shutil.move(str(p), str(ann_dir / p.name))
    missing = [f for f in ANNOTATIONS if not (ann_dir / f).exists()]
    if missing:
        raise RuntimeError(f"标注没落到位，缺: {missing}。实际目录: {sorted(x.name for x in ann_dir.iterdir())[:10]}")
    print(f"[download] 标注完成 -> {ann_dir}")

    if args.annotations_only:
        print("\n只下标注，结束。")
        return

    shards = download(FEATURE_SHARDS, root)
    if not args.skip_extract:
        merge_and_extract([Path(p) for p in shards], feat_dir)
        print(f"\n[download] 特征就绪 -> {feat_dir}")
        print("  下一步: python -m egoexo.cli probe --feat-root", feat_dir)


if __name__ == "__main__":
    main()
