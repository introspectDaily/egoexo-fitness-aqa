"""逐帧 CLIP 特征的发现、探测与预抽取。

数据集给的目录结构（已实测）：

    <features_root>/EgoExo_Fitness_CLIP_Vid_Feat_w_Rotate/<record_id>/<view>/clip_vit_b32_vid_frame_feat.pth

即 **每个 (record_id, view) 一个 .pth**，装的是该视角整段视频的逐帧 CLIP ViT-B/32 特征（D=512）。
一个 record 可能有 8000 帧 → 该 .pth 约 16MB(fp32)，全量 6.4GB。

因此**不要**在训练时按需读 .pth（每次都要解 16MB pickle，且被多个 action 反复读）。
本模块在训练前做一次「预抽取」：只把每个 action 时间窗内的帧切出来、重采样到固定 T 帧，
拼成一个大 (N, T, D) float16 数组 + 索引 json。全量约 176MB，之后训练就是纯内存读取。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURE_FILENAME = "clip_vit_b32_vid_frame_feat.pth"
CLIP_B32_DIM = 512  # CLIP ViT-B/32 的嵌入维度


# ---------------------------------------------------------------- 发现


def discover(features_root: str | Path) -> dict[tuple[str, str], Path]:
    """递归找出所有 (record_id, view) -> 特征文件路径。

    用「遍历 + 从父目录名推断」而不是硬拼路径：用户可能把 tar 解到任何地方，
    也可能把 .aa/.ab 合并后解到别的子目录。
    """
    root = Path(features_root)
    if not root.exists():
        raise FileNotFoundError(f"特征目录不存在: {root}")

    found: dict[tuple[str, str], Path] = {}
    for p in root.rglob(FEATURE_FILENAME):
        # .../<record_id>/<view>/clip_vit_b32_vid_frame_feat.pth
        if len(p.parents) < 2:
            continue
        view = p.parent.name
        record_id = p.parent.parent.name
        if view not in {"ego_l", "ego_m", "ego_r", "exo_l", "exo_m", "exo_r"}:
            # 目录层级不对（比如被裹了两层），尝试上溯找到合法 view 名
            parts = p.parts
            hit = None
            for i, name in enumerate(parts):
                if name in {"ego_l", "ego_m", "ego_r", "exo_l", "exo_m", "exo_r"}:
                    hit = i
                    break
            if hit is None or hit == 0:
                continue
            view, record_id = parts[hit], parts[hit - 1]
        found[(record_id, view)] = p

    if not found:
        raise FileNotFoundError(
            f"在 {root} 下没找到任何 {FEATURE_FILENAME}。\n"
            f"  常见原因: (a) 分片没合并就解压 → tar 报 unexpected end of file 但你没注意；\n"
            f"            (b) 解压到了别处；\n"
            f"            (c) 你只下了 frames 包而不是 features 包。"
        )
    return found


# ---------------------------------------------------------------- 张量归一化


# .pth 里 tensor 常见 key 名。实测 EgoExo-Fitness 用的是 "clip_feat"，
# 但同时保留其他候选，因为作者以后重新打包可能会换名。
_FEAT_KEYS = (
    "clip_feat",
    "feat",
    "feats",
    "feature",
    "features",
    "video_features",
    "frame_features",
    "x",
)


def to_frame_matrix(obj, dim: int = CLIP_B32_DIM) -> np.ndarray:
    """把 .pth 里反序列化出来的任意东西规整成 (T, D) 的 float32 数组。

    实测结构：`{"clip_feat": <tensor>, "view": "ego_l", "record": "08ALrC"}`
    但这些 .pth 是作者用 torch.save 随手存的，格式无文档。所以做防御式解析：
    先查常见 key，再回退到“取第一个 tensor/ndarray 值”，最后处理 ndim==3
    （空间 token 时对 token 维求均值）。
    """
    import torch

    if isinstance(obj, dict):
        picked = None
        for key in _FEAT_KEYS:
            if key in obj:
                picked = obj[key]
                break
        if picked is None:
            # 未知 key：取第一个 tensor/ndarray 值（元信息如 'view'/'record' 是字符串，会被跳过）
            for v in obj.values():
                if torch.is_tensor(v) or isinstance(v, np.ndarray):
                    picked = v
                    break
        if picked is None:
            raise ValueError(f"无法从 .pth 的 dict 里找到特征张量，keys={list(obj)[:10]}")
        obj = picked

    if isinstance(obj, (list, tuple)):
        obj = obj[0]

    if isinstance(obj, np.ndarray):
        arr = obj
    elif torch.is_tensor(obj):
        arr = obj.detach().cpu().float().numpy()
    else:
        raise TypeError(f"不认识的 .pth 内容类型: {type(obj)}")

    if arr.ndim == 3:
        # (T, S, D) 空间 token -> 对 token 维求均值
        # 判定哪一维是 D：等于 dim 的那个；退而求其次取最后一维
        if arr.shape[-1] == dim:
            arr = arr.mean(axis=1)
        elif arr.shape[1] == dim:
            arr = arr.mean(axis=2)
        else:
            arr = arr.reshape(arr.shape[0], -1)
    elif arr.ndim == 1:
        arr = arr[None, :]

    if arr.ndim != 2:
        raise ValueError(f"无法规整成 (T, D)，实际 shape={arr.shape}")

    # 决定哪一维是 D：优先等于 512 的那一维，其次取较小的一维（帧数通常远大于 512）
    if arr.shape[1] != dim and arr.shape[0] == dim:
        arr = arr.T
    elif arr.shape[1] != dim and arr.shape[0] != dim:
        arr = arr.T if arr.shape[0] < arr.shape[1] else arr

    return np.ascontiguousarray(arr, dtype=np.float32)


def probe(features_root: str | Path, max_files: int = 3) -> list[dict]:
    """打印若干个特征文件的结构，用来确认解析假设成立。"""
    import torch

    files = discover(features_root)
    print(f"[probe] 共发现 {len(files)} 个 (record, view) 特征文件")
    out = []
    for (rid, view), path in list(sorted(files.items()))[:max_files]:
        info = {"record_id": rid, "view": view, "path": str(path), "size_mb": path.stat().st_size / 1e6}
        try:
            obj = _safe_torch_load(path)
            info["raw_type"] = type(obj).__name__
            if isinstance(obj, dict):
                info["dict_keys"] = list(obj)[:10]
                # 把 dict 里每个值的类型/形状也报出来 —— 万一 key 名变了，看这里就能定位
                info["dict_values"] = {
                    k: (tuple(v.shape) if hasattr(v, "shape") else repr(v)[:40]) for k, v in list(obj.items())[:10]
                }
            if torch.is_tensor(obj):
                info["raw_shape"] = tuple(obj.shape)
                info["raw_dtype"] = str(obj.dtype)
            arr = to_frame_matrix(obj)
            info["frames"] = int(arr.shape[0])
            info["dim"] = int(arr.shape[1])
            info["frame_dim_ok"] = info["dim"] == CLIP_B32_DIM
            info["feat_min"] = float(arr.min())
            info["feat_max"] = float(arr.max())
            info["feat_norm_mean"] = float(np.linalg.norm(arr, axis=-1).mean())
        except Exception as e:  # noqa: BLE001
            info["error"] = f"{type(e).__name__}: {e}"
        out.append(info)
        print(f"[probe] {rid}/{view}: {info}")
    return out


def _safe_torch_load(path: Path):
    """先试 weights_only=True（更安全），失败再退回全量反序列化。

    这些文件来自可信数据集，且我们跑在一次性 Colab VM 上，所以回退是可接受的；
    但如果哪天要在共享/生产环境跑，应当改为只接受 weights_only。
    """
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------- 预抽取


@dataclass
class FeatureBundle:
    """预抽取结果：定长帧序列 + 与样本表的对齐索引。"""

    array: np.ndarray  # (N, T, D) float16
    sample_ids: list[str]
    index: dict[str, int]

    def get(self, sample_id: str) -> np.ndarray:
        return self.array[self.index[sample_id]]

    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "features.npy", self.array)
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_samples": len(self.sample_ids),
                    "num_frames": int(self.array.shape[1]),
                    "dim": int(self.array.shape[2]),
                    "dtype": str(self.array.dtype),
                    "sample_ids": self.sample_ids,
                },
                f,
                indent=1,
            )
        return out_dir

    @classmethod
    def load(cls, out_dir: str | Path, ram_threshold_gb: float = 2.0) -> "FeatureBundle":
        """加载预抽取结果。

        小于 ram_threshold_gb 时**直接读进内存**而不是 mmap。全量只有 176MB(fp16/32帧)，
        而 mmap 会让每个样本的读取都走一遍页缓存/磁盘路径 —— 实测 GPU 利用率只有 15%，
        瓶颈就在这类开销上，不在算力。
        """
        out_dir = Path(out_dir)
        with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
            man = json.load(f)
        size_gb = (out_dir / "features.npy").stat().st_size / 1e9
        if size_gb <= ram_threshold_gb:
            arr = np.load(out_dir / "features.npy")
            print(f"[features] 已载入内存: {size_gb:.2f}GB ({arr.shape})")
        else:
            arr = np.load(out_dir / "features.npy", mmap_mode="r")
            print(f"[features] 使用 mmap: {size_gb:.2f}GB ({arr.shape})")
        ids = man["sample_ids"]
        return cls(array=arr, sample_ids=ids, index={s: i for i, s in enumerate(ids)})


def _resample_indices(st: int, ed: int, num_frames: int, crop_ratio: float = 1.0) -> np.ndarray:
    """在 [st, ed) 内取 num_frames 个帧下标。

    crop_ratio < 1 时只取时间窗**居中**的一段 —— 动作的首尾（走近、站定、喘气）
    对质量分贡献很小，是纯噪声。论文的 3 段子步骤标注（准备/执行/放松）没随
    这份 release 一起放出来，所以用居中裁剪作为近似。
    """
    length = max(1, ed - st)
    if crop_ratio < 1.0:
        keep = max(1, int(round(length * crop_ratio)))
        offset = (length - keep) // 2
        st, length = st + offset, keep
    # endpoint=False: 不要取到 ed（那是下一动作的起点）
    return np.linspace(st, st + length - 1, num_frames).round().astype(np.int64)


def extract(
    samples,
    features_root: str | Path,
    out_dir: str | Path,
    num_frames: int = 32,
    crop_ratio: float = 1.0,
    dtype: str = "float16",
    verbose: bool = True,
) -> FeatureBundle:
    """把样本表里每个 (action, view) 切出来，重采样成定长 T 帧。

    按 (record, view) 分组，保证每个 16MB 的 .pth 只解一次。
    """
    files = discover(features_root)
    groups: dict[tuple[str, str], list] = {}
    for s in samples:
        groups.setdefault((s.record_id, s.view), []).append(s)

    missing = [k for k in groups if k not in files]
    if missing:
        raise FileNotFoundError(
            f"有 {len(missing)} 个 (record, view) 在特征目录里找不到，例如 {missing[:5]}。"
            f" 说明特征包不完整。"
        )

    n, dim = len(samples), CLIP_B32_DIM
    arr = np.zeros((n, num_frames, dim), dtype=np.dtype(dtype))
    ids = [s.sample_id for s in samples]
    index = {s: i for i, s in enumerate(ids)}

    done = 0
    for (rid, view), group in sorted(groups.items()):
        raw = to_frame_matrix(_safe_torch_load(files[(rid, view)]))
        if raw.shape[1] != dim:
            raise ValueError(f"{rid}/{view} 的特征维度是 {raw.shape[1]}，期望 {dim}")
        total = raw.shape[0]
        for s in group:
            st = max(0, min(s.st, total - 1))
            ed = max(st + 1, min(s.ed, total))
            idx = _resample_indices(st, ed, num_frames, crop_ratio)
            np.clip(idx, 0, total - 1, out=idx)
            arr[index[s.sample_id]] = raw[idx].astype(dtype, copy=False)
        done += 1
        if verbose and done % 25 == 0:
            print(f"[extract] {done}/{len(groups)} 个 (record, view) 已处理")

    bundle = FeatureBundle(array=arr, sample_ids=ids, index=index)
    bundle.save(out_dir)
    if verbose:
        print(f"[extract] 完成 -> {out_dir}  shape={arr.shape}  size={arr.nbytes/1e6:.1f}MB")
    return bundle
