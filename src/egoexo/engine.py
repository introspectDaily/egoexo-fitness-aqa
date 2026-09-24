"""训练 / 评估循环。

硬件注意：Colab T4 是 Turing 架构，**不支持 bf16**（只有 fp16）。
所以这里用 autocast(dtype=float16) + GradScaler，而不是 bf16 的写法。
本模型参数量 <5M、输入是 512 维特征而不是像素，T4 上跑得很快。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import AqaDataset, collate
from .losses import LossWeights, total_loss
from .metrics import best_threshold, keypoint_metrics, per_action_breakdown, score_metrics
from .models import AqaModel, infonce_alignment


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    dropout: float = 0.2
    dim: int = 256
    depth: int = 3
    heads: int = 4
    temporal_jitter: int = 2
    feat_dropout: float = 0.1
    score_noise: float = 0.15
    focal_gamma: float = 0.0
    ema_decay: float = 0.0  # >0 时启用 EMA 权重
    seed: int = 0
    num_workers: int = 2
    device: str = "cuda"
    amp: bool = True
    weights: LossWeights = field(default_factory=LossWeights)


# ---------------------------------------------------------------- 工具


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    """权重指数滑动平均。小数据集上通常能稳定涨 1~2 个点，几乎零成本。"""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def apply_to(self, model: torch.nn.Module) -> None:
        self.backup = {}
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.backup[k] = v.detach().clone()
                v.copy_(self.shadow[k].to(v.dtype))

    def restore(self, model: torch.nn.Module) -> None:
        for k, v in self.backup.items():
            model.state_dict()[k].copy_(v)
        self.backup = {}


def build_action_uids(samples, indices) -> torch.Tensor:
    """把「同一物理动作的不同视角」映射到同一个 uid，供 InfoNCE 用。"""
    key_to_uid: dict[str, int] = {}
    uids = []
    for i in indices:
        s = samples[i]
        key = f"{s.record_id}#{s.action_idx}"
        uids.append(key_to_uid.setdefault(key, len(key_to_uid)))
    return torch.tensor(uids, dtype=torch.long)


def compute_pos_weight(samples, indices, clip: float = 20.0) -> float:
    pos = neg = 0
    for i in indices:
        for k in samples[i].keypoints:
            pos += k.label
            neg += 1 - k.label
    if pos == 0:
        return 1.0
    return float(min(neg / pos, clip))


# ---------------------------------------------------------------- 评估


@torch.no_grad()
def evaluate(model, loader, samples, indices, device, amp: bool = True) -> dict:
    model.eval()
    preds, targets, kp_logits, kp_labels, kp_masks, order, view_types = [], [], [], [], [], [], []

    for batch in loader:
        feat = batch["feat"].to(device, non_blocking=True)
        action_id = batch["action_id"].to(device, non_blocking=True)
        view_id = batch["view_id"].to(device, non_blocking=True)
        kp_mask = batch["kp_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp and device.startswith("cuda")):
            out = model(feat, action_id, view_id, kp_mask)

        preds.append(out.score.float().cpu().numpy())
        kp_logits.append(out.keypoint_logits.float().cpu().numpy())
        targets.append(batch["score"].numpy())
        kp_labels.append(batch["keypoint"].numpy())
        kp_masks.append(batch["kp_mask"].numpy())
        order.append(batch["index"].numpy())
        view_types.extend("ego" if v else "exo" for v in batch["is_ego"].numpy().tolist())

    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    logits = np.concatenate(kp_logits)
    labels = np.concatenate(kp_labels)
    masks = np.concatenate(kp_masks)
    order = np.concatenate(order)
    view_types = np.array(view_types)

    res: dict = {"overall": {}, "ego": {}, "exo": {}}
    for name, sel in (("overall", np.ones(len(pred), bool)), ("ego", view_types == "ego"), ("exo", view_types == "exo")):
        if sel.sum() == 0:
            continue
        res[name]["score"] = score_metrics(pred[sel], target[sel])
        res[name]["keypoint@0.5"] = keypoint_metrics(logits[sel], labels[sel], masks[sel], 0.5)

    # 阈值只在总体扫一次，然后套用到 ego/exo，避免各自调阈值导致的乐观偏差
    thr, f1 = best_threshold(logits, labels, masks)
    res["best_threshold"] = {"threshold": thr, "f1": f1}
    for name, sel in (("overall", np.ones(len(pred), bool)), ("ego", view_types == "ego"), ("exo", view_types == "exo")):
        if sel.sum():
            res[name]["keypoint@best"] = keypoint_metrics(logits[sel], labels[sel], masks[sel], thr)

    res["per_action"] = per_action_breakdown(samples, order, pred, target)
    res["_raw"] = {"pred": pred, "target": target, "order": order, "view_types": view_types}
    return res


def summarize(res: dict) -> dict[str, float]:
    o = res["overall"]
    return {
        "srocc": o["score"]["srocc"],
        "plcc": o["score"]["plcc"],
        "mae": o["score"]["mae"],
        "acc1": o["score"]["acc1"],
        "kp_f1": o["keypoint@0.5"]["f1"],
        "kp_f1_best": res.get("best_threshold", {}).get("f1", float("nan")),
        "kp_prec": o["keypoint@0.5"]["precision"],
        "kp_rec": o["keypoint@0.5"]["recall"],
        "srocc_ego": res.get("ego", {}).get("score", {}).get("srocc", float("nan")),
        "srocc_exo": res.get("exo", {}).get("score", {}).get("srocc", float("nan")),
        "kp_f1_ego": res.get("ego", {}).get("keypoint@0.5", {}).get("f1", float("nan")),
        "kp_f1_exo": res.get("exo", {}).get("keypoint@0.5", {}).get("f1", float("nan")),
    }


# ---------------------------------------------------------------- 单折训练


def train_fold(
    samples,
    features,
    bundle,
    kp_text_emb: torch.Tensor,
    split,
    cfg: TrainConfig,
    out_dir: str | Path,
    log=print,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    train_ds = AqaDataset(
        samples, features, bundle, split.train_idx,
        augment=True, temporal_jitter=cfg.temporal_jitter, feat_dropout=cfg.feat_dropout, score_noise=cfg.score_noise,
    )
    val_ds = AqaDataset(samples, features, bundle, split.val_idx, augment=False)

    train_ld = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        collate_fn=collate, drop_last=len(train_ds) > cfg.batch_size, pin_memory=True, persistent_workers=cfg.num_workers > 0,
    )
    val_ld = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers,
        collate_fn=collate, pin_memory=True, persistent_workers=cfg.num_workers > 0,
    )

    model = AqaModel(
        num_actions=len(bundle.action_to_id),
        kp_text_emb=kp_text_emb,
        dim=cfg.dim, depth=cfg.depth, heads=cfg.heads, dropout=cfg.dropout,
        num_frames=bundle.num_frames,
    ).to(device)

    pos_weight = torch.tensor(compute_pos_weight(samples, split.train_idx), device=device)
    log(f"[fold {split.fold}] train={len(train_ds)} val={len(val_ds)} | "
        f"records train={len(split.train_records)} val={len(split.val_records)} | pos_weight={float(pos_weight):.3f}")
    assert not (set(split.train_records) & set(split.val_records)), "record 泄漏！"

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_ld))
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = cfg.amp and device.startswith("cuda")
    # T4 不支持 bf16，必须 fp16 + GradScaler。
    # torch>=2.1 用 torch.amp.GradScaler("cuda")，老版本只有 torch.cuda.amp.GradScaler。
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # pragma: no cover
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None

    # 全局动作 uid 表：把「同一物理动作的不同视角」映射到同一 uid，供 InfoNCE 用。
    # ⚠️ 必须按**全局样本下标**建表（batch["index"] 是全局下标），
    #    不能只对 split.train_idx 建——那样索引会错位。
    action_uids = build_action_uids(samples, np.arange(len(samples))).to(device)

    def align_fn(z, batch):
        uids = action_uids[batch["index"].to(device)]
        return infonce_alignment(z, uids, batch["is_ego"].to(device))

    best = {"score": -float("inf"), "state": None, "epoch": -1}
    history = []
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        agg: dict[str, float] = {}
        n_batches = 0
        for batch in train_ld:
            feat = batch["feat"].to(device, non_blocking=True)
            action_id = batch["action_id"].to(device, non_blocking=True)
            view_id = batch["view_id"].to(device, non_blocking=True)
            kp_mask = batch["kp_mask"].to(device, non_blocking=True)
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(feat, action_id, view_id, kp_mask)
                loss, parts = total_loss(out, batch, cfg.weights, pos_weight, cfg.focal_gamma, align_fn)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model)

            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            n_batches += 1

        avg = {k: v / max(1, n_batches) for k, v in agg.items()}

        if ema is not None:
            ema.apply_to(model)
        # 评估用的是 EMA 权重，所以「最优权重」必须从当下这刻抓，否则会存下未 EMA 的版本，
        # 结果就是：选 epoch 用的是一套权重，最后报告的是另一套。
        state_now = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        res = evaluate(model, val_ld, samples, split.val_idx, device, use_amp)
        if ema is not None:
            ema.restore(model)
        s = summarize(res)

        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in avg.items()}, **s})
        log(f"[fold {split.fold}] ep {epoch + 1:3d}/{cfg.epochs} loss={avg.get('total', float('nan')):.4f} "
            f"| val SROCC={s['srocc']:.4f} MAE={s['mae']:.4f} | KP F1={s['kp_f1']:.4f} (best {s['kp_f1_best']:.4f})")

        # 用 SROCC 早停（AQA 领域主指标），而不是 loss。
        # 少数情况下 SROCC 会是 NaN —— :func:`metrics._safe_spearman` 在预测接近常数时
        # 主动返回 NaN（此时秩相关无定义）。这时退回 -MAE，否则 best 会永远停在初始值，
        # 最后 load_state_dict(None) 直接崩。
        crit = s["srocc"]
        if np.isnan(crit):
            crit = -s["mae"]
        if crit > best["score"]:
            best = {"score": crit, "state": state_now, "epoch": epoch}

    model.load_state_dict(best["state"])
    final = evaluate(model, val_ld, samples, split.val_idx, device, use_amp)

    torch.save(
        {"state_dict": best["state"], "config": asdict(cfg), "bundle": bundle.save_meta(), "epoch": best["epoch"]},
        out_dir / f"fold{split.fold}_best.pt",
    )
    with open(out_dir / f"fold{split.fold}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=1)
    np.savez(
        out_dir / f"fold{split.fold}_val_raw.npz",
        pred=final["_raw"]["pred"], target=final["_raw"]["target"],
        order=final["_raw"]["order"], view_types=final["_raw"]["view_types"],
    )

    summary = summarize(final)
    summary["best_epoch"] = best["epoch"]
    summary["minutes"] = (time.time() - t0) / 60
    with open(out_dir / f"fold{split.fold}_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_action": final["per_action"],
                   "overall": final["overall"], "ego": final["ego"], "exo": final["exo"]}, f, indent=1, ensure_ascii=False)

    log(f"[fold {split.fold}] 最优 epoch={best['epoch']} | {summary}")
    return summary
