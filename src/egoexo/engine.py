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
    use_worst_frame: bool = True  # 关键点头是否带「最差帧」通道
    # 分数头 -> 关键点头的梯度回流强度（1=全耦合，0=切断）。分数标签 alpha=0.17，
    # 关键点标签一致率 0.83，让前者改后者表征是在用脏水洗衣服。
    kp_grad_scale: float = 1.0
    use_kp_feats: bool = True  # 分数头是否消费关键点统计量（关掉=纯视觉回归）
    eval_train: bool = False  # 每 epoch 额外算训练集指标，用于诊断过拟合/欠拟合
    seed: int = 0
    num_workers: int = 4
    device: str = "cuda"
    amp: bool = True
    # auto: 有 bf16 就用 bf16（Ampere+），否则 fp16（Turing/T4）。
    # bf16 不需要 GradScaler，动态范围和 fp32 一致，混合精度下更稳。
    amp_dtype: str = "auto"  # auto | bf16 | fp16 | fp32
    weights: LossWeights = field(default_factory=LossWeights)


# ---------------------------------------------------------------- 工具


_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _resolve_amp_dtype(name: str, device: str) -> torch.dtype:
    """把 amp_dtype 选项解析成 torch.dtype。

    auto 的含义：能用 bf16 就用 bf16。T4（Turing, sm75）不支持 bf16，会退回 fp16；
    RTX 30/40 系（Ampere+）用 bf16 —— 它不需要 GradScaler，动态范围等同 fp32，
    混合精度训练更稳，不需要调 loss scale。
    """
    if name in _AMP_DTYPES:
        return _AMP_DTYPES[name]
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


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
def evaluate(model, loader, samples, indices, device, amp: bool = True, amp_dtype_name: str = "auto") -> dict:
    model.eval()
    preds, targets, kp_logits, kp_labels, kp_masks, order, view_types = [], [], [], [], [], [], []
    prob_s, prob_k = [], []

    for batch in loader:
        feat = batch["feat"].to(device, non_blocking=True)
        action_id = batch["action_id"].to(device, non_blocking=True)
        view_id = batch["view_id"].to(device, non_blocking=True)
        kp_mask = batch["kp_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=_resolve_amp_dtype(amp_dtype_name, device), enabled=amp):
            out = model(feat, action_id, view_id, kp_mask)

        preds.append(out.score.float().cpu().numpy())
        prob_s.append(torch.sigmoid(out.score_logits.float()).cpu().numpy())
        prob_k.append(torch.sigmoid(out.keypoint_logits.float()).cpu().numpy())
        kp_logits.append(out.keypoint_logits.float().cpu().numpy())
        targets.append(batch["score"].numpy())
        kp_labels.append(batch["keypoint"].numpy())
        kp_masks.append(batch["kp_mask"].numpy())
        order.append(batch["index"].numpy())
        view_types.extend("ego" if v else "exo" for v in batch["is_ego"].numpy().tolist())

    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    prob_s = np.concatenate(prob_s)   # (N, K-1) 累积概率
    prob_k = np.concatenate(prob_k)   # (N, n_kp)  逐关键点概率
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

    # 多视角融合：同一个物理动作有最多 6 路同步视角，标签完全相同。
    # 逐视角预测再平均 = 白送的方差缩减，且是这个数据集独有的结构（别的 AQA 数据集没有多视角）。
    # 注意：这是**推理期**融合，不改训练；论文只报单视角，所以我们两个都报。
    fused = None
    if len(order) > 0:
        groups: dict[str, list[int]] = {}
        for pos, gi in enumerate(order):
            groups.setdefault(samples[gi].action_key, []).append(pos)
        if any(len(v) > 1 for v in groups.values()):
            keys = [k for k in groups if len(groups[k]) > 1]
            order_f = np.array([groups[k][0] for k in keys])
            # CORAL 要先在累积概率上平均，再算期望分；不能直接平均最终分数
            ps = np.stack([prob_s[groups[k]].mean(0) for k in keys])
            pred_f = 1.0 + ps.sum(-1)
            pk = np.stack([prob_k[groups[k]].mean(0) for k in keys])
            # 平均后的概率转回 logit，复用同一套阈值/指标代码
            eps = 1e-6
            lk = np.log(np.clip(pk, eps, 1 - eps)) - np.log(np.clip(1 - pk, eps, 1 - eps))
            # 注意用拼接后的 labels/masks，不是收集用的 kp_labels/kp_masks 列表
            # （踩过：索引 list 时传入全局样本下标，直接越界 IndexError）
            yk = np.stack([labels[groups[k][0]] for k in keys])
            mk = np.stack([masks[groups[k][0]] for k in keys])
            vt = np.array([view_types[groups[k][0]] for k in keys])
            res["fused"] = {"n_actions": len(keys), "n_views_avg": float(np.mean([len(groups[k]) for k in keys]))}
            for name, sel in (("overall", np.ones(len(keys), bool)), ("ego", vt == "ego"), ("exo", vt == "exo")):
                if sel.sum() == 0: continue
                res["fused"][name] = {
                    "score": score_metrics(pred_f[sel], target[order_f][sel]),
                    "keypoint@0.5": keypoint_metrics(lk[sel], yk[sel], mk[sel], 0.5),
                }
            thr2, f1b = best_threshold(lk, yk, mk)
            res["fused"]["best_threshold"] = {"threshold": thr2, "f1": f1b}
            fused = {"pred": pred_f, "lk": lk, "yk": yk, "mk": mk}

    res["_raw"] = {"pred": pred, "target": target, "order": order, "view_types": view_types,
                   "fused": fused}
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
        "fused_srocc": res.get("fused", {}).get("overall", {}).get("score", {}).get("srocc", float("nan")),
        "fused_kp_f1": res.get("fused", {}).get("overall", {}).get("keypoint@0.5", {}).get("f1", float("nan")),
        "fused_kp_best": res.get("fused", {}).get("best_threshold", {}).get("f1", float("nan")),
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
    # 训练集侧的无增强副本：用来区分「过拟合」和「欠拟合」。
    # 只看验证集的话，SROCC=0.15 既可能是模型把训练集背下来了但对新人不泛化，
    # 也可能是两边都没学会。加上 train 指标一眼就能分开。
    train_eval_ds = AqaDataset(samples, features, bundle, split.train_idx, augment=False) if cfg.eval_train else None

    train_ld = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        collate_fn=collate, drop_last=len(train_ds) > cfg.batch_size, pin_memory=True, persistent_workers=cfg.num_workers > 0,
    )
    val_ld = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers,
        collate_fn=collate, pin_memory=True, persistent_workers=cfg.num_workers > 0,
    )
    train_eval_ld = (
        DataLoader(
            train_eval_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers,
            collate_fn=collate, pin_memory=True, persistent_workers=cfg.num_workers > 0,
        )
        if train_eval_ds is not None
        else None
    )

    model = AqaModel(
        num_actions=len(bundle.action_to_id),
        kp_text_emb=kp_text_emb,
        dim=cfg.dim, depth=cfg.depth, heads=cfg.heads, dropout=cfg.dropout,
        num_frames=bundle.num_frames, use_worst_frame=cfg.use_worst_frame,
        kp_grad_scale=cfg.kp_grad_scale, use_kp_feats=cfg.use_kp_feats,
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
    amp_dtype = _resolve_amp_dtype(cfg.amp_dtype, device)
    # 只有 fp16 需要 GradScaler 防梯度下溢；bf16 的指数位和 fp32 一样，不需要。
    # torch>=2.1 用 torch.amp.GradScaler("cuda")，老版本只有 torch.cuda.amp.GradScaler。
    need_scaler = use_amp and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=need_scaler)
    except (AttributeError, TypeError):  # pragma: no cover
        scaler = torch.cuda.amp.GradScaler(enabled=need_scaler)
    log(f"[fold {split.fold}] amp={use_amp} dtype={amp_dtype} scaler={need_scaler} device={device}")
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
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(feat, action_id, view_id, kp_mask)
                loss, parts = total_loss(out, batch, cfg.weights, pos_weight, cfg.focal_gamma, align_fn)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()  # 必须在 optimizer.step() 之后，否则 torch 会跳过 lr 调度的第一个值
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
        res = evaluate(model, val_ld, samples, split.val_idx, device, use_amp, cfg.amp_dtype)
        if ema is not None:
            ema.restore(model)
        s = summarize(res)

        # 训练集侧指标（可选）：诊断过拟合 vs 欠拟合
        tr = None
        if train_eval_ld is not None:
            if ema is not None:
                ema.apply_to(model)
            res_tr = evaluate(model, train_eval_ld, samples, split.train_idx, device, use_amp, cfg.amp_dtype)
            if ema is not None:
                ema.restore(model)
            tr = summarize(res_tr)

        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in avg.items()}, **s,
                        **({"tr_srocc": tr["srocc"], "tr_mae": tr["mae"], "tr_kp_f1": tr["kp_f1"]} if tr else {})})
        extra = f" | train SROCC={tr['srocc']:.4f} MAE={tr['mae']:.4f} KP F1={tr['kp_f1']:.4f}" if tr else ""
        log(f"[fold {split.fold}] ep {epoch + 1:3d}/{cfg.epochs} loss={avg.get('total', float('nan')):.4f} "
            f"(score={avg.get('score', float('nan')):.3f} kp={avg.get('keypoint', float('nan')):.3f} "
            f"act={avg.get('action', float('nan')):.3f} align={avg.get('align', float('nan')):.3f}) "
            f"| val SROCC={s['srocc']:.4f} MAE={s['mae']:.4f} KP F1={s['kp_f1']:.4f} (best {s['kp_f1_best']:.4f}){extra}")

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
    final = evaluate(model, val_ld, samples, split.val_idx, device, use_amp, cfg.amp_dtype)

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
