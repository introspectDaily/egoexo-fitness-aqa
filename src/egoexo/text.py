"""关键点文本 -> CLIP 文本嵌入（冻结，只算一次）。

为什么用 CLIP ViT-B/32 的文本编码器：视觉特征是 `clip_vit_b32_*`，
只有用**同一个** CLIP 模型的文本塔，文本和视觉才落在同一个联合嵌入空间里，
cross-attention 才有意义。换别的句向量模型会破坏这个对齐。

102 条唯一关键点句子，编码一次约 1 秒，之后从缓存读，训练完全不碰文本编码器。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

DEFAULT_CLIP_TEXT_MODEL = "openai/clip-vit-base-patch32"


def build_kp_text_table(
    kp_vocab: dict[str, list[str]],
    action_to_id: dict[str, int],
    out_dir: str | Path | None = None,
    model_name: str = DEFAULT_CLIP_TEXT_MODEL,
    device: str = "cpu",
    batch_size: int = 64,
) -> torch.Tensor:
    """返回 (num_actions, max_kp, text_dim) 的定长表；padding 位置填 0。

    表按 action_id 索引，训练时用 `action_id` 一次查表拿到该动作的全部关键点嵌入。
    """
    cache = Path(out_dir) if out_dir else None
    if cache is not None and (cache / "kp_text_emb.pt").exists():
        table = torch.load(cache / "kp_text_emb.pt", map_location="cpu", weights_only=True)
        cached = json.loads((cache / "kp_text_vocab.json").read_text(encoding="utf-8"))
        if cached == kp_vocab:
            print(f"[text] 命中缓存 {cache/'kp_text_emb.pt'}  shape={tuple(table.shape)}")
            return table

    # 保险：即使调用方忘了设，这里也确保走镜像（国内网络必需）
    from .secrets import apply_hf_endpoint

    apply_hf_endpoint()

    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    print(f"[text] 加载 {model_name} 的文本编码器（首次运行需下载 ~600MB）")
    tok = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device).eval()

    texts = [t for v in kp_vocab.values() for t in v]
    embs: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            out = model(**enc)
            embs.append(out.text_embeds.float().cpu())
    flat = torch.cat(embs, dim=0)  # (sum_n, D)
    text_dim = flat.size(-1)

    max_kp = max(len(v) for v in kp_vocab.values())
    table = torch.zeros(len(action_to_id), max_kp, text_dim)
    cursor = 0
    for name, _ in sorted(action_to_id.items(), key=lambda kv: kv[1]):
        kps = kp_vocab[name]
        table[action_to_id[name], : len(kps)] = flat[cursor : cursor + len(kps)]
        cursor += len(kps)

    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        torch.save(table, cache / "kp_text_emb.pt")
        (cache / "kp_text_vocab.json").write_text(json.dumps(kp_vocab, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[text] 已缓存 -> {cache/'kp_text_emb.pt'}")

    return table
