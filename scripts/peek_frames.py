#!/usr/bin/env python3
"""从 HF **流式**抽几帧出来肉眼看看 —— 不用下 67GB 帧包，也不用下 6.4GB 特征包。

为什么能这么快
--------------
`frames_open.tar.gz` 是一条**顺序** gzip 流，内部按
`<...>/<record_id>/<view>/frame_0000000001.jpg` 排列。所以我们只做
HTTP 流式读 + 顺序解压，抽到需要的帧就 **提前 break**：
分片 .ab 及之后的部分根本不会被请求，通常只下载几十 MB。

它同时回答三个还没实测过的问题：
  1. 帧到底长什么样（尺寸/朝向/有没有黑边）—— 直接看图
  2. 官方 `rotate_dict`（exo_l=90 / exo_r=270）把图转成什么样
  3. ⭐ `tensor[i]` 对应 `frame_{i+1}.jpg` 还是 `frame_{i}.jpg`
     `--verify-clip` 用官方 CLIP ViT-B/32 现算一帧特征，和 .pth 里的行比余弦相似度，
     一次定死 0/1-based，不用再猜。

用法（Colab）
------------
    from egoexo.secrets import hf_login; hf_login()
    !python scripts/peek_frames.py --out data/peek --rotate --verify-clip

默认 `--frame-idx auto`：自动挑一个**落在已标注动作内部**的帧，
免得你盯着"人还在走过去"的空档帧怀疑数据有问题。

本地不联网推演（只看标注侧，不下载）
    python scripts/peek_frames.py --dry-run --raw-dir ../data/egoexo/raw_annotations
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
from pathlib import Path

from PIL import Image

REPO_ID = "Lymann/EgoExo-Fitness"
FRAMES_TAR = "frames_open/frames_open.tar.gz.aa"
FEAT_TAR = "features_open/visual/EgoExo_Fitness_CLIP_Vid_Feat_w_Rotate.tar.gz.aa"

VIEWS = ["ego_l", "ego_m", "ego_r", "exo_l", "exo_m", "exo_r"]

# 来自 features/get_frame_features_CLIP.py —— 官方自己的旋转表。
# 注意：**帧包里存的是未旋转的原图**，旋转是抽 CLIP 特征时现场做的
#（所以特征包叫 ..._w_Rotate）。你用帧训 VideoMAE 时转不转，是必须自己拍板的事。
OFFICIAL_ROTATE = {"ego_m": 0, "ego_r": 0, "ego_l": 0, "exo_r": 270, "exo_l": 90, "exo_m": 0}

FPS = 30
MEMBER_RE = re.compile(
    r"(?:^|/)(?P<rid>[^/]+)/(?P<view>ego_l|ego_m|ego_r|exo_l|exo_m|exo_r)/frame_(?P<fid>\d+)\.jpg$"
)
FEAT_RE = re.compile(
    r"(?:^|/)(?P<rid>[^/]+)/(?P<view>ego_l|ego_m|ego_r|exo_l|exo_m|exo_r)/"
    r"clip_vit_b32_vid_frame_feat\.pth$"
)


# ---------------------------------------------------------------- 带预算的流


class BudgetReader(io.RawIOBase):
    """包住 HTTP 响应体：计数 + 超预算就炸。

    没有它，一旦 `--record` 指向 tar 很靠后的位置，脚本会安静地把 3GB 下完。
    宁可报错，也不要悄悄烧流量。
    """

    def __init__(self, raw, budget_bytes: int):
        self._raw = raw
        self.n = 0
        self.budget = budget_bytes

    def readable(self) -> bool:
        return True

    def read(self, size=-1):
        chunk = self._raw.read(size)
        if chunk:
            self.n += len(chunk)
            if self.budget and self.n > self.budget:
                raise SystemExit(
                    f"\n[peek] 已下载 {self.n / 1e6:.1f}MB，超过预算 {self.budget / 1e6:.0f}MB 仍未取全。\n"
                    f"       → 把 --frame-idx 调小 / 换更靠前的 --record / 调大 --budget-mb。"
                )
        return chunk or b""

    def readinto(self, b):
        chunk = self.read(len(b))
        n = len(chunk)
        b[:n] = chunk
        return n


def endpoint() -> str:
    """默认 huggingface.co；国内网络设 HF_ENDPOINT=https://hf-mirror.com 即可。

    hf-mirror 会转发 Authorization 头，所以 gated 仓库照样能过（已实测 200）。
    huggingface_hub / hf_hub_download 也认同一个环境变量。
    """
    import os

    return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


class _IterReader(io.RawIOBase):
    """把「分块迭代」的 HTTP 响应体包成 tarfile 能用的 read(n)。"""

    def __init__(self, it):
        self._it = it
        self._buf = b""

    def readable(self) -> bool:
        return True

    def read(self, size=-1):
        if size is None or size < 0:
            out = self._buf + b"".join(self._it)
            self._buf = b""
            return out
        while len(self._buf) < size:
            try:
                self._buf += next(self._it)
            except StopIteration:
                break
        out, self._buf = self._buf[:size], self._buf[size:]
        return out

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def _http_get(url: str, headers: dict, timeout: float = 120.0):
    """返回 (status, headers, body_reader, close)。

    优先用 **httpx**：huggingface_hub 1.x 自带它，而 1.x 已经不再依赖 requests
    （所以一个按官方 requirements 装好的环境里往往根本没有 requests）。
    没有 httpx 就退回 requests（Colab 的 hf_hub 0.x 环境）。两条路都不需要额外安装。
    """
    try:
        import httpx  # hf_hub >=1.0 的依赖

        client = httpx.Client(follow_redirects=True, timeout=timeout)
        req = client.build_request("GET", url, headers=headers)
        resp = client.send(req, stream=True)

        def close():
            resp.close()
            client.close()

        return resp.status_code, resp.headers, _IterReader(resp.iter_bytes()), close
    except ImportError:
        import requests

        r = requests.get(url, headers=headers, stream=True, timeout=timeout)

        def close():
            r.close()

        return r.status_code, r.headers, _IterReader(r.iter_content(65536)), close


def open_stream(path_in_repo: str, token: str | None, budget_mb: int, what: str):
    url = f"{endpoint()}/datasets/{REPO_ID}/resolve/main/{path_in_repo}"
    headers = {"Accept-Encoding": "identity"}  # 关键：别让 CDN 对 .gz 再套一层 gzip
    if token:
        headers["Authorization"] = f"Bearer {token}"
    status, hdrs, body, close = _http_get(url, headers)
    if status in (401, 403):
        close()
        raise SystemExit(
            f"\n[peek] {what} 拿不到（HTTP {status}）。排查顺序：\n"
            f"  1. 登录账号 == 在 {endpoint()}/datasets/{REPO_ID} 点过 Agree 的账号\n"
            f"  2. Fine-grained token 要额外勾上\n"
            f"     'Read access to contents of all public GATED repos you can access'\n"
            f"  3. token 要能被 huggingface_hub.get_token() 读到\n"
            f"     （默认位置 ~/.cache/huggingface/token，或设 HF_TOKEN）\n"
            f"  4. 失败时先手动验一条：\n"
            f"     curl -sIL -H \"Authorization: Bearer $TOKEN\" \n"
            f"       {endpoint()}/datasets/{REPO_ID}/resolve/main/raw_annotations/meta_records.json\n"
        )
    if status >= 400:
        close()
        raise SystemExit(f"[peek] {what} HTTP {status}")
    enc = (str(hdrs.get("Content-Encoding") or "identity")).strip().lower()
    if enc not in ("", "identity"):
        close()
        raise SystemExit(
            f"[peek] 服务器返回 Content-Encoding: {enc}，中间层会先解一层，"
            f"tarfile 收到已解压字节会报错。"
        )
    total = hdrs.get("Content-Length")
    extra = f"（该分片 {int(total) / 1e6:.0f}MB，但只按需读）" if total else ""
    print(f"[peek] {what}: 流式打开成功{extra}")

    class _Resp:
        def close(self):
            close()

    return BudgetReader(body, budget_mb * 1024 * 1024), _Resp()


# ---------------------------------------------------------------- 目标帧解析


def load_meta(raw_dir: Path) -> dict[str, dict[str, int]]:
    p = raw_dir / "meta_records.json"
    if not p.exists():
        return {}
    meta = json.loads(p.read_text())
    return {
        rid: {v: d["num_frames"] for v, d in meta["records"][pos]["frames"].items()}
        for rid, pos in meta["record_index"].items()
    }


def load_action_windows(raw_dir: Path) -> dict[str, list[tuple[int, int, int, str, bool]]]:
    """record -> [(k, st, ed, action_name, has_iaj)]，k 是 **0-based 下标**。

    ⚠️ 实测（913 条 IAJ 全量核对）：`action_info` 每行是 `[class_id, st, ed]`，
    第一列是**类别 id，不是下标**；IAJ 的 key `{rid}_action_{k}` 里的 k 才是
    `action_info` 的 0-based 下标。832/913 完全等于 `action_info[k]`，
    59 条是相邻同动作被合并成一个窗口，22 条是边界被标注者微调，
    **0 条**与 `action_info[k-1]` 对齐 —— 所以不存在 1-based 这回事。

    用 action_level（覆盖全部 1086 个动作）而不是只用 IAJ（913 个），
    这样几乎任何 record 都能挑出一个真动作的中间帧。
    """
    out: dict[str, list[tuple[int, int, int, str, bool]]] = {}
    al_path = raw_dir / "action_level_annotations.json"
    iaj_path = raw_dir / "interpretable_action_judgement.json"
    if not al_path.exists():
        return out
    al = json.loads(al_path.read_text())
    iaj: dict[str, dict] = json.loads(iaj_path.read_text()) if iaj_path.exists() else {}

    for rid, entry in al.items():
        rows = []
        for i, (cls_id, st, ed) in enumerate(entry.get("action_info", [])):
            info = iaj.get(f"{rid}_action_{i}")
            anns = (info or {}).get("annotations") or []
            name = anns[0].get("action_name") if anns else None
            rows.append((i, int(st), int(ed), name or f"class{cls_id}", info is not None))
        out[rid] = rows
    return out


def resolve_frame_idx(rid: str, spec: str, views: list[str], records, windows) -> list[int]:
    """把 --frame-idx（数字或 auto）解析成实际帧号列表，并夹进各视角有效范围。"""
    if spec != "auto":
        return [int(x) for x in spec.split(",") if x.strip()]

    rows = windows.get(rid, [])
    if not rows:
        nf = records.get(rid, {}).get(views[0], 900) if records else 900
        print(f"[peek] {rid} 没有 action_level 标注，退回取中段帧 {nf // 2}")
        return [nf // 2]

    # 选**最早的一个「像样的」动作**：既保证帧号一定在动作内部，
    # 又让 tar 流只需要前进一点点 —— 这是"看一眼"最省的选法。
    # 太短（<3s）的可能只是被切碎的一段，跳过。
    ok = [r for r in rows if r[2] - r[1] >= 3 * FPS] or rows
    ok.sort(key=lambda r: r[1])
    k, st, ed, name, has_iaj = ok[0]
    mid = (st + ed) // 2
    print(
        f"[peek] auto 选中最早的像样动作 action_{k}（= action_info[{k}]，0-based）[{name}] "
        f"st_ed=[{st},{ed}] 长 {ed - st + 1} 帧（{(ed - st + 1) / FPS:.1f}s）"
        f"{'，有 IAJ 标注' if has_iaj else '，无 IAJ（只看画面，没有关键点）'}"
    )
    return [mid]


def clamp(rid: str, want: list[int], views: list[str], records) -> dict[str, list[int]]:
    """每个视角实际可取哪些帧号（各视角总帧数有 ±10 帧差异）。"""
    out = {}
    for v in views:
        nf = (records.get(rid, {}) or {}).get(v)
        idx = [min(f, nf - 1) if nf else f for f in want]
        out[v] = idx
        if nf and idx != want:
            print(f"[peek]   {v}: 共 {nf} 帧 → 请求帧号夹到 {idx}")
    return out


# ---------------------------------------------------------------- 抽帧


def fetch_frames(token, record, spec, views, strip, budget_mb, records, windows):
    """返回 (record_id, {帧号: {view: jpg_bytes}}, 目标视角, 预算读数)。"""
    reader, resp = open_stream(FRAMES_TAR, token, budget_mb, "frames_open 分片 .aa")
    got: dict[int, dict[str, bytes]] = {}
    strip_buf: list[tuple[int, bytes]] = []
    rid_seen: str | None = None
    want: set[int] = set()
    target_views: list[str] = []
    strip_view: str | None = None
    n_members = 0
    passed: set[str] = set()
    strip_hi = 0

    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                g = MEMBER_RE.search(m.name)
                if not g:
                    continue
                rid, view, fid = g.group("rid"), g.group("view"), int(g.group("fid"))
                n_members += 1

                if rid_seen is None:
                    if record and rid != record:
                        continue  # 还没走到目标 record；这些字节仍会被下载，靠 budget 兜底
                    rid_seen = rid
                    idx = resolve_frame_idx(rid, spec, views, records, windows)
                    want = set(idx)
                    target_views = [v for v in views if not records.get(rid) or v in records[rid]]
                    if not target_views:
                        target_views = list(views)
                    strip_view = target_views[0]
                    strip_hi = max(want) + strip
                    print(f"[peek] tar 里第一个可用 record = {rid}；目标帧 {sorted(want)} / 视角 {target_views}")
                    print(
                        f"[peek] ⚠️ 一个 record 约 6 视角 × 数千帧 ≈ 0.5GB；"
                        f"tar 顺序未知，所以只取「第一个出现的 record」最省。"
                    )

                if rid != rid_seen:
                    break  # 目标 record 翻完了
                if view not in target_views:
                    continue

                if n_members % 500 == 0:
                    print(
                        f"\r[peek]   已扫 {n_members} 成员 / 已用 {reader.n / 1e6:6.1f}MB"
                        f" / 当前位置 {view} #{fid}",
                        end="",
                        flush=True,
                    )

                # ⚠️ 一个 member 只能 extractfile 一次：tar 是流，extractfile 之后再取
                # 同一个 member 会 seek 回退，直接 StreamError('seeking backwards is not allowed')。
                need_want = fid in want and view not in got.get(fid, {})
                need_strip = strip > 0 and view == strip_view and max(want) <= fid < strip_hi
                if need_want or need_strip:
                    f = tar.extractfile(m)
                    data = f.read() if f is not None else None
                    if data is not None:
                        if need_want:
                            got.setdefault(fid, {})[view] = data
                        if need_strip:
                            strip_buf.append((fid, data))
                if fid > max(want):
                    passed.add(view)  # 这个视角的目录已翻过目标帧

                # 该视角要么已抓齐全部目标帧，要么已经翻过目标帧号（目录里没有 / 视角缺失）
                views_done = all(
                    v in passed or all(v in got.get(f, {}) for f in want) for v in target_views
                )
                strip_done = strip == 0 or len(strip_buf) >= strip
                if views_done and strip_done:
                    print(f"\n[peek] 目标取全，提前中断 → 实际只下载 {reader.n / 1e6:.1f}MB")
                    break
    finally:
        resp.close()

    if rid_seen is None:
        raise SystemExit(f"[peek] {FRAMES_TAR} 里没找到 record={record!r}")
    if not any(got.values()) and not strip_buf:
        raise SystemExit("[peek] 没抓到任何帧：--views 名不对，或帧号全落在范围外")

    for fid, data in strip_buf:
        got.setdefault(fid, {}).setdefault(strip_view, data)
    return rid_seen, got, target_views, reader.n


# ---------------------------------------------------------------- 标注上下文


def annotation_context(raw_dir: Path, rid: str, fids: list[int]) -> None:
    """把「这几个帧号落在哪个动作里」打出来。没有标注目录就安静跳过。"""
    if not (raw_dir / "meta_records.json").exists():
        print(f"[peek] 没找到 {raw_dir}/meta_records.json，跳过标注上下文")
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    try:
        from egoexo.annotations import load_dataset  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"[peek] （没读到 egoexo 包，跳过标注上下文: {e}）")
        return

    ds = load_dataset(raw_dir, debias_scores=False)
    rec = ds.records.get(rid)
    if rec is None:
        print(f"[peek] 标注里没有 record={rid}")
        return
    acts = sorted((a for a in ds.actions.values() if a.record_id == rid), key=lambda a: a.st)

    print(f"\n[peek] 标注上下文：record {rid} / actor {rec.actor} / "
          f"{len(rec.views)} 视角 / {rec.num_actions} 个动作")
    print("       各视角帧数: " + ", ".join(f"{v}={rec.num_frames[v]}" for v in rec.views))
    for fid in fids:
        hit = [a for a in acts if a.st <= fid <= a.ed]
        print(f"       帧 {fid}（{fid / FPS:.2f}s）落在: " + (", ".join(
            f"action_{a.action_idx} [{a.action_name}] st_ed=[{a.st},{a.ed}] "
            f"分={a.score:.2f}({a.n_annotators}位)" for a in hit) or "**空档，没有标注动作**"))
        for a in hit:
            for kp in a.keypoints:
                print(f"          {'✓' if kp.satisfied else '✗'} {kp.text}")
            for c in a.comments:
                if c:
                    print(f"          评论: {c[:110]}")


# ---------------------------------------------------------------- 渲染


def render(rid, got, views, out_dir: Path, rotate: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    fids = sorted(got)
    saved: list[Path] = []
    pil_rows = []
    for v in views:
        row = []
        for f in fids:
            if v not in got[f]:
                continue
            data = got[f][v]
            p = out_dir / f"{v}_frame_{f:010d}.jpg"
            im = Image.open(io.BytesIO(data)).convert("RGB")
            if rotate:
                # ⚠️ 必须写**旋转后**的像素，不能直接 write_bytes(data) ——
                # 否则 rotated/ 目录里放的是原图，你对着它比半天也看不出差别。
                im = im.rotate(OFFICIAL_ROTATE.get(v, 0), expand=True)
                p = out_dir / f"{v}_frame_{f:010d}.jpg"
                if not p.exists():
                    im.save(p, quality=95)
                    saved.append(p)
            elif not p.exists():
                p.write_bytes(data)
                saved.append(p)
            row.append((f, im))
        if row:
            pil_rows.append((v, row))
    return pil_rows, saved


def montage(pil_rows, out_png: Path, title: str, cols: int = 6, cell_w: int = 456):
    """拼成一张总览图：行 = 视角，列 = 帧号。**只用 PIL**，不依赖 matplotlib。

    训练环境（尤其国内新机器）少一个依赖就少一次下载失败的可能。
    """
    from PIL import ImageDraw, ImageFont

    n_r = len(pil_rows)
    if not n_r:
        return None
    n_c = min(max(len(r) for _, r in pil_rows), cols)
    if not n_c:
        return None

    try:  # Pillow >= 10.1 的 load_default 支持 size，否则退回内置小字体
        font = ImageFont.load_default(size=15)
        font_title = ImageFont.load_default(size=19)
    except TypeError:
        font = font_title = ImageFont.load_default()

    gap, bar, head = 6, 20, 34
    cell_h = 0
    scaled = []
    for view, row in pil_rows:
        out_row = []
        for fid, im in row[:n_c]:
            w, h = im.size
            nh = max(1, round(h * cell_w / w))
            out_row.append((fid, im.resize((cell_w, nh), Image.LANCZOS)))
            cell_h = max(cell_h, nh)
        scaled.append((view, out_row))

    W = cols * cell_w + (cols + 1) * gap
    H = head + n_r * (bar + cell_h + gap) + 32
    canvas = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(canvas)
    d.text((gap, 8), title, fill=(240, 240, 240), font=font_title)

    for i, (view, row) in enumerate(scaled):
        y = head + i * (bar + cell_h + gap)
        for j in range(cols):
            x = gap + j * (cell_w + gap)
            if j >= len(row):
                continue
            fid, im = row[j]
            canvas.paste(im, (x, y + bar))
            d.text((x + 2, y + 1), f"{view}  #{fid}  {im.size[0]}x{im.size[1]}  {fid / FPS:.2f}s",
                   fill=(180, 230, 180), font=font)
    canvas.save(out_png)
    return out_png


def make_gif(pil_rows, out_gif: Path):
    if not pil_rows:
        return None
    row = pil_rows[0][1]
    if len(row) < 2:
        return None
    ims = [im for _, im in row]
    ims[0].save(out_gif, save_all=True, append_images=ims[1:], duration=int(1000 / FPS), loop=0)
    return out_gif


# ---------------------------------------------------------------- CLIP 校验


def load_clip_encoder(device):
    """优先 openai/clip（与官方抽特征脚本同一份代码）；没有就用 transformers 的同一权重。

    Colab 预装了 transformers，所以这条 fallback 通常不用额外装任何东西。
    两者共享同一个 512 维图像嵌入空间，同一张图的余弦相似度应 ≈ 1。
    """
    import torch

    try:
        import clip  # openai/clip

        model, preprocess = clip.load("ViT-B/32", device=device)

        def encode(im):
            with torch.no_grad():
                x = preprocess(im).unsqueeze(0).to(device)
                return model.encode_image(x).float().cpu().squeeze(0)

        return encode, "openai/clip ViT-B/32（与官方脚本完全一致）"
    except ImportError:
        pass

    from transformers import CLIPImageProcessor, CLIPModel  # Colab 预装

    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(name)

    def encode(im):
        px = proc(images=im, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            return model.get_image_features(pixel_values=px).float().cpu().squeeze(0)

    return encode, f"transformers {name}（HF 转换版，同一权重）"


def _torch_load(obj):
    import torch

    try:
        return torch.load(obj, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0 没有 weights_only
        return torch.load(obj, map_location="cpu")


def load_feat_tensor(feat_root: Path, rid: str, view: str, token, budget_mb: int):
    """先找**本地已解压**的特征（服务器上一般已经有了），找不到才流式从 HF 取。

    服务器上 data/features_open/ 通常是现成的 —— 这种情况下校验是零下载成本。
    """
    name = "clip_vit_b32_vid_frame_feat.pth"
    local = feat_root / rid / view / name
    if not local.exists():
        hits = sorted(feat_root.rglob(f"{rid}/{view}/{name}"))
        local = hits[0] if hits else None
    if local is not None and local.exists():
        print(f"[peek] 本地已有特征: {local}（{local.stat().st_size / 1e6:.1f}MB，不用下载）")
        return _torch_load(local)

    print(f"[peek] 本地 {feat_root} 里没有 {rid}/{view}，改为流式取")
    reader, resp = open_stream(FEAT_TAR, token, budget_mb, "features_open 分片 .aa")
    data = None
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                g = FEAT_RE.search(m.name)
                if not g or g.group("rid") != rid or g.group("view") != view:
                    continue
                f = tar.extractfile(m)
                if f is None:
                    continue
                print(f"[peek] 拿到 {rid}/{view} 特征: {m.name}（已用 {reader.n / 1e6:.1f}MB）")
                data = f.read()
                break
    finally:
        resp.close()
    return _torch_load(io.BytesIO(data)) if data is not None else None


def verify_clip(token, rid: str, view: str, fids: list[int], out_dir: Path, budget_mb: int,
                feat_root: Path):
    """现算一帧的特征，与 .pth 里的行比余弦相似度 —— 把 0/1-based 定死。"""
    import torch

    feat = load_feat_tensor(feat_root, rid, view, token, budget_mb)
    if feat is None:
        print(f"[peek] 没拿到 {rid}/{view} 的特征，跳过校验")
        return

    t = feat["clip_feat"] if isinstance(feat, dict) else feat
    print(f"[peek] clip_feat.shape = {tuple(t.shape)}  dtype={t.dtype}  （T 应等于该视角总帧数）")

    # 故意用 CPU：只有 1~2 张图，却能在**你的训练正占着 GPU** 时不碰显存、不抢算力
    encode, which = load_clip_encoder("cpu")
    deg = OFFICIAL_ROTATE.get(view, 0)
    print(f"[peek] 用 {which}；按官方口径 rotate {deg}° 现算，与 .pth 的行比余弦相似度：")
    for fid in fids:
        p = out_dir / f"{view}_frame_{fid:010d}.jpg"
        if not p.exists():
            cand = sorted(out_dir.glob(f"{view}_frame_*.jpg"))
            if not cand:
                print("[peek]   本地没有该视角 jpg，跳过")
                return
            p = cand[0]
        im = Image.open(p).convert("RGB").rotate(deg, expand=True)
        v = encode(im)
        sims = []
        for j in (fid - 1, fid, fid + 1):
            if 0 <= j < t.shape[0]:
                sims.append((j, float(torch.nn.functional.cosine_similarity(v, t[j].float(), dim=0))))
        best = max(sims, key=lambda x: x[1]) if sims else None
        line = " | ".join(f"row{j}={s:.4f}" for j, s in sims)
        print(f"   {p.name} → {line}" + (f"   ⇒ 命中 row{best[0]}" if best else ""))


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description="流式抽几帧 EgoExo-Fitness 的帧出来看")
    ap.add_argument("--out", default="data/peek")
    ap.add_argument("--record", default=None, help="指定 record_id；默认取 tar 里第一个（最省流量）")
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--frame-idx", default="auto", help="'auto' 或帧号（1-based，可逗号分隔）")
    ap.add_argument("--strip", type=int, default=8, help="额外抓多少连续帧做 GIF（0=不要）")
    ap.add_argument("--rotate", action="store_true", help="额外输出按官方 rotate_dict 旋转后的图")
    ap.add_argument("--verify-clip", action="store_true", help="现算 CLIP 特征校验行号↔帧号")
    ap.add_argument("--raw-dir", default="data/raw_annotations")
    ap.add_argument("--feat-root", default="data/features_open",
                    help="本地已解压的特征根目录；--verify-clip 优先读它，不用下载")
    ap.add_argument("--budget-mb", type=int, default=800)
    ap.add_argument("--dry-run", action="store_true", help="不联网，只打标注上下文与计划")
    args = ap.parse_args()

    out_dir = Path(args.out)
    raw_dir = Path(args.raw_dir)
    views = [v.strip() for v in args.views.split(",") if v.strip()]

    records = load_meta(raw_dir)
    windows = load_action_windows(raw_dir)
    print(f"[peek] 本地标注: {len(records)} 个 record"
          + (f" / {len(windows)} 个有动作边界" if windows else "（没找到 action_level json，先跑 download_data.py --annotations-only）"))

    if args.dry_run:
        rid = args.record or next(iter(records), "ThEnUZ")
        idx = resolve_frame_idx(rid, args.frame_idx, views, records, windows)
        print(f"[dry-run] record={rid} 目标帧 {idx}（各视角实际: {clamp(rid, idx, views, records)}）")
        print(f"[dry-run] 输出目录 {out_dir}/，成本 ≈ {len(idx) * len(views)} 张 × ~14KB")
        annotation_context(raw_dir, rid, idx)
        return

    from huggingface_hub import get_token

    token = get_token() or __import__("os").environ.get("HF_TOKEN")
    print(f"[peek] token: {'有' if token else '无（gated 数据会 401）'}")

    rid, got, target_views, nbytes = fetch_frames(
        token, args.record, args.frame_idx, views, args.strip, args.budget_mb, records, windows
    )
    fids = sorted(got)
    print(f"[peek] 取到 record={rid}，帧号 {fids}")

    for v in target_views:
        for f in fids:
            if v in got[f]:
                im = Image.open(io.BytesIO(got[f][v]))
                print(f"   {v}/frame_{f:010d}.jpg  {len(got[f][v]) / 1024:5.1f}KB  "
                      f"{im.size[0]}×{im.size[1]}  {im.mode}  {f / FPS:.2f}s")

    pil_rows, saved = render(rid, got, target_views, out_dir, rotate=False)
    out = [p for p in [montage(pil_rows, out_dir / "montage_raw.png",
                               f"{rid} — 原始朝向（frames_open 原样，未旋转）")] if p]
    if args.rotate:
        rows_r, saved_r = render(rid, got, target_views, out_dir / "rotated", rotate=True)
        saved += saved_r
        p = montage(rows_r, out_dir / "montage_rotated.png",
                    f"{rid} — 按官方 rotate_dict 旋转后（exo_l 90° / exo_r 270°）")
        if p:
            out.append(p)
    g = make_gif(pil_rows, out_dir / f"strip_{target_views[0]}.gif")
    if g:
        out.append(g)

    print(f"\n[peek] 单帧 {len(saved)} 张 → {out_dir}/")
    for p in out:
        print(f"[peek] {p}")
    print(f"[peek] 本次实际下载 ≈ {nbytes / 1e6:.1f}MB（对比：整包 67GB）")

    annotation_context(raw_dir, rid, fids)

    if args.verify_clip:
        print()
        verify_clip(token, rid, target_views[0], fids[:1], out_dir, args.budget_mb,
                    Path(args.feat_root))

    print(
        "\n[peek] 看图时确认这四件事：\n"
        "  1. ego_* 是头戴广角（有畸变），exo_l / exo_r 是侧面固定机位\n"
        "  2. exo 若是「躺着」的 → frames_open 确实未旋转，与官方抽特征脚本一致\n"
        "  3. 分辨率论文附录写的是 456×256；montage 标题上的数字若不同，说明 release 改过\n"
        "  4. GIF 是 30fps 连续帧 → 确认没有抽帧"
    )


if __name__ == "__main__":
    main()
