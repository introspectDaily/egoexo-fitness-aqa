"""把 EgoExo-Fitness 的 4 个原始 json 解析成扁平的样本表。

原始标注的实测结构（已逐字段核对，不要凭论文描述猜）：

meta_records.json
    {"records": [ {original_actor, views, frames:{view:{path,num_frames}},
                   num_views, num_sequences(str!), sequences:{sequence_start_end_frame},
                   num_actions}, ... ],
     "record_index": {record_id: 该 record 在 records 里的下标}}
    ⚠️ record 对象里**没有** record_id 字段，id 只能从 record_index 反查。

action_level_annotations.json
    {record_id: {num_actions, action_info: [[action_id, st_frame, ed_frame], ...]}}

interpretable_action_judgement.json   ← 本任务的主标注
    {"{record_id}_action_{k}": {annotations: [ {key_point_verification: [[text, "True"|"False"], ...],
                                                action_quality_score: 1..5,
                                                comment, action_name, action_guidance, annotator}, ... ],
                                st_ed_frame: [st, ed], frame_root: "frames_open/<rid>"}}
    ⚠️ k 是 **1-based**，对应 action_info[k-1]；实测 st_ed_frame 与 action_info 完全一致。

subaction_level_annotations_ant13_style_v1.json
    ActivityNet 风格，key = "{actor}_{seq}-{x}-{y}_{view}"，带 subset: train|test。
    实测 subset 是**按 (record, view, sequence) 给的**，同一个 record 内会 train/test 混在一起
    → 不能直接拿来做 AQA 的划分（会泄漏），只能当参考。

重要实测数字：
    76 个 record / 76 个唯一 original_actor（1:1，所以"按人划分" == "按 record 划分"）
    1086 个唯一单动作；6131 = sum(num_actions × num_views)，即**按视角展开后**的实例数
    913 个单动作有 IAJ 标注，共 1525 条标注（340 个只有 1 位标注者，548 个 2 位，最多 5 位）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- 常量

VIEW_ORDER = ["ego_l", "ego_m", "ego_r", "exo_l", "exo_m", "exo_r"]
EGO_VIEWS = {"ego_l", "ego_m", "ego_r"}
EXO_VIEWS = {"exo_l", "exo_m", "exo_r"}

SCORE_MIN, SCORE_MAX = 1, 5

_IAJ_KEY_RE = re.compile(r"^(?P<rid>.+)_action_(?P<k>\d+)$")


# ---------------------------------------------------------------- 数据结构


@dataclass(frozen=True)
class Record:
    """一个 record = 一名被试的一次录制，含最多 6 路同步视角。"""

    record_id: str
    actor: str
    views: tuple[str, ...]
    num_frames: dict[str, int]  # view -> 该视角总帧数
    num_actions: int

    @property
    def view_type(self) -> str:
        return "mixed"


@dataclass
class Keypoint:
    """一条技术关键点及其在该样本上的验证结果。"""

    text: str
    satisfied: bool

    @property
    def label(self) -> int:
        """论文口径：正类 = "unsatisfies"（少数类）。"""
        return 0 if self.satisfied else 1


@dataclass
class Action:
    """一个**单动作**（跨视角共享同一份标注）。"""

    record_id: str
    action_idx: int  # 1-based
    action_name: str
    st: int
    ed: int
    guidance: str
    keypoints: list[Keypoint]  # 取第一位标注者的关键点文本（文本各标注者一致，仅结果不同）
    scores: list[int]  # 每位标注者一个分
    annotators: list[str]
    comments: list[str | None]
    # [标注者][关键点位置] -> 0/1，原始未聚合的验证结果（算 α 用）
    per_annotator: list[list[int]] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.record_id}_action_{self.action_idx}"

    @property
    def score(self) -> float:
        """标注者平均分。≥2 位标注者时是软标签，直接当回归目标会引入噪声。"""
        return sum(self.scores) / len(self.scores)

    @property
    def n_annotators(self) -> int:
        return len(self.scores)

    @property
    def duration(self) -> int:
        return self.ed - self.st + 1

    @property
    def keypoint_labels(self) -> list[int]:
        """多数表决后的关键点标签（0=satisfied, 1=unsatisfies，正类为少数类）。"""
        return [k.label for k in self.keypoints]

    @property
    def keypoints_per_annotator(self) -> list[list[int]]:
        """[关键点位置][标注者] -> 0/1。用来算关键点的标注者间一致性（Krippendorff α）。

        注意这不是 :attr:`keypoints`：后者已经多数表决过了，用它算 α 恒等于 1。
        """
        n_ann = len(self.annotators)
        out: list[list[int]] = [[] for _ in self.keypoints]
        for a in range(n_ann):
            row = self.per_annotator[a] if a < len(self.per_annotator) else []
            for i in range(len(self.keypoints)):
                out[i].append(row[i] if i < len(row) else 0)
        return out


@dataclass
class Sample:
    """训练/评估的最小单元 = (单动作 × 视角)。"""

    sample_id: str
    record_id: str
    actor: str
    action_idx: int
    action_name: str
    view: str
    view_type: str  # "ego" | "exo"
    st: int
    ed: int
    num_frames_in_view: int
    score: float
    scores_raw: list[int]
    keypoints: list[Keypoint]
    guidance: str
    n_annotators: int

    @property
    def action_key(self) -> str:
        return f"{self.record_id}_action_{self.action_idx}"


@dataclass
class Dataset:
    records: dict[str, Record]
    actions: dict[str, Action]
    samples: list[Sample]
    action_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.action_names:
            self.action_names = sorted({a.action_name for a in self.actions.values()})


# ---------------------------------------------------------------- 解析


def _load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "t")


def load_records(raw_dir: Path) -> dict[str, Record]:
    meta = _load(raw_dir / "meta_records.json")
    index = meta["record_index"]
    records = meta["records"]
    out: dict[str, Record] = {}
    for rid, pos in index.items():
        r = records[pos]
        # 按固定顺序排，保证同一 record 内样本顺序稳定可复现
        views = tuple(v for v in VIEW_ORDER if v in r["views"])
        out[rid] = Record(
            record_id=rid,
            actor=r["original_actor"],
            views=views,
            num_frames={v: r["frames"][v]["num_frames"] for v in views},
            num_actions=int(r["num_actions"]),
        )
    return out


def load_actions(raw_dir: Path, records: dict[str, Record]) -> dict[str, Action]:
    """解析 IAJ。只有被 IAJ 覆盖的动作才会出现在结果里（实测 1086 中的 913 个）。"""
    iaj = _load(raw_dir / "interpretable_action_judgement.json")
    al = _load(raw_dir / "action_level_annotations.json")

    out: dict[str, Action] = {}
    skipped = {"no_meta": 0, "no_action_level": 0, "bad_key": 0, "no_annotations": 0}

    for key, entry in iaj.items():
        m = _IAJ_KEY_RE.match(key)
        if not m:
            skipped["bad_key"] += 1
            continue
        rid, k = m.group("rid"), int(m.group("k"))
        if rid not in records:
            skipped["no_meta"] += 1
            continue
        if rid not in al:
            skipped["no_action_level"] += 1
            continue

        anns = entry.get("annotations") or []
        if not anns:
            skipped["no_annotations"] += 1
            continue

        # st_ed_frame 优先；缺失时回落到 action_level（实测两者一致）
        st, ed = entry.get("st_ed_frame") or (None, None)
        if st is None:
            info = al[rid]["action_info"]
            if not (1 <= k <= len(info)):
                skipped["bad_key"] += 1
                continue
            _, st, ed = info[k - 1]

        # 关键点文本以第一位标注者为准（文本一致，只是 True/False 不同）
        kp_texts = [t for t, _ in anns[0]["key_point_verification"]]
        # 逐标注者收集其验证结果，长度不足时截断对齐
        per_ann_kp = []
        for a in anns:
            kv = a["key_point_verification"]
            per_ann_kp.append([_parse_bool(v) for _, v in kv[: len(kp_texts)]])

        # 用多数表决合成每个关键点的结论
        keypoints: list[Keypoint] = []
        for i, text in enumerate(kp_texts):
            votes = [pk[i] for pk in per_ann_kp if i < len(pk)]
            satisfied = (sum(votes) * 2 > len(votes)) if votes else True
            keypoints.append(Keypoint(text=text, satisfied=satisfied))

        # 保留逐标注者的原始 0/1，供一致性分析使用
        n_kp = len(kp_texts)
        per_annotator = [pk + [0] * (n_kp - len(pk)) if len(pk) < n_kp else pk[:n_kp] for pk in per_ann_kp]

        out[key] = Action(
            record_id=rid,
            action_idx=k,
            action_name=anns[0]["action_name"],
            st=int(st),
            ed=int(ed),
            guidance=anns[0].get("action_guidance") or "",
            keypoints=keypoints,
            scores=[int(a["action_quality_score"]) for a in anns],
            annotators=[a.get("annotator") or "?" for a in anns],
            comments=[a.get("comment") for a in anns],
            per_annotator=per_annotator,
        )

    if any(skipped.values()):
        print(f"[annotations] IAJ 跳过: {skipped}")
    return out


def load_dataset(raw_dir: str | Path) -> Dataset:
    raw_dir = Path(raw_dir)
    records = load_records(raw_dir)
    actions = load_actions(raw_dir, records)

    samples: list[Sample] = []
    for act in actions.values():
        rec = records[act.record_id]
        for view in rec.views:
            nf = rec.num_frames[view]
            # 边界裁剪到该视角实际帧数内（各视角帧数有 ±10 帧的差异）
            st = max(0, min(act.st, nf - 1))
            ed = max(st + 1, min(act.ed, nf))
            samples.append(
                Sample(
                    sample_id=f"{act.key}_{view}",
                    record_id=act.record_id,
                    actor=rec.actor,
                    action_idx=act.action_idx,
                    action_name=act.action_name,
                    view=view,
                    view_type="ego" if view in EGO_VIEWS else "exo",
                    st=st,
                    ed=ed,
                    num_frames_in_view=nf,
                    score=act.score,
                    scores_raw=list(act.scores),
                    keypoints=list(act.keypoints),
                    guidance=act.guidance,
                    n_annotators=act.n_annotators,
                )
            )

    samples.sort(key=lambda s: (s.record_id, s.action_idx, VIEW_ORDER.index(s.view)))
    return Dataset(records=records, actions=actions, samples=samples)


# ---------------------------------------------------------------- 关键点词表


def build_keypoint_vocab(ds: Dataset) -> dict[str, list[str]]:
    """action_name -> 该动作的关键点文本列表（顺序即标注顺序）。

    实测同一 action_name 的关键点条数恒定（7/8/9/12 条），文本也一致，
    所以可以用 action_name 索引一个定长词表，而不必每个样本各存一份。
    """
    vocab: dict[str, list[str]] = {}
    for act in ds.actions.values():
        texts = [k.text for k in act.keypoints]
        if act.action_name in vocab:
            if vocab[act.action_name] != texts:
                # 极少见：同名动作的关键点文本有出入。保留先出现的，并提示
                print(f"[annotations] ⚠️ '{act.action_name}' 关键点文本不一致，保留首次出现的版本")
        else:
            vocab[act.action_name] = texts
    return vocab


def load_official_subsets(raw_dir: str | Path) -> dict[str, str]:
    """从 subaction 标注里取官方 train/test 划分（key = record_id, view, seq 三元组）。

    ⚠️ 仅作参考：实测同一 record 内会 train/test 混排，直接拿来训 AQA 会泄漏。
    """
    sub = _load(Path(raw_dir) / "subaction_level_annotations_ant13_style_v1.json")
    out: dict[str, str] = {}
    for k, v in sub["database"].items():
        out[k] = v.get("subset", "?")
    return out


def action_subsets(raw_dir: str | Path) -> dict[str, set[str]]:
    """record_id -> 它出现过的 subset 集合。用来验证"官方划分是否按 record 隔离"。"""
    sub = load_official_subsets(raw_dir)
    out: dict[str, set[str]] = {}
    for k, s in sub.items():
        rid = k.split("_", 1)[0]
        out.setdefault(rid, set()).add(s)
    return out
