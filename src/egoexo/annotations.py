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
    ⚠️ k 的语义（913 条 IAJ 全量实测，别再用论文/注释里的说法）：
       `action_info` 每行是 `[class_id, st, ed]`，**第一列是类别 id，不是下标**。
       IAJ key 里的 k 是 action_info 的 **0-based 下标**（每个 record 都从 0 开始且连续）。
       st_ed_frame 与 action_info[k]：832 条完全相等，59 条是相邻同动作被合并成一个窗口，
       22 条是边界被标注者微调，**0 条**与 action_info[k-1] 对齐。
       旧注释写的 “k 是 1-based、对应 action_info[k-1]” 是错的，
       因为当时可能只有 st_ed_frame 路径跑通而没核对 fallback。

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
    # 去偏后的每位标注者分数（apply_annotator_debias 填充）；None 表示不去偏
    debiased: list[float] | None = None

    @property
    def key(self) -> str:
        return f"{self.record_id}_action_{self.action_idx}"

    @property
    def score(self) -> float:
        """标注者平均分。≥2 位标注者时是软标签，直接当回归目标会引入噪声。"""
        if self.debiased is not None:
            return sum(self.debiased) / len(self.debiased)
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
    debias_info: dict = field(default_factory=dict)

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

        # st_ed_frame 优先（实测 913 条**全部**自带，下面这条 fallback 目前永远不会触发，
        # 但保留正确的 0-based 写法，避免以后作者重打包数据时静默拿错动作窗口）
        st, ed = entry.get("st_ed_frame") or (None, None)
        if st is None:
            info = al[rid]["action_info"]
            if not (0 <= k < len(info)):
                skipped["bad_key"] += 1
                continue
            _, st, ed = info[k]

        # 关键点文本以第一位标注者为准（文本一致，只是 True/False 不同）
        kp_texts = [t for t, _ in anns[0]["key_point_verification"]]
        # 逐标注者收集其验证结果，长度不足时截断对齐
        # 存 int 而不是 bool：后面要拿它当下标建重合矩阵，而 numpy 里 `m[True, False]`
        # 是**布尔掩码索引**（会插新轴），不是取 (1,0) 元素 —— 静默算错。
        per_ann_kp = []
        for a in anns:
            kv = a["key_point_verification"]
            per_ann_kp.append([int(_parse_bool(v)) for _, v in kv[: len(kp_texts)]])

        # 用多数表决合成每个关键点的结论
        # 多数表决也基于 int，避免 True/False 混用
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


def load_dataset(raw_dir: str | Path, debias_scores: bool = False, debias_min_n: int = 30) -> Dataset:
    """加载数据集。

    debias_scores=True 时，先把每位标注者的分数减去其个人均值再加回全局均值。

    为什么需要这个：实测 26 位标注者的个人平均分从 **2.57 到 4.06**，跨度 1.5 分，
    且同一 record 内各标注者的宽严排序跨 record 一致（不是他们看了不同的动作，
    是真的个人尺度不同）。所以“标注者平均分”这个目标本身含一个系统偏移。
    去偏后逐对完全相等率 31%→41%，平均绝对差 0.90→0.73。

    debias_min_n 用来过滤标注量太少的标注者：他们的个人均值本身噪声巨大，
    减它只会引入更多噪声。不足 debias_min_n 条的标注者不去偏。
    """
    raw_dir = Path(raw_dir)
    records = load_records(raw_dir)
    actions = load_actions(raw_dir, records)

    debias_info: dict = {}
    if debias_scores:
        actions, debias_info = apply_annotator_debias(actions, min_n=debias_min_n)

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
    ds = Dataset(records=records, actions=actions, samples=samples)
    ds.debias_info = debias_info
    return ds


# ---------------------------------------------------------------- 标注者偏差校正


def annotator_bias_table(actions: dict[str, Action], min_n: int = 30) -> dict[str, dict]:
    """每位标注者的平均分、标准差、标注量。用来判断存不存在系统性宽/严。"""
    import statistics

    buckets: dict[str, list[int]] = {}
    for a in actions.values():
        for ann, s in zip(a.annotators, a.scores):
            buckets.setdefault(ann, []).append(s)

    table = {}
    for ann, vals in buckets.items():
        table[ann] = {
            "n": len(vals),
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "used": len(vals) >= min_n,
        }
    return table


def apply_annotator_debias(actions: dict[str, Action], min_n: int = 30) -> tuple[dict[str, Action], dict]:
    """每人减去自己的均值，再加回全局均值。返回新 actions 与统计信息。"""
    import statistics

    table = annotator_bias_table(actions, min_n=min_n)
    all_scores = [s for a in actions.values() for s in a.scores]
    grand = statistics.mean(all_scores)

    offset = {ann: info["mean"] - grand for ann, info in table.items() if info["used"]}

    for act in actions.values():
        raw = act.scores
        # 逐标注者减去其个人偏移；未参与去偏的标注者（样本太少）保留原分
        corrected = [s - offset.get(ann, 0.0) for ann, s in zip(act.annotators, raw)]
        act.debiased = corrected

    used_means = [table[a]["mean"] for a in offset]
    return actions, {
        "grand_mean": grand,
        "min_n": min_n,
        "n_annotators_total": len(table),
        "n_annotators_used": len(offset),
        "mean_range": [min(used_means), max(used_means)] if used_means else None,
        "offsets": offset,
        "table": table,
    }


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
