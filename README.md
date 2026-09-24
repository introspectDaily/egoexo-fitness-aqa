# EgoExo-Fitness 微调：动作质量分 + 可解释关键点验证

在 [EgoExo-Fitness](https://github.com/iSEE-Laboratory/EgoExo-Fitness)（ECCV 2024）上微调，
任务形态：**单动作视频 → 1~5 分质量分（有序回归） + 逐条技术关键点达标/不达标（多标签二分类）**。

对应论文里的两个 benchmark：
- **Cross-View Skill Determination** —— 「做得怎么样」（分数）
- **Guidance-based Execution Verification (GEV)** —— 「具体哪一条没做到」（关键点）

> 论文自己的 **最强 GEV 基线 F1 只有 0.5439**。所以这个任务远没被解决，
> 稳过 0.55 就是明确增量。别把目标定在 0.9。

---

## 1. 先看实测数据事实（和论文描述的差异，很重要）

这些数字是从发布的 4 个 json 里**实测**出来的，不是抄论文：

| 项 | 实测值 |
|---|---|
| record 数 | **76** |
| 唯一 `original_actor` | **76**（1 record = 1 人 → **按人划分 == 按 record 划分**） |
| 有 IAJ 标注的单动作 | **913** |
| 样本数（单动作 × 视角） | **5371** |
| 动作类别 | 12 |
| 唯一关键点句子 | 102（每动作 7~12 条） |
| 视角分布 | ego 2685 / exo 2686（均衡） |
| 分数均值 / 标准差 | 3.377 / 0.944 |
| **unsatisfies 占比** | **22.02%** → `pos_weight ≈ 3.54` |
| 单动作时长 | 中位 17s，p95 33s，max 59.5s |
| 标注者人数 | 1 位: 1994 / 2 位: 3227 / 3~5 位: 150 |

### ⚠️ 三个必须知道的坑

**(1) 官方 train/test 划分会泄漏。**
`subaction_level_annotations` 里的 `subset` 是**按 (record, view, sequence) 给的**，
68 个 record **全部**是 train/test 混排（纯 train 0 个、纯 test 0 个）。
直接套用去训 AQA，同一个物理动作的不同视角会同时出现在训练和验证集里。

→ 本仓库用 **GroupKFold(by `record_id`)**，这是唯一无泄漏的协议。每次训练前都有 `assert_no_leakage` 断言。

**(2) 论文说的「6131 个单动作」是展开视角之后的数量。**
`6131 = Σ(num_actions × num_views)`。**唯一物理动作只有 1086 个，有 IAJ 标注的只有 913 个。**
所以有效数据量比看起来小得多 —— 这是「必须用轻量模型 + 强正则」的根本原因。

**(3) 分数是 ≥2 位标注者的平均，是软标签。**
340 个动作只有 1 位标注者，其余 2~5 位。不同标注者对同一动作的分歧就是标签噪声。
本仓库：训练时给分数加高斯噪声（`--score-noise`，等价标签平滑），
并且提供了 **Krippendorff's α** 计算（`stats` 子命令）让你先量化标注一致性。

---

## 2. 方案：为什么是轻量模型而不是直接上 VideoMAE

三个约束决定了模型必须小：

1. **数据集只发布逐帧 CLIP ViT-B/32 特征**（`clip_vit_b32_vid_frame_feat.pth`），原始视频要邮件索取。
   论文自己的 GEVFormer 就是在冻结的 CLIP 特征上训一个轻量 Transformer（TCM + CMV）。
2. **有标注的物理动作只有 913 个**（× 视角 = 5371 条，但同 record 内高度相关，有效样本量还要打折）。
3. **跨视角泛化是崩塌的**：论文 Table 4 里，在 exo 上训、在 ego 上测，Top-1 从 0.93 掉到 **0.09**
   （12 类随机是 0.083）—— 等于完全失效。而且论文发现**简单混训 ego+exo 不涨点甚至掉点**，
   唯一有效的是显式的跨视角对齐损失。

所以本仓库的架构是：

```
冻结 CLIP-B/32 帧特征 (T, 512)
  → LayerNorm + Linear 降维到 256          ← 瓶颈，也是正则
  → 正弦位置编码
  → 3 层 TransformerEncoder (4 heads)
  → Attention Pooling                       ← 挑出"关键那一帧"，比 mean pooling 强
  → + view embedding (6 路视角)
  ├─ CoralHead      → 4 个累积 logits → 期望值 = 1 + Σσ(·)      【有序回归】
  ├─ KeypointHead   → 文本 query × 视觉 token 的 cross-attention 【GEV 核心】
  └─ action_head    → 12 类                                     【辅助正则】
  ＋ InfoNCE 跨视角对齐（同一物理动作的不同视角拉近，λ=0.7）
```

可训参数 **< 5M**。T4 上单折几分钟。

### 损失

```
L = 1.0·L_coral + 1.0·L_keypoint + 0.3·L_action + 0.7·L_align
```

- `L_coral`：**有序回归**，不用 MSE。1~5 分是有序的，MSE 隐含「1→2 的差距 == 4→5 的差距」，
  这在主观评分上不成立。
- `L_keypoint`：加权 BCE，`pos_weight≈3.54`。不加权的话模型退化成「全预测 satisfied」，
  能拿到 0.78 的 accuracy 但 **F1 = 0**。
- `L_align`：**λ=0.7 是论文的参数**，且论文证明它有效（ego F1 +0.0087 / exo 只掉 0.0022）。

---

## 3. Colab 运行（推荐路径）

### 3.1 前置：HF 权限

数据集是 gated 的。必须满足**两个条件缺一不可**：

1. 登录用的账号 = 在数据集页面点过 "Agree and access repository" 的那个账号
2. token 具备 gated 仓库的读权限

| token 类型 | 配置 | 能读 gated 吗 |
|---|---|---|
| Classic | 勾 `read` | ✅ 最省事 |
| Fine-grained | 只勾 `Read access to contents of all public repos you can access` | ❌ **401** |
| Fine-grained | **额外**勾上 `Read access to contents of all public GATED repos you can access` | ✅ |

> ⚠️ Fine-grained token 最常见的错误就是漏勾第二项。这两项在权限页面上**是分开的**，
> 报错表现为 `401 Unauthorized` / `Access to dataset is restricted`，看起来像「同意没生效」。

### 3.2 Colab Secret 配置

在 Colab 左侧 🔑 图标里添加两个 Secret，**名字必须完全一致**：

| Secret 名 | 用途 |
|---|---|
| `HUGGINFACE_ACCESS_KEY_COLAB_CLI` | 下数据集 |
| `GITHUB_ACCESS_KEY_FINE_GRAINED_FOR_COALB` | 只在需要 push 时用；仓库是 public 的话**不需要** |

**每一行右侧的「Notebook access」开关必须打开**（默认是关的，忘了开会报「Secret 不存在」）。

### 3.3 跑起来

打开 `colab/run.ipynb`，从上往下执行。或者手动：

```bash
git clone https://github.com/introspectDaily/egoexo-fitness-aqa.git
cd egoexo-fitness-aqa
pip install -q -r requirements.txt

# 0) 先自检：合成随机数据跑通全链路，不需要下载任何东西
python -m egoexo.cli smoke --epochs 3

# 1) 标注（6MB）
python scripts/download_data.py --annotations-only
python -m egoexo.cli stats --raw-dir data/raw_annotations

# 2) 特征（6.4GB，Colab 上约 5~10 分钟）
python scripts/download_data.py
python -m egoexo.cli probe --feat-root data/features_open

# 3) 预抽取（把每个动作的时间窗切成定长 32 帧，约 176MB，之后训练秒读）
python -m egoexo.cli extract --raw-dir data/raw_annotations \
    --feat-root data/features_open --out data/precomputed --num-frames 32

# 4) 5 折训练
python -m egoexo.cli train --precomputed data/precomputed --out runs/exp1 --folds 5 --epochs 40
```

`smoke` 那一步不要跳过：它用合成数据把「建表 → 前向 → 反向 → 评估 → 存档」全跑一遍。
过了它，后面出问题就一定是数据/环境问题，能省掉大量瞎猜。

---

## 4. 安全说明（凭证怎么用）

代码里的实现见 `src/egoexo/secrets.py`，原则四条：

1. **绝不打印明文** —— 所有日志走 `mask()`，只出前 6 位 + 长度。
   这一条在 notebook 里尤其重要：**cell 的输出会被写进 .ipynb 文件**，
   `print(token)` 等于把凭证存进了可能被分享的笔记本。
2. **绝不用命令行参数传凭证** —— `argv` 在 `ps` 里对同机所有进程可见，走环境变量或 Secret。
3. **绝不落盘到项目目录** —— 只用 `huggingface_hub.login()` 写到 `~/.cache/huggingface/token`
   （Colab VM 是一次性的，随 VM 销毁）。
4. **HF 登录用 `login()` 而不是设 `HF_TOKEN`** —— `login()` 不把凭证放进进程环境，
   子进程读不到；但 shell 里如果残留旧账号的 `HF_TOKEN`，它的优先级**高于**本地登录文件，
   排查 401 时先 `env | grep HF_TOKEN`。

**让 Colab 完全不碰 GitHub token 的做法**：把仓库设成 public，Colab 侧只 `git clone`，不需要任何 GitHub 凭证。
token 只在你本地（push 时）用。

---

## 5. 代码结构

```
src/egoexo/
  secrets.py      凭证安全读取（Colab Secret → 环境变量，全程掩码）
  annotations.py  解析 4 个 json → 扁平样本表（含关键点多数表决）
  features.py     .pth 发现 / 结构探测 / 定长帧预抽取
  splits.py       GroupKFold(by record_id) + 泄漏断言
  data.py         Dataset / collate
  models.py       时序编码器 + CORAL + 文本条件关键点头 + 跨视角 InfoNCE
  losses.py       有序回归损失 / 加权 BCE / Focal / 总损失
  metrics.py      SROCC / PLCC / MAE / F1 / Krippendorff's α
  text.py         关键点文本 → CLIP 文本嵌入（冻结，缓存）
  engine.py       训练与评估循环（T4 适配：fp16 + GradScaler）
  cli.py          stats / probe / extract / smoke / train
scripts/
  download_data.py
colab/
  run.ipynb
```

---

## 6. 评估口径

主指标 **SROCC**（AQA 领域标准，跨样本尺度免疫），同时报 PLCC / MAE / ±1 分准确率。

关键点 **F1，且正类 = "unsatisfies"** —— 这是论文 Table 7 的口径。
「satisfies」占 78%，报 accuracy 会让「全预测 satisfied」拿到 0.78 的假高分。

所有指标**按 ego / exo / 总体三列分别报**。跨视角是这个数据集的核心变量，
只报总体会掩盖「exo 0.9 / ego 0.3」这种塌陷。另外给出 per-action 拆解，看哪几类动作学不会。

阈值只在总体扫一次再套用到 ego/exo，避免各自调阈值带来的乐观偏差。

---

## 7. 待填：结果

| 配置 | SROCC | PLCC | MAE | KP F1@0.5 | KP F1@best | SROCC ego | SROCC exo |
|---|---|---|---|---|---|---|---|
| 论文 GEVFormer（仅 GEV，无分数） | — | — | — | — | **0.5439** | — | — |
| 本仓库 baseline | | | | | | | |

---

## 8. 引用

```bibtex
@inproceedings{li2024egoexo,
  title={EgoExo-Fitness: Towards Egocentric and Exocentric Full-Body Action Understanding},
  author={Li, Yuan-Ming and Huang, Wei-Jin and Wang, An-Lan and Zeng, Ling-An and Meng, Jing-Ke and Zheng, Wei-Shi},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```

数据集许可是「仅限非商业研究」。商用需单独谈授权。
