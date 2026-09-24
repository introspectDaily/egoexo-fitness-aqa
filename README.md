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

## 附：实测发现（这一节是跑通真实数据后加上的，都是可复现的数字）

### A. 分数标注噪声极大，且存在系统性标注者偏差

26 位标注者对同一动作打分：

| 指标 | 值 |
|---|---|
| Krippendorff's α（有序） | **0.1726** |
| 两两完全相等 | 31.1% |
| 两两差距 ≤1 分 | 81.4% |
| 两两平均绝对差 | 0.90 |
| 两位标注者间 Spearman | **0.457** |
| 标注者个人均分区间 | **2.57 ~ 4.06**（跨度 1.5 分 / 满分 5） |

**两个结论**：
1. **排序一致性（0.46）远好于绝对值一致性（0.17）** → 这从数据上证了「SROCC 是主指标、MAE 只能当参考」，也说明标注噪声是 SROCC 上限的主因。
2. **标注者不是按 record 块分配的**（中位覆盖 8 个 record，最多 58/76），且同 record 内各人的宽严排序跨 record 一致 → 存在真实的个人尺度差异，`--score-debias` 合法。
   去偏后：两两完全相等 31.1% → 39.8%，平均绝对差 0.901 → 0.753。

> 计算 α 时踩过一个坑：units 里传 bool 会让 `coincidence[True, False]` 变成
> numpy 的**布尔掩码索引**（插新轴返回副本），不报错、静默算成 nan。
> 已强制 int 化，并用纯 Python 另写一份实现交叉验证过（两边都是 0.1726）。

### B. 分数几乎完全由关键点决定 —— 这改变了架构选择

| 目标 | 可靠性 | CV SROCC（5 折 GroupKFold） |
|---|---|---|
| 关键点标签（多标注者两两一致率） | **83.3%** | — |
| 分数标签（α） | 0.17 | — |
| 用视频特征端到端直接回归分数 | — | **0.18** |
| **只用「不达标关键点比例」这一个特征回归分数** | — | **0.65** |

一个单特征线性模型打败神经网络端到端回归 3.6 倍。原因：标注流程本来就是
**先验关键点 → 写评论 → 给分**，分数是关键点的函数；而端到端回归既在跟 α=0.17
的噪声对抗，又要自己从零发现这个关系。

因此 `ScoreHead` 被改成以关键点概率为主要输入（`concat(视觉表征, [达标率, 不达标率, std, min_p, 关键点数占比])` → CORAL）。
副作用是分数变成「关键点结论的函数」，天然可解释。

### C. 瓶颈就是关键点预测精度，可以定量预测分数能到多少

在真值关键点上按比例随机翻转，模拟模型判错，再看推导出的分数 SROCC：

| 关键点翻转率 | 分数 SROCC |
|---|---|
| 0.00（真值） | 0.66 |
| 0.10 | 0.54 |
| 0.20 | 0.44 |
| 0.30 | 0.31 |
| 0.40 | 0.17 |

模型当前 prec≈0.42 / rec≈0.55 → 有效翻转率 ≈0.62 → **预测 SROCC ≈0.15**，
与实测 0.17 吻合。

**所以要 SROCC ≥ 0.5，关键点翻转率必须 ≤ 0.12（即 F1 ≈ 0.8）。**
现在 F1 0.48，差距全在这里。

### D. 关键点任务本身的地板与天花板

- 67 个关键点位置（12 个动作 × 7~12 条），**只有 6 个是退化的**（不达标率 <3% 或 >97%），55 个是「高信息」的（15%~85%）
- 「全预测多数类」的 F1 = **0.3265**（≈ 论文 Random 0.3178）
- 模型当前 F1 0.48~0.50 ≈ 论文 CLIP-GEV 朴素基线 0.488，**低于论文 GEVFormer 的 0.5439**
- 各动作的不达标率差异很大：Jumping Jacks 0.106（最容易）→ Sit-ups 0.327 / Side Knee Raise 0.310（最难）

### E. 由此推出的下一步

`clip_vit_b32_vid_frame_feat.pth` 的形状是 **(T, 512)** —— 是**全局池化**后的帧级特征，
7×7=49 个 patch token 已经被压成 1 个向量。而「膝盖内扣」这类错误本质是**空间**信息。

所以最可能的上限突破点是换视觉表征（用 `frames_open` 的原始帧跑 VideoMAEv2 / InternVideo2），
而不是继续调这个模型的超参。代价：帧包 ~67GB，Colab 磁盘（113G，现剩 ~56G）需要先腾空间
并改成流式解压。

---

## 附二：局域网 3080 机器 + 国内网络（另一套可用环境）

在 RTX 3080 20G / 32 核 / 国内网络上跑通的那套配置。和 Colab 的差异主要是三处：
**走 hf-mirror、复用 uv 缓存的 torch、用 bf16 而不是 fp16**。

### 1. 关键环境事实（实测）

| 项 | 值 |
|---|---|
| GPU | RTX 3080 20480 MiB，compute cap **8.6**（Ampere）|
| bf16 | **支持**（`torch.cuda.is_bf16_supported() == True`）|
| uv | 0.11.17，缓存 44GB，其中 `archive-v0` 37GB |
| ComfyUI venv | `/home/doit/.venv/comfyui`，Python 3.11.15，`torch 2.12.0+cu130` |
| `github.com` | **不可达** |
| `codeload.github.com` | 可达（要拉代码走 tarball，或直接局域网 rsync）|
| `huggingface.co` | **不可达** |
| `hf-mirror.com` | 可达，且**能代理 gated 数据集**（带 token 时 API 200、resolve 200）|
| `download.pytorch.org` | 可达 |
| `pypi.org` | 不可达；用 `mirrors.aliyun.com` |

### 2. 建环境：同 Python/torch 版本，让 uv 命中缓存

**要点：Python 版本和 torch 版本与 ComfyUI 完全一致**（都是 3.11 / 2.12.0+cu130）。
uv 的缓存按「包+版本+Python ABI」索引，版本对齐后整个安装过程**零下载**（实测 3 秒）。

```bash
cd ~/Desktop/egoexo-fitness-aqa
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/

uv venv .venv --python 3.11

# torch 必须显式给 pytorch 的 cu130 源，否则会从默认源解析成别的 CUDA build
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
  "torch==2.12.0" "torchvision==0.27.0"

uv pip install --python .venv/bin/python \
  "numpy==2.4.6" "scipy==1.17.1" "transformers==5.9.0" "huggingface-hub==1.17.0"

# --no-deps：我们自己的包是纯 Python，不需要动上面任何一个依赖
uv pip install --python .venv/bin/python --no-deps -e .
```

> 版本号从 ComfyUI 的 venv 里读出来对齐就行：
> `uv pip list --python /home/doit/.venv/comfyui/bin/python | grep -E "^(torch|numpy|scipy|transformers)"`
>
> ⚠️ uv 的 `--offline` **不能**用来复用缓存：离线时它连索引元数据都读不到，会直接报
> `torch was not found in the cache`。必须联网解析一次，uv 才会去复用缓存的 wheel 归档。

### 3. hf-mirror + 凭证

huggingface.co 在国内不可达，而 `huggingface_hub` 只认 `HF_ENDPOINT` 环境变量。
代码里已把这件事做掉（`secrets.apply_hf_endpoint()`），在**任何** HF 调用之前生效
—— 否则报的是 DNS/连接超时这类误导性错误，很难定位到「其实是镜像没设」。

```bash
umask 077
cat > ~/.hf_env <<'EOF'
export HUGGINFACE_ACCESS_KEY_COLAB_CLI=hf_xxxx
export HF_ENDPOINT=https://hf-mirror.com
EOF
chmod 600 ~/.hf_env
```

`secrets.py` 的取值顺序：**Colab Secret → `~/.hf_env` → 环境变量**。
`~/.hf_env` 强制 `0600`，权限不对直接拒读（一个全局可读的凭证文件等于没设防）。

### 4. bf16

T4（Turing, sm75）不支持 bf16，所以最早的代码写死 fp16 + GradScaler。
3080 是 Ampere，**bf16 更合适**：指数位与 fp32 相同 → 不会梯度下溢 → **不需要 GradScaler**，
也不用调 loss scale。代码现在自动判断：

```bash
--amp-dtype auto   # 默认。支持 bf16 就用 bf16，否则 fp16
```

每个 fold 开头会打印实际用的 dtype，避免「以为在用 bf16」：

```
[fold 0] amp=True dtype=torch.bfloat16 scaler=False device=cuda
```

### 5. 并行扫参

实测这个任务 **GPU 利用率只有 15%**（模型 <5M 参数，输入是预抽取好的 176MB 特征矩阵，
瓶颈在 Python/DataLoader，不在算力）。20GB 显存跑 8 个配置绰绰有余。

`--fold N --folds 5` 支持只跑第 N 折，所以可以把不同配置的**同一折**并行跑，
验证集完全相同，结果可直接横比：

```bash
for name in base no_worst crop10 f64 big debias noalign bigbatch; do
  python -m egoexo.cli train --out runs/par/$name \
    --folds 5 --fold 0 --epochs 40 --amp-dtype bf16 --num-workers 3 &
done
wait
```

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
