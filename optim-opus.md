# 在 VHR 之上的可发表优化方案（科研论文版）

本文面向「在 VHR 基础上做出可发表创新点」的目标，给出 **6 条具有潜在贡献度** 的优化方法。每条都包含：**动机 → 方法描述 → 形式化（必要时）→ 在本仓库的实现入口 → 实验设计 → 预期贡献与风险**。

设计准则：

1. **机制可解释**：每条方法都对应 VHR 在三类模型上观察到的一个具体失败模式（见 `study.md`）。
2. **改动可控**：尽量在 `vhr.py` / `generation.py` / `main.py` 内部完成，不依赖额外训练。
3. **指标可比**：统一报 **CHAIR（CHAIRs/CHAIRi/Len）+ POPE（Acc/F1/Yes%）+ LLaVA-Bench-W**，并提供消融。

---

## 方法 1：自适应头加权 **A-VHR**（Adaptive-ratio VHR）

### 动机

原 VHR 用固定 `aug_ratio=2` 与 **head_vary > median** 的硬阈值挑头：

- 高层语义化更强的 head 被等比例放大，易引发**列举型/语言风格型**幻觉（与 InstructBLIP 长输出现象一致）。
- 中位数阈值在不同输入下选中头数恒为一半，**与样本不确定性无关**。

### 方法

将 `aug_ratio` 从超参变成**与 head_vary 分布相关的连续函数**，并按层引入**深度衰减系数** \(\gamma_l\)：

\[
r_{l,h}(x)=1+\big(\rho_{\max}-1\big)\cdot \gamma_l\cdot \sigma\!\Big(\tfrac{v_{l,h}-\mu_l}{\tau\,s_l}\Big)
\]

- \(v_{l,h}\)：第 \(l\) 层第 \(h\) 个 head 的 head_vary（与 `vhr.py` 中 `head_vary` 一致）。
- \(\mu_l, s_l\)：层内均值与标准差；\(\sigma\) 为 sigmoid，温度 \(\tau\) 控陡峭度。
- \(\gamma_l = \exp(-\alpha\,(L-l)/L)\)：**越靠近输出层，加权越保守**（抑制语义层放大）。
- \(\rho_{\max}\) 为可调上限（默认 \(2.0\)）。

### 实现入口

`vhr.py::vhr_forward` 中将硬阈值替换为上述连续函数：

```python
v = head_vary
mu, s = v.mean(), v.std() + 1e-6
gate = torch.sigmoid((v - mu) / (tau * s))
ratio = 1 + (rho_max - 1) * gamma_l * gate
attn_output[:, :, :, :] = ratio.view(1, -1, 1, 1) * attn_output
```

`replace_attn` 中传入 `gamma_l` 由 `layer_idx` 计算。

### 实验

- **数据**：CHAIR、POPE 三分割、LLaVA-Bench-W。
- **消融**：仅 \(\gamma_l\)；仅连续 gate；二者联合。
- **变量**：\(\tau \in \{0.5,1,2\}\)；\(\rho_{\max}\in\{1.5,2,3\}\)；\(\alpha\in\{0,0.5,1\}\)。

### 预期贡献与风险

- **贡献**：把 VHR 从“两段式硬阈值”推广为**可解释的连续 head 重加权**；解释“层越深越保守”这一行为。
- **风险**：超参面变大；在 LLaVA-1.5 上若 gate 过弱反而退化为 baseline。

---

## 方法 2：**Entropy-Gated VHR (EG-VHR)** — 仅在「犹豫步」生效

### 动机

VHR 在每个解码步都重加权，但**幻觉常出现在不确定的 token**（罕见物体、模糊关系）。在确定性高的步上额外动注意力，更易制造**风格类副作用**（重复、列举）。

### 方法

设第 \(t\) 步 next-token 分布的熵为 \(H_t = -\sum_i p_i \log p_i\)。  
定义门控：

\[
g_t = \mathbb{1}\!\left[H_t \ge q_{\rho}(H_{1:t-1})\right]
\]

\(q_\rho\) 为前 \(t-1\) 步熵的滑动 \(\rho\)-分位数。**仅当 \(g_t=1\) 才施加 VHR**；否则使用原始 attention。

### 实现入口

`generation.py::_vhr_decoding` 循环里：

- 在调用前向之前用一份**轻量的最近 logits 缓存**估计熵；
- 通过模块级标志位（`self_attn._vhr_active`）控制 `vhr_forward` 内是否乘 ratio。

### 实验

- **指标**：CHAIR / POPE / 长度。
- **可视化**：把每条样本中实际激活 VHR 的步占比与该样本是否含幻觉做相关性分析（贡献点）。

### 贡献与风险

- **贡献**：将“干预何时发生”作为新维度，提供**步级可解释性**。
- **风险**：滑动分位数对短序列不稳定；需要 warm-up 步策略。

---

## 方法 3：**Q-Former-Aware VHR (QFA-VHR)** — 修复 InstructBLIP 对比路径

### 动机

`generation.py::_vhr_decoding` 中 InstructBLIP 的对比路径采用 **`inputs_embeds[:, 32:]`** 硬截断（与 `num_query_tokens=32` 对齐）。这对当前 BLIP2-7B 可用，但：

- 不同 BLIP 变体 query 数不同（如 InstructBLIP-Flan-T5 系列、视频版）。
- 单一对比难以反映 **Q-Former 的视觉信息含量**——只是“去掉前 32 个 embedding”，并未真正模拟“弱视觉”。

### 方法

设计 **多尺度对比池**：

1. **Empty-Q**：用零向量替换 query embedding（彻底去视觉）。
2. **Noisy-Q**：对 `pixel_values` 加扩散噪声（与 VCD 同步），再过 Q-Former 得到 **弱视觉 query**。
3. **Permuted-Q**：随机打乱 32 个 query 顺序（破坏空间对应而保留风格）。

对每条对比构造分别计算 head_vary，再做**层内最大值** \(v_{l,h}=\max_k v_{l,h}^{(k)}\)，识别在多种弱化下都「显著变化」的视觉敏感 head。

### 实现入口

新增 `vhr_blip.py`，在 `_vhr_decoding` 内当 `image_start<=0` 时按 `model_kwargs_tc` 列表跑多次小前向，再合并 `head_vary`。

### 实验

- **主实验**：仅在 InstructBLIP 上对比 baseline VHR vs QFA-VHR。
- **关键消融**：单 Empty-Q、单 Noisy-Q、单 Permuted-Q、两两组合、三者联合。
- **预期**：CHAIRs 下降 + Len 不暴涨。

### 贡献与风险

- **贡献**：首个针对 **Q-Former 类 LVLM** 的对比构造改进，明确把 VHR 推广到非 LLaVA 架构。
- **风险**：每步多前向带来 \(\sim 2\!-\!3\times\) 推理代价；可设“仅初始几步多对比”降低开销。

---

## 方法 4：**Visual Consistency Re-ranking (VCR-VHR)** — 生成–验证两阶段

### 动机

VHR 是 **生成期** 的注意力干预，但仍可能产出与图不符的物体名词。**生成后**对“被提到物体是否真实存在”做轻量验证，能捕获仅靠注意力调节难以避免的幻觉。

### 方法

两阶段：

1. **Stage-1 生成**：在 \(N\) 个候选中（VHR + 不同 \(\tau, \alpha, \rho\)、或 sampling）产生候选 captions。
2. **Stage-2 验证**：对每个候选抽取**名词短语**，用本模型自身在 **POPE 风格 yes/no 提示**上做 0-shot 校验：
   \[
   \text{score}(c)=\sum_{n\in N(c)} \log P(\text{“yes”}\mid \text{“Is there a } n\text{?”}, x)
   \]
   选 score 最高的候选作为输出（必要时附长度归一）。

### 实现入口

新增 `verifier.py`，在 `main.py` 推理循环里把单次 `generate` 改为「\(N\)-best + verify」。

### 实验

- **指标**：CHAIR、POPE、Len 全报；与 self-consistency baseline 对比。
- **预期**：CHAIRi 显著下降；CHAIRs 下降；Len 与单次生成相当。

### 贡献与风险

- **贡献**：**生成期 VHR + 输出期视觉一致性** 的复合框架，直接面向「物体级幻觉」。
- **风险**：推理代价随 \(N\) 线性增长；需在论文里报告推理成本曲线。

---

## 方法 5：**Object-Constrained Decoding (OCD-VHR)** — 把视觉证据嵌入解码

### 动机

CHAIR 的「物体级幻觉」本质是 **生成不在图中的实体**。如果在解码时对 **可能不存在的物体名词** 施加软惩罚，可与 VHR 的 head 调节互补。

### 方法

1. **物体词表 \(\mathcal{O}\)**：取 CHAIR 同源的 **MSCOCO 80 类 + WordNet 同义词**（不需额外标注）。
2. **存在性概率 \(p(o\mid x)\)**：用一次轻量「Is there a \(o\) in this image?」前向得到（也可对图像级 CLIP/SigLIP zero-shot）。
3. **解码惩罚**：当待选 token 解码后形成的子词串属于 \(o\in\mathcal{O}\) 时，将其 logit 修正为
   \[
   \tilde{\ell}_o = \ell_o - \lambda(1 - p(o\mid x))
   \]
   仅对 \(\mathcal{O}\) 中物体生效，对其他词不变。

### 实现入口

`generation.py::_vhr_decoding` 的 `next_token_scores = logits_processor(...)` 之后插入一个新的 `LogitsProcessor`：`ObjectExistencePenalty`。

### 实验

- **数据**：CHAIR（强相关）、POPE（间接相关）。
- **消融**：仅 OCD（无 VHR）、仅 VHR、OCD+VHR；\(\lambda\in\{1,2,5\}\)。
- **预期**：CHAIRi/CHAIRs 同时下降，POPE 不退步。

### 贡献与风险

- **贡献**：以**物体级显式约束**与**头级隐式约束**正交结合，**无需训练**。
- **风险**：词表外幻觉无法覆盖；验证前向 \(\mathcal{O}(|\mathcal{O}|)\) 次代价较高（可只对图像 caption 中**已出现**的候选物体校验）。

---

## 方法 6：**Cross-Layer Head Routing (CLHR)** — 把 VHR 与 DoLa/EAH 在层维度上融合

### 动机

VHR 在「**最后 \(k\) 层**」一刀切；DoLa 在「**早退层 vs 末层**」对 logits 做对比；EAH 在 **特定层** 对图像区注意力施加惩罚。三者对**层**的看法不同。

### 方法

将层划分为 **三个角色**：

| 段 | 范围 | 干预 |
|----|------|------|
| **L-low** | \(\,[\,0, l_1\,]\) | EAH 风格的图像区软惩罚 |
| **L-mid** | \([\,l_1, l_2\,]\) | DoLa 对比层（与末层对比） |
| **L-high** | \([\,l_2, L\,]\) | A-VHR / EG-VHR |

\(l_1, l_2\) 通过 **少量验证集网格搜索** 选取（保持训练-free 属性）。

### 实现入口

- `vhr.py::replace_attn` 增加「按段配置」接口；
- `generation.py` 在解码循环中根据层段切换分支；
- 需小心 KV cache 与 attn 实现的兼容（EAH 需 `eager`）。

### 实验

- **基线**：单方法 vs 段路由。
- **可视化**：层段对最终 CHAIRs 的边际贡献。

### 贡献与风险

- **贡献**：**层角色分工** 的统一视角，把已有方法纳入一个可调框架，是论文「方法图」级的贡献点。
- **风险**：实现复杂度最高；需保证不同 attn 实现共存。

---

## 方法对比表

| 方法 | 主要解决问题 | 与 VHR 关系 | 训练-free | 工程复杂度 | 预期收益面 |
|------|--------------|-------------|-----------|------------|------------|
| **A-VHR** | 头放大粗粒度 | 替换权重函数 | 是 | 低 | 三模型通用 |
| **EG-VHR** | 干预步与不确定性脱钩 | 增门控 | 是 | 中 | 通用，对 LLaVA-1.6 长输出尤其 |
| **QFA-VHR** | InstructBLIP 对比路径不准 | 改对比构造 | 是 | 中 | InstructBLIP 专用 |
| **VCR-VHR** | 输出后未验证 | 串联生成+校验 | 是 | 中 | 物体幻觉显著 |
| **OCD-VHR** | 物体词缺乏显式约束 | 加 LogitsProcessor | 是 | 中 | 物体级直接 |
| **CLHR** | 层角色未分工 | 包络 VHR/DoLa/EAH | 是 | 高 | 方法学贡献 |

---

## 推荐路径

如果以**最小工作量获得论文主贡献**为目标，建议：

1. **主创新**：选 **A-VHR + EG-VHR** 作为论文核心方法，机制相邻、可写在一节里，互为消融。
2. **架构泛化**：补 **QFA-VHR**，证明方法在 **Q-Former 类** 模型上同样有效，扩展实验覆盖面。
3. **正交结合**：补 **OCD** 或 **VCR** 一项，作为 **「正交模块也可叠加」** 的额外结果，强化贡献深度。
4. **方法学叙事**：在 Related Work / Analysis 中引入 **CLHR 的层角色视角**，把 VHR/DoLa/EAH 串起来作为分析框架，即使不实现也能让叙事更完整。

---

## 评测与报告要点

- **统一三模型**：LLaVA-1.5、LLaVA-1.6、InstructBLIP；POPE 三分割 + CHAIR 5-seed + LLaVA-Bench-W。
- **报告 Len 与 Yes%**：避免“压幻觉但更短/更否”的伪改进。
- **推理代价**：报告每条 caption 的平均前向次数与 wall-clock，便于审稿人评估实用性。
- **置信区间**：所有数值给出均值 ± 标准差（CHAIR 已自然有 5 seed，POPE 可对三分割报均值/标准差）。
- **可复现**：固定模型 revision、随机种子、`transformers` 版本，于附录给出全部超参。

---

## 与本仓库的对接清单

- `vhr.py`：实现 A-VHR、QFA-VHR 的对比聚合接口、CLHR 的层段开关。
- `generation.py::_vhr_decoding`：实现 EG-VHR 的步级门控、OCD 的 LogitsProcessor 接入、QFA 的多对比前向。
- `main.py`：新增 `--method` 标签 `avhr` / `egvhr` / `qfa-vhr` / `ocd` / `vcr`；`--vhr_aug_max`、`--vhr_tau`、`--vhr_alpha`、`--vhr_topq`、`--ocd_lambda`、`--vcr_n` 等超参。
- `eval/eval.py`：当前已支持 CHAIR / POPE；如做 VCR/OCD，需在 `main.py` 输出端记录候选数与代价信息，便于成本-收益曲线。

---

## 小结

- **A-VHR** 与 **EG-VHR** 在 VHR 内部把「**多少**」与「**何时**」加权解耦，构成最具可解释性的核心创新点；
- **QFA-VHR** 把方法从 LLaVA 泛化到 Q-Former 类模型，回答了 `study.md` 中观察到的 InstructBLIP 失效；
- **OCD / VCR** 提供与 VHR **正交**的输出端约束，使整套方法在 CHAIR 这类「物体级」指标上有清晰收益；
- **CLHR** 提供方法学叙事，让论文的 Related Work 与 Analysis 部分形成闭环。

按此清单组合，足以支撑一篇以 **「无训练、可插拔、跨架构」** 为卖点的 LVLM 抗幻觉方法论文。

---

# 附录：实验代码版本管理与回滚方案

针对**单 `main` 分支、一堆未提交改动 + 大文件未忽略**的现状，给一个**适合 ML 研究迭代**的分层方案。从「最低成本」到「最完整」分四档，按需选用即可。

## 第一步：先把 baseline 锁死（必做，10 分钟）

现在最大的风险是**当前 working tree 既有有用代码改动，又混着大量数据/模型/日志**——一旦做新实验就回不来。

### 1) 加 `.gitignore`，把数据/模型/日志/缓存挡掉

仓库里还没有 `.gitignore`，建议创建如下（**可直接采用**）：

```gitignore
# data / models / results / logs - 不应进 git
data/
models/
results/
POPE/
val2014/
train2014/
*.zip
*.zip.*
__pycache__/
*.log
logs/
.cache/
.ipynb_checkpoints/

# editor
.vscode/
.idea/
*.swp
.DS_Store
```

### 2) 把当前**有意义的代码改动**整体提交为 baseline

按 `git status`，真正需要进 git 的只有：

- `eval/eval.py`、`main.py`、`scripts/cat_files.sh`、`scripts/infer_instructblip.sh`（已修复）
- 新增：`scripts/prepare_data.sh`、`fuxian.md`、`study.md`、`optim.md`、`optim-opus.md`
- `models/instructblip-vicuna-7b/config.json`（`eos_token_id` 那个修正）— 注意：模型权重不进 git，只单独把这一个 config 文件 add 进来

操作建议（**复制即可**）：

```bash
cd /home/nlp/VHR

# 1. 写 .gitignore
cat > .gitignore <<'EOF'
data/
models/
results/
POPE/
val2014/
train2014/
*.zip
*.zip.*
__pycache__/
*.log
logs/
.cache/
.ipynb_checkpoints/
.vscode/
.idea/
*.swp
.DS_Store
EOF

# 2. 把代码侧改动 add 进来
git add .gitignore
git add eval/eval.py main.py scripts/cat_files.sh scripts/infer_instructblip.sh
git add scripts/prepare_data.sh
git add fuxian.md study.md optim.md optim-opus.md

# 3. 单独把那个 config 改动 add 进来（不要 add models/ 整个）
git add -f models/instructblip-vicuna-7b/config.json

# 4. 处理被删除的文件（这些是历史遗留，按你的实际意图决定）
git add scripts/infer_llava1_5.sh scripts/infer_llava1_6.sh requirements.txt
# 如果 requirements.txt 不应该被删，先 git checkout -- requirements.txt 恢复

git commit -m "baseline: fix instructblip eos/vhr_layers/merge & add analysis docs"

# 5. 打个里程碑 tag，永远可以回到这里
git tag -a v0-baseline-fixed -m "InstructBLIP VHR 复现修复后的稳定基线"
```

完成后你随时可以 `git checkout v0-baseline-fixed -- <file>` 把单个文件恢复，或 `git checkout v0-baseline-fixed` 整体回到这个状态。

---

## 第二步：每个实验开一条分支（**强烈推荐**）

`optim-opus.md` 里 6 条方法天然适合「**一方法一分支**」。

```bash
git checkout -b feat/avhr      v0-baseline-fixed   # 实验 A-VHR
git checkout -b feat/egvhr     v0-baseline-fixed   # 实验 EG-VHR
git checkout -b feat/qfa-vhr   v0-baseline-fixed   # 实验 QFA-VHR
git checkout -b feat/ocd       v0-baseline-fixed   # 实验 OCD
# ...
```

好处：

- **失败回滚** = `git checkout main` 或直接 `git branch -D feat/xxx`，随时丢弃。
- **正交对比** = 每条分支独立跑，结果文件名带 `--method` 区分，互不污染。
- **后续合并** = 哪个有效就 `git checkout main && git merge feat/xxx`；哪条失败就 `git branch -D` 删掉。

每条分支建议在每个**有效中间结果**处提交一次，比如：

```bash
git add vhr.py main.py
git commit -m "avhr: continuous head gating, gamma_l=0.5, rho_max=2"
```

---

## 第三步：实验过程中的「快速保存」

实验中常常代码还没写完但想试试看，**不一定每次都正式 commit**，可以：

### A. `git stash`（临时藏起来）

```bash
git stash push -m "试一下 tau=0.5 的版本"
# 跑一下 baseline 对比
git stash pop          # 拿回来
git stash list         # 看所有暂存
```

### B. WIP commit（更安全，不容易丢）

```bash
git commit -am "WIP: tau=0.5 试一下"
# 实验完不想要 → git reset --soft HEAD^
# 实验完想保留 → git commit --amend 改个正式信息
```

**更推荐 WIP commit**：stash 在切分支、合并冲突时容易出问题，commit 在 reflog 里永远找得回来。

### C. 学会 `git reflog` —— 后悔药

无论你 reset / 删分支 / rebase 出错，`git reflog` 都能找回最近 90 天内 HEAD 走过的每一步：

```bash
git reflog
git checkout <那个 sha>
```

---

## 第四步（可选，强烈推荐多 GPU 用户）：`git worktree` 并行实验

如果你想**同时**在 GPU 0/1/2 上跑不同方法，又不想反复 `git checkout`：

```bash
git worktree add ../VHR-avhr   feat/avhr
git worktree add ../VHR-egvhr  feat/egvhr
git worktree add ../VHR-qfa    feat/qfa-vhr
```

效果：

- 三份**独立的工作目录**，共享同一份 `.git`，互不干扰。
- 每个目录可以在不同 GPU 上跑各自的 `bash scripts/infer_instructblip.sh`。
- 实验完：`git worktree remove ../VHR-avhr`。

这是 ML 多卡多实验最舒服的方案。

---

## 第五步：实验**结果**怎么保存（不进 git，但要可追溯）

代码改动靠 git，**结果/日志/模型**别往 git 里塞。建议结果按「分支 + 时间戳」分目录：

修改 `main.py` 输出目录的小补丁（思路）：

```python
output_dir = f"results/{branch_name}/{run_tag}/{model_name}"
```

或者更简单，**约定一个命名规则**：

```bash
# 跑实验前
EXP_TAG="avhr_tau0.5_rho2_$(date +%m%d_%H%M)"
mkdir -p results/$EXP_TAG
# 跑完后把关键 metric 用一行写到一个固定 csv
echo "$(git rev-parse --short HEAD),$EXP_TAG,CHAIRs=...,CHAIRi=...,Len=..." >> experiments.csv
```

把 `experiments.csv` 也提交进 git，于是**每条实验都能反查到「代码 sha + 超参 + 结果」**——这对后期论文写作非常关键。

---

## 推荐的最简工作流（一图流）

```text
main (origin/main)
 │
 ├── tag: v0-baseline-fixed  ←── 当前修复后的稳定基线（永远可回）
 │
 ├── feat/avhr      ←── 方法 1：A-VHR
 │     └── WIP commits, 失败就 git branch -D feat/avhr
 │
 ├── feat/egvhr     ←── 方法 2：EG-VHR
 ├── feat/qfa-vhr   ←── 方法 3：QFA-VHR
 ├── feat/ocd       ←── 方法 5：OCD
 │
 └── (验证有效后)→ git merge feat/xxx → main → 新 tag v1-avhr
```

---

## 你现在最该做的 3 件事

1. **立刻**写 `.gitignore`，提交 baseline，打 tag `v0-baseline-fixed`（10 分钟，永久受益）。
2. 每个 `optim-opus.md` 里的方法**开一条 `feat/*` 分支**再动手。
3. 维护一份 `experiments.csv`（commit sha + 超参 + 指标），写论文时直接查。
