# A-VHR：Adaptive-ratio VHR 实验记录

> **文档说明**
>
> 本文档专门记录 **A-VHR（Adaptive-ratio VHR）** 这一方法（即 `optim-opus.md` 方法 1）的全过程，
> 涵盖**方法设计、代码修改、操作流程、实验记录、结果分析**。所有后续与 A-VHR 相关的修改、调参、
> 跑数、Bug 修复、阶段性结论，都**统一更新到本文档**，按时间顺序追加到第 6 节「进度日志」并在
> 相应章节同步修订。论文写作时直接以本文档为单源。
>
> - **方法定位**：在 `optim-opus.md` 的 6 条优化方案中作为核心创新点之一（与 EG-VHR 配合构成主创新）。
> - **代码分支**：`feat/avhr`（基于 `v0-baseline-fixed` 拉出，独立于 `main`）。
> - **基线参考**：原版 VHR（`vhr_layers=14`、`aug_ratio=2`、`vhr_filter=True`）+ fuxian.md 修复后的设置。
> - **创建日期**：2026-04-20
> - **最后更新**：2026-04-20

---

## 1. 方法设计

### 1.1 动机

原版 VHR（`vhr.py::vhr_forward`）采用「**两段式硬阈值**」机制对视觉敏感 head 加权：

1. 对每个 head 计算 `head_vary = ((attn_tc − attn_full)²).sum(-1)`（视觉去除前后的 attention 输出差异）。
2. 选取 `head_vary > head_vary.median()` 的所有 head，**统一乘以固定** `aug_ratio=2`。

这种设计存在两个问题：

| 问题 | 表现 | A-VHR 对策 |
|------|------|-----------|
| 阈值与样本无关 | 不论该样本视觉敏感性强弱，永远选中一半 head | 用连续 sigmoid 替代硬选 |
| 加权与层无关 | 高语义化的顶层 head 也被等比例放大，易引发列举型/语言风格型幻觉 | 引入层深衰减系数 `gamma_l`，越靠近输出层越保守 |

### 1.2 形式化定义

对作用层 \(l\)、第 \(h\) 个 head：

\[
r_{l,h}(x) = 1 + (\rho_{\max} - 1) \cdot \gamma_l \cdot \sigma\!\left(\frac{v_{l,h} - \mu_l}{\tau \cdot s_l}\right)
\]

其中：

- \(v_{l,h}\)：head_vary（与原版 VHR 一致，已计算）。
- \(\mu_l, s_l\)：层内 head_vary 的均值与标准差。
- \(\sigma\)：sigmoid。
- \(\tau\)：温度，越小越接近原版硬阈值（\(\tau \to 0^+\) 退化为 step 函数）。
- \(\rho_{\max}\)：单 head 加权上限。
- \(\gamma_l = \exp\!\left(-\alpha \cdot \frac{l - l_{\text{start}}}{\max(1,\, l_{\text{end}} - l_{\text{start}})}\right)\)：层深衰减。
  - 在 VHR 作用范围 \([l_{\text{start}}, l_{\text{end}}]\) 内，浅层 \(\gamma_l = 1\)、深层 \(\gamma_l = e^{-\alpha}\)。
  - \(\alpha = 0\) 时退化为常数 1（不衰减）。

### 1.3 设计要点

- **退化等价性**：当 \(\tau \to 0^+\) 且 \(\alpha = 0\) 时，A-VHR 几乎退化为原版 VHR（除头数严格 50% 这点）。
  这一点是回归测试与公平对比的基础。
- **计算成本**：相比原版 VHR 仅多 1 次 sigmoid + 1 次乘法，**几乎零额外开销**。
- **可解释性**：\((\tau, \alpha, \rho_{\max})\) 三个超参各自承担明确语义（陡峭度、层深保守度、加权上限），方便消融。

---

## 2. 代码修改说明

### 2.1 受影响文件

| 文件 | 改动类型 | 关键内容 |
|------|----------|----------|
| `vhr.py` | 修改 | `vhr_forward` 增加 4 个 A-VHR 参数；新增 `_compute_gamma_l`；`replace_attn` 增加 4 个开关 |
| `generation.py` | 修改 | `generation_config_init` 注册 `vhr_use_avhr / vhr_tau / vhr_alpha / vhr_rho_max`；VHR_GENERATION 分支透传 |
| `main.py` | 修改 | 新增 CLI `--vhr_tau / --vhr_alpha / --vhr_rho_max`；`avhr` 出现在 `--method` 时打开 A-VHR；修复 `model_name` 派生 |
| `scripts/infer_instructblip.sh` | 修改 | `vhr_layers 18 → 14`（fuxian.md 修复） |
| `scripts/infer_instructblip_avhr.sh` | 新增 | A-VHR 推理 launcher，单卡/多卡/nohup 一键 |
| `models/instructblip-vicuna-7b/config.json` | 已修复 | `text_config.eos_token_id=2`（fuxian.md 修复，本次确认无需再改） |

### 2.2 关键改动片段

**`vhr.py::vhr_forward` 新签名（默认值与原版兼容）**

```python
def vhr_forward(
    self,
    ...,
    aug_ratio: float = 1,
    filter: bool = False,
    use_avhr: bool = False,
    avhr_tau: float = 1.0,
    avhr_rho_max: float = 2.0,
    avhr_gamma_l: float = 1.0,
    **kwargs,
):
```

**A-VHR 分支（在第二次 prefill 触发，缓存 per-head ratio 给后续 decoding 步复用）**

```python
if use_avhr:
    mu = head_vary.mean()
    s = head_vary.std() + 1e-6
    gate = torch.sigmoid((head_vary - mu) / (avhr_tau * s))      # [num_heads]
    ratio = 1.0 + (avhr_rho_max - 1.0) * avhr_gamma_l * gate     # [num_heads]
    ratio = ratio.to(attn_output.dtype).view(1, -1, 1, 1)
    self.aug_ratio_per_head = ratio
    self.aug_heads = None
    attn_output = attn_output * ratio
else:
    self.aug_heads = (head_vary > head_vary.median()).nonzero().flatten()
    self.aug_ratio_per_head = None
    attn_output[:, self.aug_heads, :, :] = aug_ratio * attn_output[:, self.aug_heads, :, :]
```

**`vhr.py::_compute_gamma_l`**

```python
def _compute_gamma_l(layer_idx, target_layers, alpha):
    if not target_layers:
        return 1.0
    l_start = min(target_layers)
    l_end = max(target_layers)
    span = max(1, l_end - l_start)
    return math.exp(-alpha * (layer_idx - l_start) / span)
```

**`main.py` 触发逻辑（method 含 `avhr` 时打开 A-VHR）**

```python
if "vhr" in self.method:
    vhr_layers = list(range(len(get_layers(self.model))))[-self.vhr_layers:] if self.vhr_layers > 0 else []
    vhr_layers = [1] + vhr_layers if self.vhr_layer1 else vhr_layers
    generate_kwargs.update({...})
    if "avhr" in self.method:
        generate_kwargs.update({
            "vhr_use_avhr": True,
            "vhr_tau": self.vhr_tau,
            "vhr_alpha": self.vhr_alpha,
            "vhr_rho_max": self.vhr_rho_max,
        })
```

### 2.3 兼容性

- **原版 VHR 不受影响**：`use_avhr` 默认 `False`，所有原有调用路径保持原行为。
- **方法识别**：`"avhr" in self.method` 与 `"vhr" in self.method` 同时为真，所以 A-VHR 自动复用 VHR 已有的 `vhr_image_start` / `vhr_image_end` 设置逻辑（main.py 中 `elif "vhr" in self.method:` 分支）。

### 2.4 Lint / 语法验证

| 检查项 | 结果 |
|--------|------|
| `python -c "ast.parse(...)"` 三个文件 | OK |
| Cursor lint | 无错误 |
| `gamma_l` 数学验证（`vhr_layers=14`，LLaMA-7B 32 层，`alpha=0.5`） | layer 18 → 1.0000，layer 31 → 0.6065 ✓ |
| `gamma_l` 退化（`alpha=0`） | 全部 1.0000 ✓ |

---

## 3. 操作说明

### 3.1 环境与前置

- 已在 `feat/avhr` 分支（commit `0ff92f7`）。
- InstructBLIP 模型权重位于 `models/instructblip-vicuna-7b/`，`text_config.eos_token_id=2`。
- **层数说明（更正）**：论文对 **InstructBLIP 用最后 18 层**（`§4.4`，且不含 layer1），对 LLaVA 系列才是「第 2 层 + 最后 14 层」。早前本文档「默认 `vhr_layers=14` 与论文一致」的说法**仅对 LLaVA 成立，对 InstructBLIP 不准确**。
  - 复现验证（见 [`reproduce.md`](../reproduce.md)）：InstructBLIP 用 **h18** 重跑 VHR 得 CHAIRs 37.84 / CHAIRi 9.72 / Len 106.42，与论文 37.76 / 9.75 / 106.49 几乎一致；h14 偏离论文。
  - 因此 **InstructBLIP 上的 A-VHR 实验改用 `vhr_layers=18`** 以与论文对齐，其同层数基线 B0 = 原版 `vhr-h18`（已有复现结果可直接对照）。
  - 注意：切换 14↔18 不仅增减作用层，还会改变 A-VHR 层深衰减 `gamma_l` 的起点 `l_start` 与跨度 `span`，进而重塑每层的衰减曲线。

### 3.2 单卡前台跑（看进度、调参用）

```bash
cd /home/nlp/VHR
NUM_CHUNKS=1 CUDA_DEVICE=cuda:2 \
  AVHR_TAU=1.0 AVHR_ALPHA=0.5 AVHR_RHO_MAX=2.0 \
  bash scripts/infer_instructblip_avhr.sh avhr
```

### 3.3 多卡并行（4 卡，自动合并 + 评估）

```bash
NUM_CHUNKS=4 \
  AVHR_TAU=1.0 AVHR_ALPHA=0.5 AVHR_RHO_MAX=2.0 \
  bash scripts/infer_instructblip_avhr.sh avhr
```

### 3.4 后台不挂断（关电脑不影响）

```bash
mkdir -p logs
nohup bash scripts/infer_instructblip_avhr.sh avhr \
  > logs/avhr_$(date +%m%d_%H%M).log 2>&1 &
```

### 3.5 输出文件命名

`results/instructblip-vicuna-7b/{dataset}_{seed}-avhr-h{layers}-t{tau}-a{alpha}-r{rho_max}-{max_new_tokens}.json`

例如：

```
results/instructblip-vicuna-7b/chair_0-avhr-h14-t1.0-a0.5-r2.0-512.json
```

这样不同超参组合的结果**互不覆盖**，可同目录共存便于对比。

### 3.6 显存与时间估计

- InstructBLIP-7B fp16，单卡推理峰值 ≈ 18 GB。
- CHAIR 500 张 × 5 seeds + POPE 三分割 ≈ 5500 次 generation。
- 单卡 max_new_tokens=512 估时约 4–6 小时（与原版 VHR 同量级，A-VHR 几乎零额外开销）。
- 4 卡并行约 1–1.5 小时。

### 3.7 评估

脚本末尾会自动调用：

```bash
python eval/eval.py "results/$model_name/*chair_*-${method}-${max_new_tokens}.json" --eval_chair
python eval/eval.py "results/$model_name/*pope_*_0-${method}-${max_new_tokens}.json" --eval_pope
```

输出 CHAIRs / CHAIRi / Len 与 POPE Acc / F1 / Yes%。

---

## 4. 实验设计

### 4.1 评测协议（统一）

| 指标 | 数据 | 说明 |
|------|------|------|
| **CHAIRs / CHAIRi / Len** | CHAIR 500 imgs × 5 seeds | 物体级幻觉 + 平均长度 |
| **POPE Acc / F1 / Yes%** | POPE popular / adversarial / random | 存在性幻觉 |
| **样例可视化** | 取若干含/不含幻觉样例对比 | 解释性 |

所有结果记录在第 5 节「实验结果」表格中，每行附 commit sha 与日期。

### 4.2 消融网格

| 实验 ID | tau | alpha | rho_max | 验证目的 | 状态 |
|---------|-----|-------|---------|----------|------|
| **B0** | — | — | — | 原版 VHR (`vhr-h14`) baseline | 待跑 |
| **A1（退化）** | 0.01 | 0 | 2.0 | sigmoid 几乎硬阈值 + 不衰减；应≈ B0，回归用 | 待跑 |
| **A2（仅连续 gate）** | 1.0 | 0 | 2.0 | 单独看连续化收益 | 待跑 |
| **A3（仅层深衰减）** | 0.01 | 1.0 | 2.0 | 单独看 γ_l 收益 | 待跑 |
| **A4（默认）** | 1.0 | 0.5 | 2.0 | A-VHR 主结果 | 待跑 |
| **A5（强）** | 0.5 | 1.0 | 2.5 | 看上限 | 待跑 |
| **A6（弱）** | 2.0 | 0.5 | 1.5 | 看下限 | 待跑 |

### 4.3 时间安排

- 第 1 阶段（B0 + A4，2 组）：先跑 baseline 与默认 A-VHR，观察方向是否对，预计 0.5–1 天。
- 第 2 阶段（A1–A3，3 组）：消融，验证两个机制各自贡献，预计 1–2 天。
- 第 3 阶段（A5–A6，2 组）：超参鲁棒性，预计 1 天。
- 第 4 阶段：跨模型扩展（LLaVA-1.5 / LLaVA-1.6），写入第 7 节「跨模型扩展」。

### 4.4 成功判定（第 1 阶段达标线）

- A4 相对 B0：CHAIRs ↓ ≥ 0.5 个百分点 **且** Len 不显著增长（±5%）**且** POPE 不退步。
- 若不达标 → 进入「调参 / 调机制」迭代，所有调整记录到第 6 节进度日志。

---

## 5. 实验结果

> 每跑完一组就在此表追加一行；commit 列填实际跑数时所在的 git sha。
>
> **重要更新（2026-06-25）**：实验层数统一改为 **h18**（InstructBLIP 论文设定），不再用 h14。
> 因此 **B0 = 原版 `vhr-h18`**，直接采用 [`reproduce.md`](../reproduce.md) 已复现的结果（与论文 37.76/9.75 一致）。
> 下表 CHAIR 为 5 seed 均值±std；POPE 为三分割「Acc」。参考：Greedy 基线 CHAIRs 45.80 / CHAIRi 13.10 / Len 92.07 / POPE-F1 85.25。

| 日期 | 实验 ID | 层 | tau | alpha | rho_max | CHAIRs | CHAIRi | Len | POPE-pop Acc | POPE-adv Acc | POPE-rand Acc | 备注 |
|------|---------|----|-----|-------|---------|--------|--------|-----|--------------|--------------|---------------|------|
| 2026-06-24 | **B0** | h18 | — | — | — | **37.84±2.29** | **9.72±0.79** | 106.42 | 85.43 | 82.97 | 89.33 | 原版 VHR（=reproduce.md，对齐论文 37.76/9.75/106.49）|
| 2026-06-25 | **A4** | h18 | 1.0 | 0.5 | 2.0 | **47.00±2.39** | **13.16±0.66** | 100.95 | 85.07 | 82.10 | 89.97 | A-VHR 默认；**显著劣于 B0**，见 §7 |
| TODO | A1 | h18 | 0.01 | 0 | 2.0 |  |  |  |  |  |  | 退化等价测试（应≈B0）|
| TODO | A2 | h18 | 1.0 | 0 | 2.0 |  |  |  |  |  |  | 仅连续 gate |
| TODO | A3 | h18 | 0.01 | 1.0 | 2.0 |  |  |  |  |  |  | 仅层深衰减 |
| TODO | A5 | h18 | 0.5 | 1.0 | 2.5 |  |  |  |  |  |  | 强 |
| TODO | A6 | h18 | 2.0 | 0.5 | 1.5 |  |  |  |  |  |  | 弱 |

> A4 POPE 三分割 F1：pop 85.03 / adv 82.55 / rand 89.42，**均值 85.67**（B0 为 85.47）。POPE 上 A-VHR 与 VHR 基本持平。

### 5.1 跨模型扩展（待第 4 阶段填）

| 模型 | 方法 | CHAIRs | CHAIRi | Len | POPE Acc | 备注 |
|------|------|--------|--------|-----|----------|------|
| LLaVA-1.5-7B | VHR baseline |  |  |  |  |  |
| LLaVA-1.5-7B | A-VHR (A4 配置) |  |  |  |  |  |
| LLaVA-1.6-7B | VHR baseline |  |  |  |  |  |
| LLaVA-1.6-7B | A-VHR (A4 配置) |  |  |  |  |  |

---

## 6. 进度日志

### 2026-06-25

- ✅ **B0 + A4 跑完（均 h18，6 卡并行 GPU1/2/3/4/5/7）**。
  - **B0（原版 vhr-h18）**：CHAIRs 37.84 / CHAIRi 9.72 / Len 106.42 / POPE-F1 85.47（直接采用 `reproduce.md` 复现结果，对齐论文）。
  - **A4（avhr-h18-t1.0-a0.5-r2.0）**：CHAIRs 47.00±2.39 / CHAIRi 13.16±0.66 / Len 100.95 / POPE-F1 85.67。
  - **结论：A4 默认配置显著劣于 B0**（CHAIRs +9.16，CHAIRi +3.44），甚至略差于 greedy（45.80）。POPE 持平。详见 §7。
- 🐞 **修复 `scripts/cat_files.sh`**：原逻辑要求 0–7 号分片全部存在才合并，否则「删分片却不合并」→ eval 出 nan（这正是 04-20 A-VHR 跑失败的真因）。改为「只合并实际存在的分片 + 合并成功才删」，2/6 卡均可安全合并。
- ⚠️ **环境注意**：CHAIR 评测须用 conda `vhr` 环境（nltk 3.5 带 punkt）；`.venv` 的新版 nltk 需 `punkt_tab`，当前机器联网受限下不到，故评测走 conda env。
- ⏳ **下一步**：A4 既然偏弱，优先验证「是否因加权强度不足」——
  1. 跑 **A1（tau=0.01, alpha=0, rho_max=2.0）** 退化测试：若 A1≈B0，证明实现正确、差距来自 A4 的软化。
  2. 跑 **A5（强：tau=0.5, alpha=1.0, rho_max=2.5）** 看提高上限+陡峭 gate 能否追平/超过 B0。
  3. 视情况扫 rho_max ∈ {2.5, 3.0}、alpha ∈ {0, 0.25}。

### 2026-04-20

- ✅ **方法实施**（commit `0ff92f7`，分支 `feat/avhr`）。
  - `vhr.py` 扩展 `vhr_forward`，新增连续 gate + 层深衰减。
  - `generation.py` 注册 4 个新超参并透传。
  - `main.py` 增加 CLI 参数与 method 触发逻辑。
  - `scripts/infer_instructblip_avhr.sh` 新增 launcher。
- ✅ **基线修复**：
  - `scripts/infer_instructblip.sh`：`vhr_layers 18 → 14`。
  - `main.py`：`model_name` 派生改为 `realpath + basename`。
  - `models/instructblip-vicuna-7b/config.json`：`text_config.eos_token_id=2`（已就位）。
- ✅ **代码验证**：三文件 AST/lint OK；`gamma_l` 数学正确性手算验证通过。
- ⏳ **下一步**：
  1. 先跑 B0（原版 VHR 基线）确认 fuxian.md 修复生效，CHAIRs 应在 ~37–40。
  2. 再跑 A4（A-VHR 默认配置）做主结果。
  3. 视结果安排消融（A1–A3, A5–A6）。

> 用法：每次跑完一组实验或做了一次代码改动，**追加新条目**到本节顶部，附日期与简要说明，必要时同步修订前面章节。

---

## 7. 结果分析（第 1 阶段：B0 vs A4）

### 7.1 现象

| 指标 | Greedy | B0 (VHR) | A4 (A-VHR) | A4 − B0 |
|------|--------|----------|------------|---------|
| CHAIRs ↓ | 45.80 | **37.84** | 47.00 | **+9.16（变差）** |
| CHAIRi ↓ | 13.10 | **9.72** | 13.16 | **+3.44（变差）** |
| Len | 92.07 | 106.42 | 100.95 | −5.47 |
| POPE-F1 | 85.25 | 85.47 | **85.67** | +0.20（基本持平）|

**A4 默认配置的 A-VHR 在 CHAIR 上明显劣于原版 VHR，几乎退回 greedy 水平**；POPE 因是 yes/no 短答、对加权强度不敏感，三者基本持平。**第 1 阶段未达 §4.4 的达标线（CHAIRs 应 ↓≥0.5）**。

### 7.2 机制解读：根因是「加权强度被显著削弱」

A-VHR 把原版「**top-50% 头 × 固定 2.0**」换成「**所有头 × 连续 ratio**」：

\[ r_h = 1 + (\rho_{\max}-1)\cdot\gamma_l\cdot\sigma\big((v_h-\mu)/(\tau s)\big) \]

在 A4（τ=1.0, α=0.5, ρ_max=2.0）下，这套连续加权对"看图头"的放大**普遍小于原版的 2.0**：

- **sigmoid 软化**：gate 以均值为中心，只有 `v_h` 远大于均值的头才接近 1；大量中高敏感头的 gate 仅 0.5–0.8 → ratio 仅 ~1.5–1.8（< 2.0）。
- **层深衰减再打折**：α=0.5 使深层 γ_l 低至 0.61，深层"看图头"被进一步压低到 ~1.3–1.5。
- **净效果**：A-VHR 实际施加的强化**弱于**原版 VHR → 抵抗语言偏置的力度不足 → 幻觉回升。这与 §论文 Table 6「α 越小幻觉越重」的趋势一致（A-VHR 等效平均倍率明显 < 2）。

即：A4 的两项"温柔化"设计（连续 gate + 层深衰减）方向上都在**减弱**干预，叠加后强度不够，所以效果差。

### 7.3 失败模式

- Len 反而比 B0 短（100.95 < 106.42），说明 A-VHR 没有像脏数据那样"停不下来"，**纯粹是去幻觉能力不足**，不是生成失控。
- POPE 不退步说明实现没有破坏模型基本判断力，问题集中在"开放式描述里压制不住虚构物体"。

### 7.4 超参敏感性（待 A1/A2/A3/A5 验证的假设）

- **怀疑点**：A4 偏弱主要来自 ρ_max=2.0 上限 + sigmoid 软化 + α=0.5 衰减"三重打折"。
- **A1（τ=0.01, α=0, ρ_max=2.0）**：gate 趋近硬阈值、不衰减 → 预期 ≈ B0；若成立则证明**实现正确**，差距纯来自 A4 的软化参数。
- **A5（τ=0.5, α=1.0, ρ_max=2.5）**：提高上限 + 更陡 gate，但 α=1.0 衰减更强，方向相反，需实测；更干净的"增强"应是 **ρ_max≥2.5 且 α=0**。
- 初步推断：要让 A-VHR 追平/超过 VHR，需 **ρ_max 提到 2.5–3.0、α 降到 0–0.25、τ 调小让 gate 更接近硬选**。

### 7.5 跨模型一致性

- 暂未在 LLaVA-1.5/1.6 上跑（第 4 阶段）。当前结论仅限 InstructBLIP。

### 7.6 写作要点

- 目前 A-VHR 是**负结果**：默认超参下不如原版 VHR。若后续消融仍无法超过 B0，则本方法的价值点需重新定位（例如改为"更少 head 干预下维持同等效果"的效率论证，而非直接刷低 CHAIRs）。

---

## 8. 已知限制与后续

- 当前 `gamma_l` 是按 layer index 线性插值后取 exp 的简化形式，可考虑替换为 **学习式曲线**（仍 training-free 的话用一次少量验证集网格搜索）。
- A-VHR 仅作用在「VHR 已选定的最后 k 层」，未覆盖 EAH 风格的浅层干预，对应 `optim-opus.md` 方法 6 (CLHR) 的层段路由可作为后续扩展。
- 与 EG-VHR（方法 2）正交，可叠加；建议先单独验证 A-VHR，再在另一条分支 `feat/egvhr` 做组合实验。

---

## 9. 引用与对接

- 方法定义：`optim-opus.md` 第 1 节「方法 1：自适应头加权 A-VHR」。
- 复现修复：`optim-opus.md` 附录第 1 步 / fuxian.md（已并入本仓库）。
- 论文目标段落：方法章节 §3「Adaptive-ratio Head Re-weighting」；实验章节消融小节。

---

**本文档为 A-VHR 方法的单源（single source of truth）。任何代码改动、超参调整、结果更新都应同步反映在第 6 节进度日志，并在相应章节修订。**
