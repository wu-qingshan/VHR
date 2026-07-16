# VHR 复现报告：实验结果与论文数据对比

> 论文：*Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence*（ACL 2025）
> 仓库：本项目（VHR 官方实现）
> 报告日期：2026-06-02（2026-06-02 晚更新：InstructBLIP VHR 复现成功）
> 数据来源：论文 Table 1 / Table 2 + 本机对 `results/` 下 JSON 重新运行 `eval/eval.py` 复算

---

## 0. TL;DR（一句话结论）

| 模型 | 方法 | 复现状态 |
|------|------|----------|
| LLaVA-1.5-7B | VHR | ✅ **成功**，与论文几乎一致 |
| LLaVA-NeXT-7B | VHR | ✅ **成功**，与论文几乎一致 |
| InstructBLIP-7B | Greedy | ✅ 与论文 Greedy 基本一致 |
| InstructBLIP-7B | VHR | ✅ **已成功复现**（修正 eos=2 + 层数=18 后）：37.84 / 9.72 / 106.42 / F1 85.47，与论文 37.76 / 9.75 / 106.49 / 85.52 **几乎完全一致** |
| A-VHR（本地新方法） | — | ⏳ 仅有方案，**无任何实验数字** |

**结论**：InstructBLIP 的 VHR 此前与论文差距大，根因是 **生成不会在正确位置停止 → caption 长度暴涨 → CHAIR 指标恶化**，而非算法本身错误（POPE 几乎不受影响、Greedy 正常即为佐证）。**经修正 `eos_token_id=2` 并把作用层改回论文的 18 层后重跑，已完全复现论文结果**（见 §2、§3.6）。

---

## 1. 论文报告的基准数字

### Table 1 — CHAIR（500 图 × 5 seeds，prompt "Please describe this image in detail."，`max_new_tokens=512`）

| 模型 | 方法 | CHAIRs ↓ | CHAIRi ↓ | Len |
|------|------|----------|----------|-----|
| **InstructBLIP** | Greedy | 45.32±2.24 | 12.98±0.76 | 91.06 |
| | **VHR** | **37.76±2.76** | **9.75±0.98** | **106.49** |
| **LLaVA-1.5** | Greedy | 49.68±1.47 | 14.32±0.78 | 83.06 |
| | **VHR** | **33.32±1.31** | **9.71±0.45** | **81.33** |
| **LLaVA-NeXT** | Greedy | 29.08±2.09 | 8.08±0.74 | 157.06 |
| | **VHR** | **24.96±2.09** | **6.80±0.59** | **156.92** |

### Table 2 — POPE F1（popular / adversarial / random 三分割平均）

| 模型 | Greedy | **VHR** |
|------|--------|---------|
| InstructBLIP | 85.36 | **85.52** |
| LLaVA-1.5 | 84.98 | **85.47** |
| LLaVA-NeXT | 88.51 | **88.87** |

### 论文实现要点（§4.4）

- 放大倍数 **α = 2**。
- VHR 作用层：**LLaVA 系列 = 第 2 层 + 最后 14 层**；**InstructBLIP = 最后 18 层**（不含 layer1）。
- `max_new_tokens = 512`；beam 数（涉及 beam search 时）= 5。
- InstructBLIP 需把 `config.json` 的 `text_config.eos_token_id` 改为 **2**。

---

## 2. 本机复现结果 vs 论文

> 下列数值为对仓库 `results/` 下已有 JSON 重新运行 `eval/eval.py` 得到。

| 模型 | 本地配置 | CHAIRs | CHAIRi | Len | POPE F1（三分割均） | 与论文对照 |
|------|----------|--------|--------|-----|---------------------|-----------|
| **LLaVA-1.5** | `vhr-l1h14` | **33.40±1.39** | **9.77±0.46** | **81.46** | **85.30** | 论文 VHR 33.32 / 9.71 / 81.33 / 85.47 → **一致** |
| **LLaVA-NeXT** | `vhr-l1h14` | **25.16±1.73** | **6.83±0.63** | **156.05** | **89.02** | 论文 VHR 24.96 / 6.80 / 156.92 / 88.87 → **一致** |
| **InstructBLIP** | `greedy` | **45.80±2.55** | **13.10±0.68** | **92.07** | **85.25** | 论文 Greedy 45.32 / 12.98 / 91.06 / 85.36 → **基本一致** |
| **InstructBLIP** | `vhr-h18`（✅ 修复后重跑） | **37.84±2.29** | **9.72±0.79** | **106.42** | **85.47** | 论文 VHR 37.76 / 9.75 / 106.49 / 85.52 → **几乎完全一致** ✅ |
| ~~InstructBLIP~~ | ~~`vhr-h18`（旧脏数据，已归档）~~ | ~~50.76~~ | ~~11.99~~ | ~~307.86~~ | ~~85.43~~ | eos 未修复时的产物，已移至 `_archive_dirty_vhr_h18/` |
| **A-VHR** | B0 / A1–A6 | 待跑 | 待跑 | 待跑 | 待跑 | 无数据 |

### POPE 三分割明细（本机复算）

| 模型/方法 | 子集 | Acc | Prec | Rec | F1 | Yes% |
|-----------|------|-----|------|-----|-----|------|
| LLaVA-1.5 `vhr-l1h14` 
  | random | 88.87 | 86.26 | 92.47 | 89.25 | 53.60 |
| | popular | 85.37 | 80.97 | 92.47 | 86.34 | 57.10 |
| | adversarial | 77.37 | 71.08 | 92.27 | 80.30 | 64.90 |
| | **平均** | 83.87 | 79.44 | 92.40 | **85.30** | 58.53 |
| LLaVA-NeXT `vhr-l1h14`
  | random | 91.83 | 96.52 | 86.80 | 91.40 | 44.97 |
| | popular | 89.60 | 91.95 | 86.80 | 89.30 | 47.20 |
| | adversarial | 86.33 | 86.14 | 86.60 | 86.37 | 50.27 |
| | **平均** | 89.25 | 91.54 | 86.73 | **89.02** | 47.48 |
| InstructBLIP `greedy` 
  | random | 90.20 | 92.83 | 87.13 | 89.89 | 46.93 |
| | popular | 83.43 | 81.13 | 87.13 | 84.02 | 53.70 |
| | adversarial | 80.70 | 77.26 | 87.00 | 81.84 | 56.30 |
| | **平均** | 84.78 | 83.74 | 87.09 | **85.25** | 52.31 |
| InstructBLIP `vhr-h18`（✅ 修复后） 
  | random | 89.33 | 95.60 | 82.47 | 88.55 | 43.13 |
| | popular | 85.43 | 87.67 | 82.47 | 84.99 | 47.03 |
| | adversarial | 82.97 | 83.30 | 82.47 | 82.88 | 49.50 |
| | **平均** | 85.91 | 88.86 | 82.47 | **85.47** | 46.55 |

---

## 3. 为什么 InstructBLIP 的 VHR 实验与原文差距大

### 3.1 异常画像

把旧 `vhr-h18` 结果与论文并排看，问题非常集中：

| 指标 | 论文 VHR | 本地旧 VHR(h18) | 偏差 |
|------|----------|-----------------|------|
| CHAIRs ↓ | 37.76 | **50.76** | +13.0（大幅变差，甚至高于 Greedy 45.80） |
| CHAIRi ↓ | 9.75 | 11.99 | +2.24 |
| **Len** | **106.49** | **307.86** | **+201（约 3 倍）** |
| POPE F1 | 85.52 | 85.43 | −0.09（几乎无差） |

关键信号：**Len 暴涨到 ~308**，而 POPE F1 几乎不变。POPE 是 yes/no 短答（生成 1～2 个 token 即停），不依赖长文收尾；CHAIR 是开放式长描述，强依赖"在正确位置停止生成"。**所以问题出在『生成停不下来』，而不是 head 加权算法本身。**

### 3.2 根因一：生成不在正确 EOS 处停止（最关键）

- InstructBLIP 用的 Vicuna 分词器中，`token id 1 = <s>`（BOS），`token id 2 = </s>`（真正的 EOS）。
- 若 `text_config.eos_token_id` 没设成 **2**，停止判定会错位。
- **Greedy 下"碰巧"还能较早停**（Len≈92，接近论文 91），所以 Greedy 看起来正常，掩盖了配置问题。
- 但 **VHR 放大视觉敏感 head 后改变了 logit 分布**，模型几乎不再生成那个被错当成 EOS 的 token，于是绝大多数样本一路顶到 `max_new_tokens=512` 才停 → Len 从 ~106 飙到 ~308 → 长描述里堆叠更多物体 → CHAIRs/CHAIRi 一起恶化。

这解释了"Greedy 正常、VHR 崩坏、POPE 不受影响"这一组看似矛盾的现象。

> 当前状态：`models/instructblip-vicuna-7b/config.json` 的 `text_config.eos_token_id` **已为 2**（第 104 行）。即旧 `vhr-h18` 结果是在修复之前跑的，属于历史脏数据，应作废。

### 3.3 根因二：作用层数与论文不一致

- 论文明确 InstructBLIP 用 **最后 18 层**（`§4.4`）。
- 旧结果命名 `vhr-h18`（18 层，方向对），但叠加了 eos 错误，问题被放大。
- 之后为对齐 A-VHR 把脚本默认改成了 `vhr_layers=14`（见 `scripts/infer_instructblip.sh`），**这对 InstructBLIP 反而偏离论文**——论文 InstructBLIP 是 18 层、且**不含 layer1**（区别于 LLaVA 的"layer1 + 14 层"）。
- 注意：`experiment/A-VHR.md` 中"默认 `vhr_layers=14`，与论文一致"的说法对 InstructBLIP 不成立，仅对 LLaVA 系列成立。

### 3.4 根因三：最近一次 `vhr-h14` 跑未产出有效结果

- 日志 `logs_instructblip_vhr_gpu2.log` 末尾：`CHAIRs … nan`，且 POPE 评估抛 `TypeError: 'int' object is not iterable`。
- 说明该次推理 **没有生成完整的 5 seed CHAIR 文件 / 合并失败**，评测对空集合求统计得 `nan`，POPE 因 glob 无匹配触发 `eval.py` 的边界 bug。
- 结论：这次跑 **不能算作复现数据**，需要重跑。

### 3.5 小结：差距来源排序

1. **EOS 配置错误**（贡献最大，直接导致 Len×3 与 CHAIR 崩坏）。
2. **层数选择**（h14 vs 论文 h18、是否含 layer1）带来的方向性偏差。
3. **运行未跑完 / 合并失败**导致最新一组无有效数字。

### 3.6 ✅ 修复验证（2026-06-02 重跑）

按上述诊断修正后重跑 InstructBLIP VHR（`eos_token_id=2` + `vhr_layers=18` + `model_path` 指向真实权重，8× A6000 并行），结果与论文几乎完全一致：

| 指标 | 论文 VHR | 旧脏数据 | **修复后重跑** | 与论文偏差 |
|------|----------|----------|----------------|-----------|
| CHAIRs ↓ | 37.76±2.76 | 50.76 | **37.84±2.29** | +0.08 |
| CHAIRi ↓ | 9.75±0.98 | 11.99 | **9.72±0.79** | −0.03 |
| **Len** | **106.49** | 307.86 | **106.42** | −0.07 |
| POPE F1 | 85.52 | 85.43 | **85.47** | −0.05 |

**Len 从 307.86 回落到 106.42**，与论文 106.49 几乎重合，直接验证了「根因是生成停不下来」这一判断。至此三个模型的 VHR 主结果均已成功复现。

> 分 seed CHAIRs：34.0 / 36.8 / 38.4 / 39.4 / 40.6（seed 0–4）；均值 37.84。

---

## 4. 完整复现论文实验还需要做什么

### 4.1 ✅ 已完成：修正 InstructBLIP 配置并重跑 VHR

本节所述修复**已全部完成并验证通过**（结果见 §3.6）。具体改动：

**(a) eos 配置** — `models/instructblip-vicuna-7b/config.json` → `text_config.eos_token_id = 2` ✅

**(b) 作用层改回论文设定（18 层，不含 layer1）** — `scripts/infer_instructblip.sh`：`vhr_layers=14 → 18` ✅

**(c) 修复 model_path** — 原占位符 `/path/to/model_cache/...` 改为按脚本位置解析的真实权重路径 ✅

**(d) 重跑并评估** —
```bash
cd ~/VHR
bash scripts/infer_instructblip.sh vhr     # 5×CHAIR seeds + 3×POPE 子集
```

**(e) 验收**：CHAIRs 37.84 / CHAIRi 9.72 / Len 106.42 / POPE F1 85.47 — **达标，对齐论文 Table 1/2** ✅

### 4.2 论文主表完整性（还缺哪些表格 / 方法）

论文报告的是 **3 模型 × 多方法** 的完整对比；本仓库目前只跑了其中一部分：

| 论文内容 | 本地是否已有 | 待补 |
|----------|--------------|------|
| Table 1 CHAIR — InstructBLIP / LLaVA-1.5 / LLaVA-NeXT × {Greedy, Beam, DoLa, VCD, OPERA, CODE, EAH, VHR} | LLaVA-1.5 VHR、LLaVA-NeXT VHR/EAH、InstructBLIP Greedy/**VHR✅** | 各模型其余 baseline（Beam/DoLa/VCD/OPERA/CODE/EAH）若要完整复现主表 |
| Table 2 POPE F1 — 同上方法矩阵 | 同上 | 同上 |
| Table 3 LLaVA-Bench（GPT-4o 评分） | ❌ 未跑 | 需 `eval/llava_wild.py` + GPT-4 API key |
| Table 4 消融：adaptive head / outlier 去除（fixed VHR / outlier VHR） | ❌ 未跑 | 三模型各跑 fixed/outlier 两个变体 |
| Table 5 消融：layer1 vs 后 14 层 vs 两者（LLaVA-1.5） | ❌ 未跑 | LLaVA-1.5 跑三种层配置 |
| Table 6 消融：α ∈ {0.2,0.5,2,3,4}（LLaVA-1.5） | ❌ 未跑 | LLaVA-1.5 跑 5 个 α |

> 若目标只是"对齐论文主结果（VHR vs Greedy）"，只需完成 4.1 即可；若要"完整复现整篇论文"，需补齐上表所有 baseline 与消融。

### 4.3 ✅ 已完成：修复 `eval/eval.py` 的两个边界 bug

避免再次出现 `nan` / `TypeError` 掩盖真实失败（均已修复）：

1. **POPE 无匹配文件**：glob 为空时**显式报错退出**，并把平均改为按实际文件数 `/len(pope_files)`（不再写死 `/3`）。✅
2. **CHAIR 空集合**：0 个匹配文件时**直接报错提示"未找到结果文件"**，不再静默产出 `nan`。✅

### 4.4 跑数据前的环境与数据自检（来自 README）

- COCO：`data/coco/val2014` + `annotations/{captions,instances}_val2014.json`
- POPE：`data/pope/coco/coco_pope_{popular,adversarial,random}.json`
- LLaVA-Bench-W：`data/llava-bench-in-the-wild/{images, questions.jsonl}`（Table 3 才需要）
- 权重：三模型已在 `models/` 下；InstructBLIP `eos_token_id=2` 已就位。

### 4.5 与 A-VHR（本地新方法）的关系

- A-VHR 是 `experiment/A-VHR.md` 定义的本地扩展，**论文中没有**，因此无法直接与论文数字对比。
- A-VHR 的对照基线 **B0 = 原版 VHR**。因此 **必须先按 4.1 把 InstructBLIP 原版 VHR 跑对**，B0 才可信，A-VHR 的消融（A1–A6）才有意义。

---

## 5. 复现待办清单（可勾选）

- [x] 核对 `text_config.eos_token_id == 2`（已满足）
- [x] `scripts/infer_instructblip.sh`：`vhr_layers 14 → 18`（对齐论文 InstructBLIP）
- [x] 修复 `scripts/infer_instructblip.sh` 的 `model_path` 占位符 → 真实权重路径
- [x] 归档旧 `vhr-h18` 脏数据（→ `results/_archive_dirty_vhr_h18/`）
- [x] 重跑 `bash scripts/infer_instructblip.sh vhr`，验收通过（CHAIRs 37.84 / Len 106.42）✅
- [x] 修复 `eval/eval.py` 空匹配的 `nan` / `TypeError`
- [ ]（完整复现）补齐三模型其余 baseline（Beam/DoLa/VCD/OPERA/CODE/EAH）
- [ ]（完整复现）Table 3 LLaVA-Bench：`eval/llava_wild.py` + GPT-4 API
- [ ]（完整复现）Table 4/5/6 消融
- [ ] A-VHR：在 InstructBLIP B0 锁死后再跑 A1–A6

---

## 附录：数据复算命令

```bash
# LLaVA-1.5 VHR
python eval/eval.py "results/llava-1.5-7b-hf/chair_*-vhr-l1h14-512.json" --eval_chair
python eval/eval.py "results/llava-1.5-7b-hf/pope_*_0-vhr-l1h14-512.json" --eval_pope

# LLaVA-NeXT VHR
python eval/eval.py "results/llava-v1.6-vicuna-7b-hf/chair_*-vhr-l1h14-512.json" --eval_chair
python eval/eval.py "results/llava-v1.6-vicuna-7b-hf/pope_*_0-vhr-l1h14-512.json" --eval_pope

# InstructBLIP greedy
python eval/eval.py "results/instructblip-vicuna-7b/chair_*-greedy-512.json" --eval_chair
python eval/eval.py "results/instructblip-vicuna-7b/pope_*_0-greedy-512.json" --eval_pope

# InstructBLIP 旧 VHR(h18)（历史脏数据，仅供对照）
python eval/eval.py "results/home/nlp/VHR/models/instructblip-vicuna-7b/chair_*-vhr-h18-512.json" --eval_chair
python eval/eval.py "results/home/nlp/VHR/models/instructblip-vicuna-7b/pope_*_0-vhr-h18-512.json" --eval_pope
```
