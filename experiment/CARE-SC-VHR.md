# CARE-SC：反事实对齐、响应路由与稳定性感知的自对比解码

> 开始日期：2026-07-17
> 分支：`research/care-scvhr`
> 起始检查点：`7e1434c`（tag: `rrscvhr-efficiency-quality-20260717`）
> 状态：100图 dev 已完成。CARE 获得较好的 CHAIRi、对象覆盖和重复 Pareto 点，但 CHAIRs 改善不足且不显著，未通过预注册门槛；locked validation 与 reserved test 不运行，不扫描参数。

## 1. 研究动机

现有证据同时支持和限制标准 SC-VHR：

1. InstructBLIP 正式 2500 图上，SC 将 CHAIRs/CHAIRi 从 37.84/9.72 降至 30.40/8.42，且逐图显著；
2. LLaVA-1.5 三套互斥划分 pooled 300 图上，CHAIRs/CHAIRi 从 38.67/12.66 降至 28.00/9.25，但 unique correct objects 下降 3.21%；
3. Positive-Delta SC 删除所有负向修正后完全失去幻觉收益，说明对语言先验 token 的负向外推是必要成分；
4. VEG-SC 只检查最终 SC 候选相对 VHR 候选的 text-only support，虽然恢复部分覆盖，但不能可靠阻止新增幻觉；
5. 标准 SC 在 POPE 上产生保守偏置，并在正式 caption 中提高局部重复和 exact cycle。

核心问题不是“是否进行负向修正”，而是：**VHR 干预方向是否与独立的视觉反事实证据一致**。候选级硬 gate 信息过少；应在整个词表上分解干预效应和视觉证据，只保留方向一致的修正。

## 2. 方法定位

暂定名称：**CARE-SC**（Counterfactual-Aligned, Response-routed and stability-aware Evidence Self-Contrast）。

CARE-SC 不把 VHR 视为唯一可用干预，而将其作为第一种实例。若核心公式成立，后续应将 intervention 分支替换为其他内部视觉增强方法，验证方法级通用性。

三个同步分支共享同一生成历史：

- intervention 分支 $I$：启用 VHR；
- base 分支 $B$：同图像、关闭 VHR；
- text-only 分支 $T$：移除图像、关闭 VHR。

记下一 token logits 为 $\ell_I,\ell_B,\ell_T$。

### 2.1 反事实对齐修正

干预效应：

$$
d_I(y)=\ell_I(y)-\ell_B(y)
$$

视觉证据：

$$
d_V(y)=\ell_B(y)-\ell_T(y)
$$

只保留符号一致的修正：

$$
d_A(y)=d_I(y)\,\mathbf{1}[d_I(y)d_V(y)>0]
$$

最终 logits：

$$
\ell_{CA}(y)=\ell_I(y)+\alpha d_A(y)
$$

解释：

- $d_I>0,d_V>0$：VHR 提升且图像相对 text-only 支持该 token，保留正向增强；
- $d_I<0,d_V<0$：VHR 抑制且该 token 更像 text-only 语言先验，保留负向抑制；
- 符号相反：内部干预与独立视觉证据冲突，不外推该 token，回到 intervention logits。

该规则不同于 PD 的“只保留正 delta”，也不同于 VEG 的“只检查最终候选”。它没有新增连续参数，并在整个可行词表上执行干预—证据对齐。

固定沿用标准 SC 的 `alpha=0.5` 和 intervention-logit plausibility cutoff `beta=0.1`，不扫描。

### 2.2 响应形式路由

- 开放描述及非二元问题：使用反事实对齐自对比；
- binary 问题：完整回退原 VHR，不建立 base/text-only 分支。

这继承 RR-SC 已验证的 POPE 输出、F1 与单分支效率，不重新利用 POPE 调参。

### 2.3 干预诱发稳定性保护

仅当 CARE 候选与 VHR 候选不同，且 CARE 候选会形成新的 exact consecutive cycle、而 VHR 候选不会时，当前步回退 VHR logits。

固定 cycle 定义：生成 token ID block 周期不超过16，追加候选后连续出现至少3次。该规则对应正式质量诊断中预先定义的 exact-cycle 指标，不扫描 n-gram size、penalty 或周期。

稳定性保护不是全局 `no_repeat_ngram_size`：如果 VHR 本身也会形成相同循环，则不声称该问题由自对比引入，也不在本组件中修改。

## 3. 方法变体与必要消融

| 名称 | 反事实对齐 | 响应路由 | 稳定性保护 |
|---|---:|---:|---:|
| VHR | 否 | 否 | 否 |
| SC-VHR | 否，完整双边 delta | 否 | 否 |
| CA-SC | 是 | binary 回退 VHR | 否 |
| CARE-SC | 是 | binary 回退 VHR | 是 |

VEG、PD、DG 已有冻结负结果，不在新划分上重复调参。正式论文中作为机制消融引用原冻结实验。

## 4. 数据隔离

创建三个互斥的新 COCO-val 划分，并排除仓库中所有历史：

- 固定 ID 文件；
- CHAIR 生成输出中的 question ID；
- dev、validation、official、smoke 和诊断输出中的 question ID。

计划划分：

- CARE dev：100 图，仅用于首次方法判定；
- CARE locked validation：100 图，dev 通过后才允许运行；
- CARE reserved test：500 图，所有方法和参数冻结后才允许运行。

划分脚本、seed、候选池大小、排除 ID 数量和 SHA256 必须记录。结果目录全部新建并拒绝覆盖。

## 5. 首阶段模型选择

首阶段使用 LLaVA-1.5-7B，而不是继续在正式 InstructBLIP 测试集上开发，原因：

1. LLaVA 是当前 hallucination decoding 文献最常见的可比 backbone；
2. 标准 SC 已在三套 LLaVA 划分上显著降低幻觉，但存在覆盖代价，正好检验 CARE 假设；
3. 新方法不能根据已使用的 InstructBLIP 正式结果调参。

若 LLaVA dev/validation 通过，再移植到 Qwen2.5-VL-7B 或 Qwen3-VL-8B；InstructBLIP 只作为冻结跨架构复核。

## 6. Dev 预注册门槛

所有比较使用同图配对统计。CA-SC 和 CARE-SC 不允许分别扫描配置；完整 CARE 为主方法，CA 为组件消融。

相对 VHR，CARE 必须同时满足：

1. CHAIRs 至少绝对下降 8 点；
2. CHAIRi 至少绝对下降 2 点；
3. paired CHAIRs bootstrap 95% CI 上界不高于 -1；
4. unique correct objects 保留率至少 98%；
5. 平均长度变化在 ±5%；
6. exact cycle 命中数不高于 VHR。

相对标准 SC，CARE 必须满足以下 Pareto 条件：

1. CHAIRs 最多回退 2 点；
2. CHAIRi 最多回退 0.5 点；
3. unique correct objects 不低于 SC，并至少恢复 SC 相对 VHR 损失的一半；
4. per-caption repeated 3/4-gram 均不高于 SC；
5. exact cycle 不高于 SC。

若标准 SC 在新 dev 没有重现相对 VHR 的至少3点 CHAIRs 改善，则该划分不用于宣称 CARE 超越 SC，但仍记录，不更换 ID。

## 7. 停止规则

1. CARE 未通过任一核心 CHAIR/coverage 门槛：记为负结果，不运行 locked validation；
2. 不扫描 `alpha/beta`、符号阈值、cycle period、重复次数或其他连续插值；
3. CA 优于 CARE 时，只能说明稳定性组件无益，不能删除稳定性组件后在同一 dev 继续调规则；
4. locked validation 只在 dev 全部门槛通过后运行，且运行前提交代码与标签；
5. reserved test 不用于任何方法选择；
6. 正式 InstructBLIP 2500图只作为历史比较，禁止用于 CARE 参数选择。

## 8. 高质量论文所需的后续证据

即使首阶段通过，也不能仅凭 CARE-VHR 宣称完成。投稿前至少需要：

1. InstructBLIP、LLaVA-1.5、一个 Qwen/InternVL 现代 backbone；
2. CHAIR、POPE、AMBER，以及 HallusionBench/MMHal-Bench 至少一个；
3. 与 VCD、VHR、SADI/VISTA、TLVS/FADE 中可复现方法的统一协议比较；
4. correct-object coverage、文本稳定性、一般能力、时延和显存；
5. 将 CARE 应用于至少一种非 VHR 内部干预，证明公式不是 VHR 专用技巧。

## 9. 输出与回滚约束

- 新输出前缀：`care_*_results/`；
- 不修改或覆盖 `results/`、`scvhr_*`、`rrscvhr_*`、`vegscvhr_*` 等历史目录；
- 核心实现使用独立 generation flags，默认关闭；
- 新方法失败时保留实现、记录和负结果，但默认 VHR/SC/RR 路径必须逐字不变；
- 起始恢复点：`rrscvhr-efficiency-quality-20260717`。

## 10. 执行日志

- 2026-07-17：完成 CARE-SC 公式、响应路由、稳定性保护、消融矩阵、dev 门槛和停止规则预注册。
- 2026-07-17：新增 `scripts/create_care_chair_splits.py`，扫描 265 个历史 CHAIR ID/结果文件，其中191个包含可解析 image/question ID；共排除历史唯一 ID 2852 个。
- 2026-07-17：从剩余 37,652 个 COCO-val 图像中以固定 seed `20260717` 生成 dev/validation/test=100/100/500，三套之间及与历史集合均零重叠。
- 2026-07-17：划分 SHA256：dev `86850d50...ae32e`，validation `e78a04a0...964d4`，test `5031aede...196ff6`；完整来源和每个历史文件 hash 见 `experiment/care_chair_splits_manifest.json`。
- 2026-07-17：新增独立 generation flags `vhr_counterfactual_alignment` 和 `vhr_stability_guard`；默认关闭，不改变 VHR/SC/RR 路径。新增三分支全词表符号对齐、逐 token exact-cycle guard 和诊断统计。
- 2026-07-17：三个纯函数测试通过：正/负符号对齐、exact-cycle candidate 检测、仅对 CARE 独有循环回退。
- 2026-07-17：LLaVA CARE dev 首图 `33532` 的 VHR/SC/CA/CARE 四方法烟雾全部成功。CA 与 CARE 文本一致，aligned plausible token fraction=51.38%，其中负向对齐占67.69%，保留约50.24%的 intervention absolute delta；本样本未触发 cycle guard。
- 2026-07-17：用旧 VEG runner 重新生成同一图的 VHR/SC 控制输出，`question_id/text/gt/ans_conf/question` 与 CARE runner 逐字段一致，证明新 flags 未污染标准路径。
- 2026-07-17：POPE popular 前10题 binary smoke 中，CARE 与 VHR 10/10 的 ID、文本、标签、confidence、问题逐字段一致；route 全为 `vhr`，没有建立 CARE 诊断字段。
- 2026-07-17：新增冻结 dev evaluator `scripts/evaluate_care_dev.py`，在结果完成后一次性计算 CHAIR、对象覆盖、distinct、重复、cycle、配对 bootstrap/McNemar 和全部预注册 gate；输出拒绝覆盖，避免手工选择统计口径。
- 2026-07-17：四方法在4张独立 GPU 上完成 CARE dev 100图。生成时间：VHR 389秒、SC 734秒、CA 982秒、CARE 986秒；结果均为100行且输出目录独立。
- 2026-07-17：冻结 evaluator 首次误用默认 Python，因缺少 NLTK `punkt_tab` 在创建分析文件前退出；随后使用正式 `vhr` 环境原样重跑成功，未修改脚本、门槛或结果。

## 11. CARE dev 结果

| 方法 | CHAIRs ↓ | CHAIRi ↓ | Len | mentions | hallucinated | unique correct | repeated-3 ↓ | repeated-4 ↓ | cycles |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VHR | 32.0 | 10.39 | 79.75 | 635 | 66 | 205 | 0.01498 | 0.00592 | 0 |
| SC | 32.0 | 9.43 | 77.68 | 583 | 55 | 208 | 0.01666 | 0.00650 | 0 |
| CA-SC | **30.0** | **7.68** | 76.27 | 586 | **45** | **211** | **0.01476** | **0.00587** | 0 |
| CARE-SC | **30.0** | **7.68** | 76.27 | 586 | **45** | **211** | **0.01476** | **0.00587** | 0 |

### 11.1 配对统计

CARE 相对 VHR：

- CHAIRs：-2.0，bootstrap 95% CI `[-11, 7]`，修复/新增=11/9，McNemar `p=0.824`；
- CHAIRi：-2.72，95% CI `[-6.03, 0.38]`；
- Len：-3.48词，95% CI `[-6.16,-0.85]`，相对变化 -4.36%；
- hallucinated mentions：66→45，下降31.82%；
- unique correct objects：205→211，保留率102.93%；
- repeated 3/4-gram 均略低于 VHR。

CARE 相对标准 SC：

- CHAIRs：-2.0，95% CI `[-10,6]`，修复/新增=10/8，McNemar `p=0.815`；
- CHAIRi：-1.75，95% CI `[-4.44,0.77]`；
- unique correct objects：208→211；
- repeated 3/4-gram 均改善。

标准 SC 在该固定新 dev 上没有重现历史 CHAIRs 优势：VHR 与 SC 均为32.0，修复/新增=10/10，McNemar `p=1.0`。因此本划分不能支持“CARE 在稳定 SC 收益上进一步提升”的强结论，但按预注册不更换 ID。

### 11.2 机制诊断

- 平均每图 plausible tokens 319.78，其中47.61%通过符号对齐；
- 对齐 token 中71.20%为负向对齐，说明保留“图像也不支持”的负 delta 仍是主要修正来源；
- 保留 intervention absolute delta 的47.08%；
- CA 与 CARE 100/100 文本及全部指标完全一致；
- stability guard 0次触发，因为该100图中CA和VHR均没有形成预定义 exact cycle。

## 12. 门槛判定

相对 VHR：CHAIRi、coverage、长度和cycle门槛通过；CHAIRs 至少下降8点及其CI门槛失败。

相对 SC：全部 Pareto 描述性门槛通过，但核心结果不显著，且 SC 自身未在新划分重现 CHAIRs 改善。

总判定：**失败，不进入 locked validation**。不扫描 `alpha/beta`、符号阈值或软mask；不运行 `experiment/care_validation_chair_ids.json` 和 reserved test。

## 13. 方法评价与下一方向

全词表符号对齐比候选级 VEG 更合理：它在点估计上同时降低 CHAIRs/CHAIRi、减少31.8%幻觉 mentions，并增加 unique correct objects。但收益规模和显著性不足以缩小与最新方法的差距，三分支成本又达到 VHR 的约2.53倍。

这说明纯 logits 级反事实仍缺少“对象在图像哪个区域被支持”的定位证据。下一方向停止词表 delta/gate 变体，转向 **head-guided active visual verification**：利用视觉敏感 head 定位与候选实体相关的图像区域，只在高风险实体 token 上进行局部重看和验证。该方向需要在新的方法划分上重新预注册，不能使用 CARE locked validation 调整。
