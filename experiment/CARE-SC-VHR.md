# CARE-SC：反事实对齐、响应路由与稳定性感知的自对比解码

> 开始日期：2026-07-17
> 分支：`research/care-scvhr`
> 起始检查点：`7e1434c`（tag: `rrscvhr-efficiency-quality-20260717`）
> 状态：方法、隔离划分和实现均已完成；纯函数测试、开放描述烟雾与 binary 回退等价验证通过，尚未运行100图 dev。

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
