# Visual-Evidence-Gated SC-VHR 实验记录

> 开始日期：2026-07-17  
> 分支：`research/veg-scvhr`  
> 起始检查点：`c480ed8`（tag: `scvhr-switch-diagnostic-complete-20260717`）  
> 状态：dev 已完成。VEG 恢复覆盖并取得中间 Pareto 点，但 CHAIRs 未过预注册门槛且改善不显著；validation 未运行，不扫描视觉 margin，方法不作为主方案。

## 1. 动机

标准 SC-VHR 在 LLaVA 上显著降低幻觉，但 pooled unique correct objects 下降约 3.2%。PD 和 DG 两条简单路线已被否决：

- 删除全部负 delta：覆盖恢复，但幻觉收益完全消失；
- 仅在 VHR/base argmax 分歧时启用：覆盖完全恢复，但 CHAIRs 基本不改善。

100 图逐 token 诊断表明，标准 SC 的 584 次真实 token switch 中 508 次（86.99%）发生在两分支 argmax 原本相同的步骤。switch rate、两分支 delta 和 score margin 不能可靠区分修复与持续幻觉。因此需要独立于 VHR/base self-contrast 的视觉证据信号。

## 2. 三分支方法

每步维护：

1. VHR-image 分支，logits $\ell_{VHR}$；
2. base-image 分支，logits $\ell_{base}$；
3. text-only 分支，logits $\ell_{text}$。

标准 SC 先提出候选：

$$
\ell_{SC}=(1+\alpha)\ell_{VHR}-\alpha\ell_{base}
$$

记 $y_{VHR}=\arg\max\ell_{VHR}$，$y_{SC}=\arg\max\ell_{SC}$。定义 token 的在线视觉支持：

$$
g(y)=\ell_{base}(y)-\ell_{text}(y)
$$

VEG-SC-VHR 使用相对支持、无连续阈值：

$$
\ell_{VEG}=\begin{cases}
\ell_{SC}, & y_{SC}\neq y_{VHR}\ \text{且}\ g(y_{SC})\ge g(y_{VHR})\\
\ell_{VHR}, & \text{otherwise}
\end{cases}
$$

- 允许 switch 时沿用标准 SC plausibility cutoff；
- 阻止 switch 或标准 SC 不改变 top-1 时，完整回退 VHR logits且不使用 cutoff；
- 不增加 margin、温度或 top-k 参数。

## 3. 工程可行性

现有 VHR 在首步已经运行 text-only prefill 以选择 heads。该 forward 的 KV cache 当前被丢弃；VEG 将其保留并延续：

- text-only prompt 与图像 token 删除逻辑保持不变；
- 首步直接复用已有 text-only prefill logits/cache，不增加重复 prefill；
- 后续三分支追加完全相同的生成 token；
- text-only 和 base 分支均使用 `vhr_phase="vanilla"`，解码时不应用 VHR head 增强；
- 三套 KV cache 完全独立，不共享 mutable cache；VHR head selection state 只由原 text/clean prefill建立。

预计成本：首步仍为原 VHR 的 text/clean prefill加base-image prefill；逐 token 从标准 SC 的两次 forward 增至三次，时延约为 VHR 的 3×、标准 SC 的 1.5×。

## 4. 数据与冻结配置

- 模型和 VHR：LLaVA-1.5，layer 1 + last 14，magnitude top-50%，ratio 2，原 filter
- `alpha=0.5`, `beta=0.1`，greedy，`max_new_tokens=512`
- dev：`experiment/pdscvhr_dev_chair_ids.json`，已有冻结 VHR/SC 对照
- validation：`experiment/pdscvhr_validation_chair_ids.json`，仍未生成过任何方法输出；dev 通过前锁定
- 不扫描视觉支持 margin、alpha、beta、head fraction、layer 或 cutoff

## 5. 预注册 dev gate

冻结参照：VHR 39.0/11.45/80.08、681 mentions、224 unique correct；标准 SC 28.0/8.86/76.33、587 mentions、221 unique correct。

VEG 必须全部满足：

1. CHAIRs ≤31.0，相对 VHR 至少改善 8 点、相对标准 SC最多回退 3 点；
2. CHAIRi ≤9.8；
3. total generated mentions ≥ VHR 的 90%（至少 613）；
4. unique correct objects ≥ VHR 的 98%；
5. Len 在 VHR ±7%；
6. repeated mention rate 不高于 VHR 3pp；
7. 至少允许 1 个并阻止 1 个标准 SC switch，排除退化为纯 VHR或纯 SC；
8. 100 图完整、ID顺序精确、无空回答、三分支步数同步。

任一门槛失败：停止 validation，不扫描任何参数。全部通过：才在锁定 validation 一次性运行 VHR、标准 SC 和 VEG，并以相同门槛独立判定。

## 6. 输出与回滚

- 烟雾：`vegscvhr_smoke_results/`
- dev：`vegscvhr_dev_results/`
- validation：`vegscvhr_validation_results/`
- 结果目录均保持 Git 未跟踪且不覆盖历史文件。
- 回滚到 `scvhr-switch-diagnostic-complete-20260717` 可完全移除 VEG 实现。

## 7. 执行日志

- 2026-07-17：完成 text-only cache、phase 和三分支历史同步审计；确认可复用首步 text-only prefill，后续以独立 cache 延续。
- 2026-07-17：完成上述参数免费视觉支持 gate、数据、门槛和停止规则预注册；此时尚未实现 VEG。
- 2026-07-17：实现独立 `vhr_visual_evidence_gate`、三分支 cache 延续、视觉支持 gate 统计和 `scripts/run_llava15_vegscvhr.py`。标准 SC/DG/PD 默认不保留 text-only 解码 cache。
- 2026-07-17：64-token 烟雾测试通过，2 次 switch 提议中允许/阻止各 1；完整 512-token 流程在 77 步正常 EOS，4 次提议中允许 3、阻止 1。三分支 cache 无形状冲突，计数守恒，允许进入 dev。
- 2026-07-17：VEG dev 100 图完整运行，耗时 1055 秒；100 行、ID 顺序、空文本和三分支计数全部通过完整性检查。

## 8. Dev 结果

| 方法 | CHAIRs ↓ | CHAIRi ↓ | Len | generated mentions | unique correct | hallucinated mentions |
|---|---:|---:|---:|---:|---:|---:|
| VHR | 39.0 | 11.45 | 80.08 | 681 | 224 | 78 |
| 标准 SC-VHR | **28.0** | **8.86** | 76.33 | 587 | 221 | **52** |
| VEG-SC-VHR | 33.0 | 9.53 | 79.85 | 640 | 220 | 61 |

VEG 路由：

- total steps：10383；标准 SC switch 提议 582（5.61%）；
- 允许 298、阻止 284，允许比例 51.20%；
- 99/100 图至少有一次 switch 提议；每图允许比例中位数 50%。

运行成本：VEG 1055 秒，标准 SC 同集 868 秒，VEG 慢 **21.5%**；相对 VHR 458 秒为 2.30×。

逐图配对：

- VEG vs VHR：CHAIRs -5.98，95% CI `[-14,2]`；CHAIRi -1.91，95% CI `[-4.53,0.64]`；修复 11、新增 5。
- VEG vs 标准 SC：CHAIRs +5.02，95% CI `[-1,11]`；标准 SC→VEG 修复 2、新增 7。
- unique correct 保留 98.21%，total mentions 保留 93.98%，Len -0.29%；覆盖门槛均通过。

## 9. 预注册 gate 判定

| 门槛 | VEG 结果 | 判定 |
|---|---:|---|
| CHAIRs ≤31 | 33.0 | **失败** |
| CHAIRi ≤9.8 | 9.53 | 通过 |
| total mentions ≥90% | 93.98% | 通过 |
| unique correct ≥98% | 98.21% | 通过 |
| Len ±7% | -0.29% | 通过 |
| repeated rate 最大 +3pp | -0.48pp | 通过 |
| 允许且阻止 switch | 298 / 284 | 通过 |
| 完整性与三分支同步 | 通过 | 通过 |

总判定：**失败**。VEG 是 VHR 与标准 SC 之间的可解释 Pareto 点，但核心 CHAIRs 未达预注册门槛，且 100 图 paired CI 跨 0；不能进入锁定 validation，也不能声称可靠优于 VHR。

## 10. 视觉证据信号分析

按 VHR→VEG CHAIRs 转移：保持正确/新增/修复/持续幻觉为 56/5/11/28。

- 修复图的 proposed switch rate 比持续幻觉高 1.76pp，95% CI `[0.15pp,3.57pp]`；
- 修复图 allowed support margin 均值 1.43，持续幻觉 0.77，差值 CI `[0.16,1.29]`；视觉支持含有正确信号；
- 但新增幻觉与保持正确在 proposed/allowed rate、allow fraction、allowed margin 上均无可靠分离；
- 因此零阈值视觉 gate 能恢复覆盖，却不能充分阻止错误 switch。事后调高 margin 属于参数扫描，按规则不进行。

## 11. 决策

- VEG 保留为“独立视觉证据有信号但单一相对 gate 不足”的负/部分正消融；
- 不运行 `experiment/pdscvhr_validation_chair_ids.json`；
- 不扫描 margin、温度或加权函数；
- 标准 SC 仍是开放描述主方法。下一优化转向 response-type routing：开放描述使用 SC，二元存在性问题完整回退原 VHR，以解决已知 POPE recall/F1 回归并降低二元任务时延。
