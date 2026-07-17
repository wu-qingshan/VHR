# Disagreement-Gated SC-VHR 实验记录

> 开始日期：2026-07-17  
> 分支：`research/dg-scvhr`  
> 起始检查点：`106c134`（tag: `scvhr-llava-pooled-20260717`）  
> 状态：dev 已完成。DG 完全保留 unique correct coverage 并显著降低词级幻觉，但 CHAIRs 仅改善 1 点、未通过预注册门槛；validation 未运行，方法否决。

## 1. 问题与已有证据

标准 SC-VHR 的双边外推在三套互斥 LLaVA 划分共 300 图上将 CHAIRs 38.67 降至 28.00，但 unique correct objects 下降约 3.2%。Positive-Delta SC-VHR 完全删除负 delta 后恢复覆盖，却同时完全失去 hallucination benefit，证明对 base 偏好 token 的负向抑制是核心有效成分。

下一问题不是“是否保留负向抑制”，而是“何时需要负向抑制”。标准 SC 在 VHR 与 base 已经选择同一 greedy token 时仍会进行 logit 外推和 cutoff；这些无决策分歧步骤可能造成不必要的序列漂移。

## 2. 方法

对每个生成步骤，先计算：

$$
y_{VHR}=\arg\max \ell_{VHR},\qquad y_{base}=\arg\max \ell_{base}
$$

Disagreement-Gated SC-VHR（DG-SC-VHR）定义为：

$$
\ell^{DG}=\begin{cases}
(1+\alpha)\ell_{VHR}-\alpha\ell_{base}, & y_{VHR}\neq y_{base}\\
\ell_{VHR}, & y_{VHR}=y_{base}
\end{cases}
$$

- 分歧时：使用标准 SC 双边公式与原 plausibility cutoff；
- 一致时：完整回退原 VHR logits，不使用 contrast，也不使用 cutoff；
- 不引入连续阈值或可调参数。

每张图记录 `contrast_steps/total_steps`，用于分析路由频率；该统计不参与 token 选择。

## 3. 冻结配置与数据

- 模型：`models/llava-1.5-7b-hf`
- layer 1 + 最后 14 层，magnitude top-50%，ratio 2，原 filter
- `alpha=0.5`, `beta=0.1`，greedy，`max_new_tokens=512`
- dev：`experiment/pdscvhr_dev_chair_ids.json`。该集合已用于发现 PD 失败，因此属于方法开发集；已有同图 VHR 和标准 SC 输出可直接作为冻结对照。
- validation：`experiment/pdscvhr_validation_chair_ids.json`。该集合尚未生成过任何方法输出，在 dev gate 判定前继续锁定。
- 不扫描 alpha、beta、路由阈值、top-k、head fraction 或层数。

## 4. 预注册 dev gate

同一 dev 上已知冻结参照：VHR 为 39.0/11.45/80.08、681 mentions；标准 SC 为 28.0/8.86/76.33、587 mentions。

DG 必须同时满足：

1. CHAIRs ≤32.0，即相对 VHR 至少改善 7 点，且相对标准 SC 最多回退 4 点；
2. CHAIRi ≤10.5；
3. total generated mentions ≥ VHR 的 90%（至少 613）；
4. unique correct objects ≥ VHR 的 97%；
5. Len 保持在 VHR ±7%；
6. repeated mention rate 不高于 VHR 3 个百分点；
7. 100 图完整、ID 顺序精确、无空回答；
8. 不要求特定路由比例，但必须完整报告 contrast step rate。

若任一质量门槛失败：停止 validation，不调整任何参数。若全部通过：才在锁定 validation 上一次性运行 VHR、标准 SC、DG，并使用相同门槛独立判定。

## 5. 输出与回滚

- 烟雾：`dgscvhr_smoke_results/`
- dev：`dgscvhr_dev_results/`
- validation：`dgscvhr_validation_results/`
- 所有目录均不覆盖历史输出且保持 Git 未跟踪。
- 回滚到 `scvhr-llava-pooled-20260717` 可完全移除 DG 实现。

## 6. 执行日志

- 2026-07-17：建立 `research/dg-scvhr` 分支并完成上述方法、数据、gate 和停止规则预注册；此时尚未实现或运行 DG。
- 2026-07-17：以独立 `vhr_disagreement_gate` generation flag 实现路由；新增逐图 `routing_stats` 和单次加载 runner `scripts/run_llava15_dgscvhr.py`。标准 SC、PD、VCD 路径不启用该 flag。
- 2026-07-17：张量路由测试通过；新 dev 第一图 64-token 烟雾测试成功，10/64 步触发 contrast（15.625%），结果 JSON 的路由计数完整。烟雾输出不参与指标。
- 2026-07-17：DG dev 100 图完整运行，耗时 920 秒。文件为 100 行、ID 与固定顺序完全一致、无空回答，所有路由计数合法。

## 7. Dev 结果

| 方法 | CHAIRs ↓ | CHAIRi ↓ | Len | generated mentions | hallucinated mentions |
|---|---:|---:|---:|---:|---:|
| VHR | 39.0 | 11.45 | 80.08 | 681 | 78 |
| 标准 SC-VHR | **28.0** | **8.86** | 76.33 | 587 | **52** |
| DG-SC-VHR | 38.0 | 10.06 | 80.52 | 666 | 67 |

路由统计：

- pooled contrast steps：1199/10459，**11.46%**；
- 每图 contrast rate 均值 11.46%，中位数 11.11%；
- p10/p90：6.86% / 16.28%；范围 2.90%–20.00%。

逐图配对 bootstrap（20,000 次）：

- DG vs VHR：CHAIRs 差值 -0.98，95% CI `[-5,3]`；CHAIRi 差值 -1.39，95% CI `[-2.83,-0.16]`；修复 3、新增 2。
- DG vs 标准 SC：CHAIRs 差值 +10.02，95% CI `[3,18]`；DG 显著退化。

## 8. 预注册 gate 判定

| 门槛 | DG 结果 | 判定 |
|---|---:|---|
| CHAIRs ≤32 | 38.0 | **失败** |
| CHAIRi ≤10.5 | 10.06 | 通过 |
| total mentions ≥90% VHR | 97.80% | 通过 |
| unique correct ≥97% VHR | **100.00%** | 通过 |
| Len 在 VHR ±7% | +0.55% | 通过 |
| repeated rate 最大 +3pp | -0.34pp | 通过 |
| 完整性与路由统计 | 完整 | 通过 |

总判定：**失败**。DG 取得词级幻觉减少且完全保留唯一正确对象，但句级 CHAIRs 没有可靠改善，不能作为标准 SC 的替代。锁定 validation 不运行，且不扫描路由阈值或 top-k。

## 9. 机制结论

1. VHR/base 原始 argmax 分歧只占约 11.5% 生成步骤；只在这些步骤应用 contrast 不足以复现标准 SC 的句级收益。
2. 标准 SC 的关键行为应大量发生在：VHR 与 base argmax 相同，但双边外推后另一个 token 成为 argmax。即有效信号不仅是分支决策分歧，也包括完整 logit 几何中的相对斜率。
3. DG 将 unique correct objects 从 224 保持到 224，证明无分歧步骤确实贡献了覆盖扰动；但同时也贡献了绝大多数 hallucination reduction。
4. 下一步不应继续做二元“开/关 contrast”路由。更有价值的是对标准 SC 引发的 token switch 做诊断，区分同-argmax switch 中哪些对应幻觉修复，随后设计带置信边界的 suppression cap。
