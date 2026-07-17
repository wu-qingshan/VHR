# Disagreement-Gated SC-VHR 实验记录

> 开始日期：2026-07-17  
> 分支：`research/dg-scvhr`  
> 起始检查点：`106c134`（tag: `scvhr-llava-pooled-20260717`）  
> 状态：方法与门槛已预注册，尚未实现或生成 DG 输出。

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
