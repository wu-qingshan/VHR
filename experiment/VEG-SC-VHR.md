# Visual-Evidence-Gated SC-VHR 实验记录

> 开始日期：2026-07-17  
> 分支：`research/veg-scvhr`  
> 起始检查点：`c480ed8`（tag: `scvhr-switch-diagnostic-complete-20260717`）  
> 状态：工程可行性、公式和门槛已预注册，尚未实现或生成 VEG 输出。

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
