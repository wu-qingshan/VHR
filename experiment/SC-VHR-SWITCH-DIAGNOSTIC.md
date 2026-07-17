# SC-VHR Token-Switch 机制诊断

> 日期：2026-07-17  
> 分支：`research/scvhr-switch-diagnostic`  
> 起始检查点：`fe35046`（tag: `dgscvhr-dev-negative-20260717`）

## 目的

PD 与 DG 两个负结果表明：标准 SC 的收益依赖负向外推，而且不能只在 VHR/base argmax 分歧时启用。当前需要定位标准 SC 实际改变 token 的事件类型，而不是继续猜测新公式。

诊断重跑冻结标准 SC-VHR，不修改任何 logits、cutoff、token selection、head 或 cache，仅记录：

- `branch_disagreement_steps`：VHR/base 原始 argmax 不同；
- `selected_switch_steps`：标准 SC 最终 token 不同于 VHR argmax；
- `same_argmax_switch_steps`：VHR/base argmax 相同，但外推后最终 token 改变；
- `disagreement_switch_steps` 与 `disagreement_no_switch_steps`；
- same-argmax switch 中，最终 token 与原 VHR top token 的 delta 均值和 contrast score margin。

## 完整性要求

1. 使用 `experiment/pdscvhr_dev_chair_ids.json` 和冻结 LLaVA 配置；
2. 诊断输出的 100 条 caption 必须与 `pdscvhr_dev_results/.../0-chair_0-scvhr-unseen-a05-b01-512.json` 逐字一致；
3. 任一 caption 不一致即视为诊断实现污染生成，不使用统计并回滚；
4. 本阶段不运行 validation，不提出或测试新超参数。

## 执行日志

- 2026-07-17：新增独立 `vhr_switch_diagnostics` flag、逐图统计和 `scripts/run_llava15_scvhr_switch_diagnostic.py`；flag 不参与 logits 计算。
- 2026-07-17：第一张图完整 512-token 重跑与已有标准 SC 的 question ID、caption、ans_conf 逐字一致。该图 86 步中 VHR/base argmax 分歧 10 次，但分歧步骤没有 token switch；2 次最终 switch 全部发生在两分支 argmax 相同的步骤。允许进入 100 图诊断。
- 2026-07-17：100 图诊断重跑完成，耗时 831 秒。与历史标准 SC 的 100 个 question ID、caption、ans_conf 全部逐字一致，统计实现未污染生成路径。

## 诊断结果

| 事件 | 步数 | 占全部 9955 步 |
|---|---:|---:|
| VHR/base argmax 分歧 | 1135 | 11.40% |
| 标准 SC 最终 token 不同于 VHR | 584 | 5.87% |
| 同 argmax 条件下发生 switch | 508 | 5.10% |
| 分歧条件下发生 switch | 76 | 0.76% |
| 分歧但最终不 switch | 1059 | 10.64% |

关键比例：

- 所有实际 token switch 中，508/584（**86.99%**）发生在 VHR/base argmax 原本相同的步骤；
- 1135 个 branch-disagreement 步骤中只有 76（6.70%）真正改变了 VHR token；
- 因此 DG 同时漏掉约 87% 的真实 switch，并在大量最终不改变 token 的分歧步骤上启用 contrast。

按 VHR→标准 SC 的逐图 CHAIRs 转移分组：

| 转移 | 图数 | selected switch rate | same-argmax switch rate | selected switches/图 |
|---|---:|---:|---:|---:|
| 00 保持正确 | 58 | 5.78% | 5.58% | 5.24 |
| 01 新增幻觉 | 3 | 4.82% | 4.70% | 5.33 |
| 10 修复 | 14 | 6.92% | 6.78% | 7.00 |
| 11 持续幻觉 | 25 | 5.64% | 5.81% | 6.64 |

修复图的 switch rate 比持续幻觉高约 1.27pp，但 20,000 次 bootstrap 95% CI `[-0.33pp,3.25pp]`，跨 0。same-argmax switch rate、selected token delta 和 score margin 的修复/持续幻觉差异也全部跨 0。新增幻觉只有 3 图，不足以据此选择阈值。

## 机制判断与下一候选

1. 标准 SC 的关键不是检测两分支 argmax 是否不同，而是利用完整 logit slope 将原本非 top-1 的 token 推到首位。
2. 仅用 switch frequency、selected delta 或 score margin 做标量阈值，现有数据不能可靠区分 hallucination repair 与 persistent/error introduction；继续扫描阈值容易过拟合。
3. 下一候选应引入独立的视觉证据信号，而不是继续变换同一对 logits：维护在线 text-only 分支，计算 base-image 相对 text-only 的 token visual support。
4. 对标准 SC 提议的 switch，仅当候选 token 的视觉支持不低于原 VHR top token 时允许；否则回退 VHR。该方法暂称 Visual-Evidence-Gated SC-VHR（VEG-SC-VHR）。
5. VEG 需要第三套 KV cache 和每步第三次 forward，初始成本约为 VHR 的 3×；若质量-覆盖 Pareto 通过，再研究共享前缀与 cache 优化。本诊断阶段不实现或运行 VEG。
