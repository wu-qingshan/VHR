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
