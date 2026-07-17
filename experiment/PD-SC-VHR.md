# Positive-Delta SC-VHR 实验记录

> 开始日期：2026-07-17  
> 分支：`research/pd-scvhr`  
> 起始检查点：`bf74d4f`（tag: `scvhr-llava15-crossval-20260716`）  
> 状态：方法、数据与门槛已预注册，尚未查看新 dev/validation 的任何生成结果。

## 1. 当前问题

标准 SC-VHR 使用双边 logit 外推：

$$
\ell^* = \ell_{VHR} + \alpha(\ell_{VHR}-\ell_{base})
$$

它同时增强 VHR 更偏好的 token，并进一步压低 base 更偏好的 token。LLaVA-1.5 冻结验证中 CHAIRs/CHAIRi 显著改善，但 total generated mentions 只保留 90.16%/91.70%，未通过预注册的 93% coverage gate。事后诊断显示唯一正确对象仍保留 96.60%/95.33%，说明主要问题可能是双边外推对部分真实对象和重复提及的额外抑制，而不是回答整体缩短。

## 2. 候选方法

Positive-Delta SC-VHR（PD-SC-VHR）只保留 VHR 相对 base 的正向增量：

$$
\ell^{PD} = \ell_{VHR} + \alpha\max(\ell_{VHR}-\ell_{base}, 0)
$$

直觉：

1. VHR 更支持的 token 仍获得相同增强；
2. base 更支持的 token 不再被额外线性压低，而是保持 VHR logits；
3. 期望恢复真实对象覆盖，同时由被增强的视觉支持 token 在 softmax 竞争中间接压制幻觉 token。

该变体继续使用 VHR 分支设置 plausibility cutoff，不修改 head selection、KV cache 或 VHR 本身。

## 3. 冻结配置

- 模型：`models/llava-1.5-7b-hf`
- VHR 层：layer 1 + 最后 14 层
- magnitude top-50% heads，ratio 2，启用原 filter
- `alpha=0.5`, `beta=0.1`
- greedy decoding，`max_new_tokens=512`
- 不扫描 alpha、beta、head fraction、layer、cutoff 或 ratio
- dev 同时运行 VHR、标准 SC-VHR、PD-SC-VHR；validation 只在 PD dev 通过后运行，且参数完全不变

## 4. 全新数据划分

从 COCO val2014 的 40,504 个图像中，剔除：

- 既有 `experiment/*chair_ids.json`；
- 工作区所有已有原始结果 JSON 中出现过的 `question_id`；
- quickchair、模型文件和 COCO annotation JSON 不计入结果扫描。

共识别 5,633 个历史已用 ID，剩余 37,656 个。对剩余 ID 排序后使用 `random.Random(20260717).shuffle`：

- 新 dev：前 100，`experiment/pdscvhr_dev_chair_ids.json`
- 新 validation：接下来 100，`experiment/pdscvhr_validation_chair_ids.json`

两套各自唯一、互不重叠，并与已扫描的历史实验 ID 完全不重叠。validation 文件现在固定，但在 dev gate 判定前禁止生成。

## 5. 预注册 dev 通过门槛

PD-SC-VHR 相对同一新 dev 上重新生成的 VHR 必须同时满足：

1. CHAIRs 至少绝对下降 3.0 点；
2. CHAIRi 不得恶化，允许最大上升 0.5 点；
3. 平均长度保持在 VHR 的 ±7%；
4. total generated mentions 不低于 VHR 的 93%；
5. unique correct objects 不低于 VHR 的 95%；
6. 不能通过明显增加重复对象提及伪造 coverage 恢复：repeated mention rate 最大上升 3 个百分点；
7. 输出完整，100 图、ID 顺序完全一致、无空回答。

标准 SC-VHR 作为同分布机制对照，不参与 PD 参数选择。若 PD dev 未通过任一门槛：记录负结果，停止 validation 和所有参数搜索。若 dev 全部通过：才解封固定 validation，并按同一门槛独立判定。

## 6. 输出隔离与回滚

- 使用新目录 `pdscvhr_dev_results/` 与 `pdscvhr_validation_results/`，不覆盖任何历史输出。
- 烟雾测试使用 `pdscvhr_smoke_results/`，不参与指标。
- 标准 SC-VHR、VHR 和 PD-SC-VHR 使用不同 method 名。
- 任何实现问题可恢复到 `scvhr-llava15-crossval-20260716`；核心稳定 InstructBLIP 版本仍为 `scvhr-instructblip-stable-20260716`。

## 7. 执行日志

- 2026-07-17：建立 `research/pd-scvhr` 分支。
- 2026-07-17：完成历史 ID 扫描、新 dev/validation 固定划分和上述方法/门槛预注册；此时尚未实现或运行 PD-SC-VHR。
- 2026-07-17：以独立 `vhr_positive_delta_contrast` generation flag 实现 PD 公式；只有 method 名包含 `pdscvhr` 时启用。标准 SC-VHR 与 VCD 仍走原双边公式。新增 `scripts/run_llava15_pdscvhr.py`，支持同一模型加载、同一 ID 顺序依次运行 VHR/SC-VHR/PD-SC-VHR，并拒绝覆盖已有输出。
- 2026-07-17：张量级公式测试确认标准 SC 输出不变，PD 仅将负 delta 截为 0。新 dev 第一图的 64-token 烟雾测试通过，耗时 8 秒；输出位于 `pdscvhr_smoke_results/`，不参与指标。
