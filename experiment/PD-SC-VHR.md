# Positive-Delta SC-VHR 实验记录

> 开始日期：2026-07-17  
> 分支：`research/pd-scvhr`  
> 起始检查点：`bf74d4f`（tag: `scvhr-llava15-crossval-20260716`）  
> 状态：新 dev 已完成；PD-SC-VHR 恢复覆盖但完全失去幻觉收益，未通过 gate。按预注册规则停止 validation 和参数搜索，方法否决。

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
- 2026-07-17：新 dev 三方法单次加载运行完成。VHR/SC/PD 分别耗时 458/868/899 秒；各输出均为 100 行，ID 与固定文件顺序完全一致、无重复、无空回答。

## 8. 新 dev 结果

| 方法 | CHAIRs ↓ | CHAIRi ↓ | Len | generated mentions | hallucinated mentions |
|---|---:|---:|---:|---:|---:|
| VHR | 39.0 | 11.45 | 80.08 | 681 | 78 |
| 标准 SC-VHR | **28.0** | **8.86** | 76.33 | 587 | **52** |
| PD-SC-VHR | 39.0 | 11.46 | 79.76 | 646 | 74 |

逐图配对 bootstrap（20,000 次，seed 20260717）：

- 标准 SC vs VHR：CHAIRs 差值 -11.00，95% CI `[-19,-3]`；CHAIRi 差值 -2.61，95% CI `[-5.40,-0.02]`；修复 14、引入 3。
- PD vs VHR：CHAIRs 差值约 0，95% CI `[-6,6]`；CHAIRi 差值约 0，95% CI `[-1.86,1.80]`；修复 5、引入 5。

## 9. PD 预注册 gate 判定

| 门槛 | PD 结果 | 判定 |
|---|---:|---|
| CHAIRs 至少下降 3 点 | 0.0 | **失败** |
| CHAIRi 最大恶化 0.5 | +0.00 | 通过 |
| Len 在 VHR ±7% | -0.40% | 通过 |
| total mentions ≥93% | 94.86% | 通过 |
| unique correct objects ≥95% | 98.21% | 通过 |
| repeated mention rate 最大 +3pp | -1.64pp | 通过 |
| 完整性 | 100/100、ID 精确、无空文本 | 通过 |

总判定：**失败**。PD 完全恢复了覆盖侧门槛，但没有任何可靠 hallucination reduction，因此固定 validation 未运行，且不扫描部分 clipping、alpha、beta 或其他连续插值参数。

## 10. 机制结论

1. 标准 SC-VHR 的主要有效成分不是“增强 VHR 偏好 token”，而是对 base 更偏好 token 的负向外推；去掉负 delta 后，CHAIRs/CHAIRi 基本回到 VHR。
2. 负向外推确实减少 total mentions，但该指标混合了语义覆盖与重复次数。新 dev 上标准 SC 的 unique correct objects 为 221/224（98.66%，配对 CI `[-0.12,0.05]`），几乎不变。
3. 标准 SC 相比 VHR：重复 mentions 397→327，幻觉 mentions 78→52；总 mentions 681→587 的下降主要由重复和幻觉减少解释，而不是唯一真实对象消失。
4. PD 恢复的主要是正确对象的重复提及：correct mentions 603→572、unique correct 224→220，同时 hallucinated mentions 78→74 并未显著下降。
5. 因此，不应继续优化 total mention 数量本身。下一步应把 unique correct object coverage、重复率和 caption quality 作为主要约束，同时保留标准双边 SC 公式。

## 11. 后续决策

- PD-SC-VHR 路线否决并冻结为负消融。
- 不运行 `experiment/pdscvhr_validation_chair_ids.json`，该集合保持未观察状态，可供未来结构上不同的方法使用。
- 标准 SC-VHR 仍是论文主方法；下一研究重点转向：三套互斥 LLaVA 数据的 pooled 统计、caption quality/重复率、POPE 响应形式路由，以及不改变输出的效率优化。
