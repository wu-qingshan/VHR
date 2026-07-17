# VHR 改进论文成熟度与下一阶段调整建议

> 日期：2026-07-17  
> 状态：路线评估；不启动新实验，不修改任何历史结果或划分  
> 当前方法检查点：RR-SC-VHR 已完成，SAVER 仅处于工程诊断阶段

## 1. 结论

当前结果已经超过“只有 InstructBLIP 上的偶然正结果”，但仍不足以支撑 2026 年较高质量主会论文。

- InstructBLIP 正式 2500 图上，RR/SC-VHR 将 CHAIRs/CHAIRi 从 `37.84/9.72` 降至 `30.40/8.42`，CHAIRs 配对差值为 `-7.44`，95% CI `[-9.48,-5.40]`。
- LLaVA-1.5 三套互斥划分 pooled 300 图上，SC-VHR 将 CHAIRs/CHAIRi 从 `38.67/12.66` 降至 `28.00/9.25`，CHAIRs 95% CI `[-15.00,-6.33]`。
- 因此“跨架构有信号”已经成立；但 LLaVA 结果仍是小规模开发/机制证据，没有完成匹配论文主表的正式测试。
- POPE、caption quality 和覆盖分析暴露了真实代价：纯 SC 的 POPE F1 下降，RR 通过任务路由恢复；SC/RR 的重复和 exact cycle 增加；LLaVA pooled unique correct objects 下降约 3.2%。

若目标是 workshop、短文或较弱期刊，现有证据可形成一篇完整的机制型工作；若目标是 CVPR/ICCV/ECCV/ACL/EMNLP 等较高质量主会，单独依靠 InstructBLIP 正式结果和 LLaVA 小规模结果不够。

## 2. “距离 SOTA”不能只比较一个 CHAIR 数字

CHAIR 对提示词、最大生成长度、解码策略、图像集合和 recall/回答长度非常敏感。不同论文中的绝对值通常不能直接横向排名。

截至 2026-07-17，可作为当前竞争态势参考的工作包括：

- ADAPT（ECCV 2026，arXiv:2606.31054）：同时报告 LLaVA-1.5-7B/13B 与 Qwen2.5-VL-3B/7B；在其 Obj-Hal 协议中，完整 ADAPT 的 LLaVA-1.5-7B CHAIRs/CHAIRi 为 `30.7/8.5`，Qwen2.5-VL-7B 为 `16.8/5.6`；另报 AMBER、MMHal、POPE、RefCOCO grounding、MME/MMBench 和效率。完整版本包含偏好训练，training-free 版本应单独比较。
- FADE（arXiv:2606.29431）：覆盖 LLaVA-1.5、mPLUG-Owl2、InstructBLIP、InternVL3、Qwen2.5-VL 和 Qwen3-VL，并报告 POPE、CHAIR、MME、MMHal/HalBench、MMBench 与效率。它同时展示了现代模型上不同方法会出现明显的幻觉—覆盖—一般能力权衡。
- BRACS（arXiv:2605.29881）：training-free、自适应按需 steering；在统一 LLaVA-1.5 协议中将 greedy CHAIRs 从 `47.8` 降至 `40.0`，POPE 总体 F1 从 `83.64` 提升至 `86.37`，并报告 recall、MME/MMBench/MMMU/MMHal 和吞吐。

这些数字的协议并不相同，不能据此断言当前 `30.40` 一定优于或落后于某个方法。当前真正落后的方面是证据广度：现代 backbone、相同协议强基线、多维 benchmark、grounding 依据、一般能力和效率。

## 3. 当前方案的论文短板

1. **主干模型老旧。** InstructBLIP 和 LLaVA-1.5 适合与经典解码方法比较，但不能代表 2026 年主流 MLLM。
2. **正式跨模型证据不足。** LLaVA pooled 300 图说明机制可迁移，但不是独立正式主表；100 图划分的 CHAIRs 置信区间过宽，容易出现开发/验证反转。
3. **方法 Pareto 不完整。** RR 修复 POPE，但没有修复重复、cycle、传统 caption overlap 和 LLaVA 的小幅正确对象覆盖下降。
4. **推理成本偏高。** 开放描述需要双分支，当前约为 VHR 的两倍；近期方法普遍强调单分支或低触发率干预。
5. **SAVER 新颖性需要重新界定。** 近期 ADAPT、AFIP、区域感知 attention recalibration 等工作也使用 attention drift、视觉 anchor 或区域干预。SAVER 若只做 attention 热图加权，创新边界较弱；必须突出“由 SC switch 触发、实体级局部重看、候选对证据验证、按需计算”。

## 4. 建议的方法调整

不再继续扫描 VHR 选头比例、alpha/beta、简单 logits gate 或 VCD 噪声参数。RR-SC-VHR 固定为强基线和机制消融，主方法转为 **实体级按需视觉复核**：

1. SC 只提出可能改变 VHR 决策的候选，而不是无条件接受。
2. 将 subword token 聚合为实体/名词短语风险单元；单个子词直接裁图验证缺少语义完整性。
3. 使用 selected-head 空间图定位候选实体证据，并在局部图像上比较 SC 候选与 VHR 候选的条件支持。
4. 仅当局部证据支持 SC 候选时接受 switch，否则回退 VHR。
5. 把 verification rate、crop 面积、阻止的新增幻觉/误拒真实对象比例作为核心机制指标。

论文叙事应从“进一步降低 CHAIR”调整为：

> VHR/SC 增强视觉依赖仍不能区分真实局部证据与场景共现先验；我们把全局解码干预转化为按需、实体级的局部证据检验，在幻觉、覆盖、稳定性和计算之间取得更好的 Pareto。

## 5. 模型与 benchmark 的最低配置

### 模型

- LLaVA-1.5-7B：开发、与大量历史方法公平比较。
- Qwen2.5-VL-7B：现代主结果，优先级最高。
- InstructBLIP-7B：保留现有正式结果，作为 Q-Former 架构复核。
- 若资源允许，再增加 Qwen3-VL-8B 或 InternVL3-8B；LLaVA-NeXT 可作为低工程成本补充，但不能替代现代主干。

当前机器有 8 张 48GB RTX A6000，7B/8B 推理模型矩阵在硬件上可行；主要约束是实现移植和评测时间，不是显存。

### 幻觉与能力评测

- 对象生成：CHAIR，同时报告 CHAIRs、CHAIRi、Recall/unique correct、长度和对象提及数。
- 生成与判别综合：AMBER。
- 二元存在性：POPE 三分割，报告 Acc/F1/precision/recall/yes rate。
- 开放式幻觉：MMHal-Bench 或 HallusionBench 至少一个。
- 一般能力保持：MME 与 MMBench 至少一个，最好两个。
- 定位依据：RefCOCOg/RefCOCO 或 COCO box pointing-game，验证 SAVER 空间图确实具备 grounding 能力。
- 系统指标：每 token/每样本时延、峰值显存、verification rate、相对 greedy/VHR 吞吐。

### 强基线

- Greedy、VHR、VCD、OPERA：经典可比基线。
- SC-VHR、RR-SC-VHR：本文内部强基线。
- 至少复现两个近期 training-free 方法，例如 VISTA/DAMO/FADE/BRACS/ADAPT-TF 中可获得且协议可对齐者。
- 完整 ADAPT 包含训练，应与 training-free 方法分栏，避免不公平的“统一 SOTA”宣称。

## 6. 建议的阶段门槛

### 阶段 A：先验证定位，不跑幻觉效果

- 完成 raw image 到 processor resize/center-crop 再到 24x24 token 网格的精确坐标审计。
- 在独立 grounding validation 上，selected-head map 至少显著优于随机、中心先验和 all-head average；若定位无效，停止 SAVER，不进入 CHAIR。
- 验证 crop + 完整 prefix 的 local forward、临时状态清理和标准 SC 输出逐字不变。

### 阶段 B：新的 LLaVA 开发与锁定验证

- 不再使用 100 图作为最终主要判定；建议 dev 300、locked validation 500，另保留正式 test。
- 只运行一个预注册配置，不依据 validation 扫 crop 大小、阈值或 verification margin。
- 成功标准不应只要求低于 VHR，还要相对 SC 改善 Pareto：保持或降低 CHAIRs/CHAIRi，同时恢复至少一半 SC 损失的 unique correct objects，并降低 cycle/repetition，POPE 不退化。

### 阶段 C：现代模型迁移

- 先迁移 Qwen2.5-VL-7B；若 VHR head intervention 无法自然兼容 GQA/新视觉 token 布局，需要把方法抽象为层/头无关的 evidence map 接口，而不是硬编码 LLaVA token 区间。
- 现代模型上至少两个幻觉 benchmark 取得一致改善，且 MME/MMBench 不显著下降，才能形成跨架构主张。

### 阶段 D：正式主表

- 三模型、至少四个幻觉评测、一般能力、grounding、效率和完整消融。
- 使用逐样本 paired bootstrap/McNemar，并同时报告效果量、置信区间和失败案例。
- 历史正式 InstructBLIP 数据只作冻结验证，禁止继续用于方法选择。

## 7. 优先级决策

1. 完成 SAVER 工程诊断，但在效果实验前增加 grounding gate。
2. 立即开展 Qwen2.5-VL 的接口可行性审计；不必等 LLaVA 所有效果实验结束才发现无法迁移。
3. 重新审计 SAVER 与 ADAPT/AFIP/区域感知方法的创新边界，并把“局部候选对验证”写成不可替代的核心，而非 attention 重加权。
4. 优化双分支执行：研究同 batch 并行 VHR/base forward 或共享不受干预的计算，目标是把开放描述墙钟成本从约 2x 压低。
5. 在满足上述条件前，不启动 InstructBLIP 新一轮调参，也不以单一 CHAIR 数值追逐跨论文 SOTA。

## 8. 投稿成熟度判断

- **当前直接投稿较高质量主会：不建议。** 核心正结果可靠，但模型和 benchmark 证据不足，且稳定性/覆盖/成本代价尚未解决。
- **完成 LLaVA 正式结果 + Qwen2.5-VL + AMBER/MMHal + general capability：具备主会投稿基础。**
- **若 SAVER 能在三个架构上显著改善 SC 的幻觉—覆盖—稳定性 Pareto，并证明 grounding 与按需复核机制有效：可以形成较有竞争力的论文。**
- **若现代模型迁移失败：**应收缩为 InstructBLIP/Q-Former 的机制论文或负结果/分析论文，不能声称通用 MLLM 幻觉方法。
