# DC-VHR 当前研究状态

> 更新时间：2026-07-15
> 详细全过程见 `experiment/CS-VHR.md`；原项目诊断见 `EXPERIMENT_DIAGNOSIS.md`。

## 已放弃的第一候选：DC-VHR

暂定名称：Direction-Calibrated VHR（DC-VHR）。

原 VHR 仅按有图/去图 head 输出差分模长选头。DC-VHR 对每个 head 的视觉差分先经过该 head 对应的 `o_proj` 权重块，计算其与整层视觉残差的余弦对齐，再乘未压缩 L2 强度：

$$
s_{l,h} = \cos(W^O_{l,h}d_{l,h}, \sum_j W^O_{l,j}d_{l,j})\,\lVert d_{l,h}\rVert_2
$$

每层选择 top-50%，保持原倍率 2、原作用层和原 outlier filter。开放描述使用 DC 路由；以 Is/Are/Do/Does/Can/Has 等开头的二元问题自动回退原 magnitude VHR。

## 正式结果与显著性否决：InstructBLIP-Vicuna-7B

CHAIR 500图×5 seeds：

| 方法 | CHAIRs ↓ | std | CHAIRi ↓ | std | Len |
|---|---:|---:|---:|---:|---:|
| VHR | 37.84 | 2.29 | 9.72 | 0.79 | 106.42 |
| DC-VHR | **36.28** | **1.97** | **9.50** | **0.76** | 107.50 |

逐 seed DC-VHR CHAIRs：33.0 / 36.2 / 35.6 / 38.4 / 38.2。2500 图配对：修复 328，新增 289，净改善 39。CHAIRs paired bootstrap 差值 -1.56，95% CI [-3.52, 0.36]，改善概率约 94%。

进一步统一逐图 bootstrap：CHAIRs 95% CI [-3.48, 0.40]，McNemar exact p=0.126；CHAIRi 差值 -0.20，95% CI [-0.79, 0.39]。两项核心指标均不显著，因此 DC-VHR 不再作为论文主方案，只保留为“方向信息有弱信号”的失败消融。

POPE：纯 DC 路由平均 F1 85.11，略低于 VHR 85.47。响应形式门控的 10 个随机 POPE 样本与原 VHR 输出逐字一致；二元问题完整结果因此保持 VHR 的平均 F1 85.47。

## 已完成的关键消融

| 变体 | 开发 CHAIRs/CHAIRi | 结论 |
|---|---|---|
| 原 VHR | 44.0 / 15.84 | 开发基线 |
| sigmoid A-VHR | 正式 47.0 / 13.16 | 全头弱扰动，失败 |
| directional + log幅度 50% | 41.0 / 12.51 | 有效但正式 CHAIRi 不稳 |
| directional-linear 50% | **38.0 / 11.53** | 当前最佳 |
| directional 37.5% | 40.0 / 11.16 | 过度稀疏，次优 |
| directional 62.5% | 39.0 / 13.37 | 选头过多 |
| magnitude 25% | 54.0 / 17.69 | 证明收益不是简单少选头 |
| calibrated magnitude∩正方向 | 43.0 / 22.38 | 破坏头间协同 |
| no-repeat trigram | 45.0 / 17.45（DC） | 全局约束破坏正常描述 |
| repetition penalty 1.1 | 46.0 / 15.16（DC） | 负结果 |
| zero-image contrast | LLaVA dev 40.0 / 13.96 | 未优于 remove contrast |

## 跨模型状态

LLaVA-1.5：

- 开发集：VHR 43.0/15.26，DC 40.0/13.64，正向。
- 独立验证集：VHR 34.0/11.25，DC 38.0/11.83，反转。
- zero-image 结构保持对比未改善。

结论：目前只能可靠声称 InstructBLIP 上有效，尚不能声称跨架构泛化。

## 主要产物

- 方法实现：`vhr.py`、`generation.py`、`main.py`
- 固定开发/验证集：`experiment/dev_chair_ids.json`、`experiment/validation_chair_ids.json`
- 正式结果：`official_linear_results/instructblip-vicuna-7b/`
- 实验全过程：`experiment/CS-VHR.md`
- 原项目诊断：`EXPERIMENT_DIAGNOSIS.md`
- 快速评测与配对统计：`eval/quick_chair.py`、`eval/compare_quickchair.py`
- 重复分析：`eval/truncate_cycles.py`

## 下一阶段

1. 设计 LLaVA 专用、位置保持且语义合理的视觉反事实，而不是简单 zero image。
2. 增加 head 得分/选头重合度日志，分析 LLaVA 开发与验证反转原因。
3. 在 LLaVA-NeXT 和一个新模型上验证。
4. 扩展 AMBER/HallusionBench/MMHal-Bench，并报告时延与显存。
5. 若跨模型仍不稳定，将论文定位为 Q-Former/InstructBLIP 的机制改进，并补更强统计与人工分析。

## Delayed DC-VHR：暂停

显著性复核否决纯 DC-VHR 后，曾准备比较 delay={32,64,96} 的延迟路由。根据 2026-07-15 的最新研究决策，该实验已停止，遗留进程已终止，部分输出不进入评测或论文结果。

## VHR + VCD：仅完成可行性评估

当前结论是两者可以结合，但现有 `VHR_GENERATION` / `VCD_GENERATION` 互斥，且 VHR 的模块级 mutable state 会污染 clean/noisy 分支，不能直接同时传参。机械叠加创新性不足；更值得研究的是 clean、text/null-image、noisy-image 三反事实的一致性选头，再进行分支对称的 head-aware VCD。

推荐分数比较去图与噪声图两种反事实的 head 响应方向一致性，而不是惩罚 clean-noisy 差异。完整分析、公式、工程改造点、成本和未来验证矩阵见 `experiment/VHR_VCD_FEASIBILITY.md`。

后续持续研究目标恢复后，已实现低成本 CC-VHR：noisy 视图仅参与显式三阶段 prefill 选头，生成保持单 clean 分支。remove 版本为 42.0/15.31/100.59，改善伴随长度下降 6.31%；zero-query 版本为 48.0/18.04/100.89，明显退化。两者均未通过门槛，因此 CC-VHR 一致性分数路线已整体放弃，不再扫描超参。

随后验证了原 magnitude VHR 选头与分支对称 VCD。开发结果为 42.0/21.43/107.41；CHAIRs 下降 2 点，但 CHAIRi 从 15.84 大幅恶化至 21.43，幻觉提及约从 160 增至 222，平均耗时约翻倍。VCD 联合方向因此整体否决，不扫描 alpha/beta/noise step。

下一候选改为 Self-Contrast VHR：同一图像、同一生成历史下维护 VHR 分支和原模型分支，直接对比 VHR 干预造成的 logits 变化，避免强图像噪声引入额外语义漂移。

Self-Contrast VHR 固定配置 `alpha=0.5, beta=0.1` 的 100 图开发结果为 **29.0/8.97/111.03**，相对 VHR 44.0/15.84/107.36：CHAIRs -15.0，95% CI `[-25,-5]`；CHAIRi -6.87；长度 +3.42%。配对修复 22、引入 7，首次明确通过开发门槛。参数现已冻结，进入独立 validation 100 图，不再调参。

独立 validation 100 图结果为 **26.0/6.39/108.93**，相对 VHR 42.0/17.63/110.05：CHAIRs -16.0，95% CI `[-25,-7]`；CHAIRi bootstrap 95% CI `[-24.92,-1.55]`；长度 -1.02%。配对修复 20、引入 4。开发与验证均显著通过，现进入正式 CHAIR 五种子，参数与实现继续冻结。

正式 CHAIR 五种子已完成：SC-VHR 为 **30.40±2.61 / 8.42±1.17 / 112.77**，原 VHR 为 37.84/9.72/106.42。2500 图配对 CHAIRs 差值 -7.44，95% CI `[-9.48,-5.40]`，McNemar `p=1.10e-12`；CHAIRi 差值 -1.30，95% CI `[-2.26,-0.08]`。五个 seed 均改善。generated mentions 增加 6.21%，hallucinated mentions 降低 7.96%，不是靠少说。详细记录见 `experiment/SC-VHR.md`。下一步运行 POPE。

POPE 三分割也已完成。纯 SC-VHR 平均 Acc/F1/YesRate 为 **85.14/84.34/44.63**，原 VHR 为 85.91/85.47/46.55。SC-VHR precision 提高 0.72，但 recall 下降 2.69，三个分割 F1 均显著下降。结论是开放描述显著改善、存在性判断变得更保守；不能声称通用无代价提升。InstructBLIP 核心实验结束，下一阶段为 LLaVA 跨模型、质量与效率验证。
