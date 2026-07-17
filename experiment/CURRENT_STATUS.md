# VHR 改进当前研究状态

> 更新时间：2026-07-17
> 详细全过程见 `experiment/CS-VHR.md`；原项目诊断见 `EXPERIMENT_DIAGNOSIS.md`。

## 当前执行阶段：Qwen2.5-VL 现代主干迁移

当前论文短板已经从“是否存在正信号”转为现代主干、评测广度和方法 Pareto 不足。
因此停止继续扫描 InstructBLIP/LLaVA 的手工 gate，从稳定标签
`rrscvhr-efficiency-quality-20260717` 建立独立分支 `research/qwen-scvhr`。

本阶段先在隔离 `.venv-qwen/` 中审计 Qwen2.5-VL-7B-Instruct 的 GQA、3D RoPE、
动态 image token 和 cache，再实现 Qwen 专用 VHR/SC。工程等价性全部通过前不创建
效果划分；dev 未过预注册门槛时不运行 locked validation，也不扫描参数。完整协议见
`experiment/QWEN-SC-VHR.md`。

E0/E1/E2 已全部通过：固定 revision 完整下载；原生动态 image token、确定性和
EOS 正常；关闭 patch 首步 logits 误差为 0；独立循环与 HF generate 一致；ratio=1、
A->B 状态清理、GQA query-head 选头、SC 双 cache 和 RR 路由等价性均通过。随后创建
互斥 300/500/500 划分并只运行 dev；locked validation/test 始终未读取。

Qwen 互斥 dev 300 图现已完成且失败：vanilla/VHR/SC 的 CHAIRs/CHAIRi/Len 为
`23.67/6.46/147.48`、`29.67/8.26/315.60`、`27.00/7.93/345.63`。SC 相对 VHR
CHAIRs 只改善 2.67 点且 CI 跨 0，同时 unique correct 损失 6.41%、长度增加 9.52%、
cycle 增加 15.33pp。locked validation/test 未运行。结论是固定 VHR head amplification
在现代 Qwen 上产生架构级负迁移，下一方向必须以 vanilla/base 为安全锚点做不可逆
稳定性回退，而不是继续扫描 VHR 参数。

当前新候选为 SRR-SC-VHR：开放描述在 SC 候选首次将形成已冻结 exact cycle 时，
不可逆回退 vanilla/base cache，并停止后续 VHR forward；binary 继续走 VHR。SRR
必须在同一 dev 上恢复到接近 vanilla 的幻觉、覆盖和稳定性 Pareto，才允许进入尚未
查看的 locked validation。协议见 `experiment/SRR-SC-VHR.md`。

SRR dev 也已完成并停止：它把 SC exact cycle 从 233/300 降至 0、长度从345.63降至
223.26、unique correct 从482恢复到512、时延从25.29降至14.78秒，但 CHAIRs 仍为
27.00，未接近 vanilla 23.67，且严格 coverage/长度/max-hit 门槛失败。locked
validation/test 仍未运行。结论转为：稳定性回退是必要安全组件，但主生成轨迹必须从
vanilla 出发，再做对象级视觉证据验证，不能默认沿VHR/SC生成到出现退化才回退。

当前主线为 OER（Object-Evidence Revision）可行性：先在已打开 dev 的 vanilla 对象
提及上测 Qwen 自身 Yes/No logit 是否能区分真实与幻觉对象。只有固定 verifier 达到
AUC、balanced accuracy 和 score-gap 门槛，才实现通用 noun-phrase 提取与最小 caption
修订；协议见 `experiment/QWEN-OBJECT-EVIDENCE.md`。

对象 verifier feasibility 已通过：617 queries（531真实/86幻觉），ROC-AUC 0.926、
balanced accuracy 0.795，正负 mean score gap 0.587、95% CI `[0.513,0.658]`。这成为
目前现代 Qwen 上最可靠的机制信号。下一步实现通用 noun-phrase + 最小证据修订，但
不能用0.5阈值机械删除整句，因为仍有45个真实对象被判No。

SEAR现已完成并按预注册否决。通用spaCy提取recall为93.33%、平均19.17候选/图；300图
CHAIRs由23.67显著降至17.00，CI为`[-9.67,-4.00]`，但unique correct objects只保留
91.53%。311个删除句移除28次幻觉mention的同时移除243次正确mention，证明整句删除
粒度过粗。locked validation/test未运行，也不扫描阈值或stoplist。下一方法必须在新的
未使用dev上做phrase/clause级局部修订，并保持vanilla主轨迹。详见
`experiment/QWEN-SEAR.md`。

当前候选为VERA-VHR：vanilla生成主轨迹，只有冻结verifier发现`score<=0.1`对象时才对
局部句子生成base与VHR两个修订候选；候选必须通过二次视觉验证并保留全部高置信
protected object roots，否则逐字回退原句。已创建排除4446个历史/保留ID的全新
300/500/500划分，原Qwen locked集合继续封存。协议见`experiment/QWEN-VERA-VHR.md`。

## 最新结论：RR-SC-VHR 效率成立，但暴露生成稳定性代价

统一单 GPU、同一次模型加载 benchmark 已完成。两轮正反顺序的全部输出严格等价，轮间时延波动均小于 2%，不触发第三轮：

| 任务 | VHR | SC | RR | RR 相对 SC |
|---|---:|---:|---:|---:|
| CHAIR 秒/图 | 4.2424 | 8.8068 | 8.7104 | 同一路径，约等速 |
| POPE 秒/题 | 0.4956 | 0.9135 | 0.4969 | **1.838× 加速** |

CHAIR 上 RR=SC、POPE 上 RR=VHR 均逐字段一致。RR 因此同时保留 30.40/8.42 的 CHAIR 结果、恢复 VHR 的 POPE F1 85.47，并在 binary 上恢复单分支成本。

完整文本质量评测发现新的负结果：RR/SC 平均更长（106.56→112.87 words），geometric BLEU 7.84→6.71、METEOR 16.37→14.84；per-caption repeated 2/3/4-gram 均增加。exact cycle 从 6/2500（0.24%）升到 33/2500（1.32%），配对 McNemar `p=7.43e-6`。因此当前方案是“更少对象幻觉 + binary 无回归 + binary 更高效”，不是全面无损提升。

下一主方向：**stability-aware token routing**。只在 SC 候选相对 VHR 引入重复 n-gram/连续循环风险时回退 VHR，避免此前全局 no-repeat 对正常描述的破坏。本次正式 CHAIR 只能用于发现问题，后续阈值和规则必须在新开发集上确定，并使用新的锁定验证集；不能回看正式结果调参。详细数字见 `experiment/RR-SC-VHR-EFFICIENCY-QUALITY.md`。

## 最新进展：SC-VHR 的 LLaVA-1.5 冻结迁移

InstructBLIP 稳定结果与 POPE 负结果保持不变。冻结 SC-VHR 已在 LLaVA-1.5 的独立 dev/validation 各 100 图上完成：

| 数据 | VHR CHAIRs/CHAIRi/Len | SC-VHR CHAIRs/CHAIRi/Len | generated mentions 保留率 |
|---|---|---|---:|
| dev | 43.0 / 15.26 / 80.14 | **35.0 / 12.34 / 78.42** | 90.16% |
| validation | 34.0 / 11.25 / 82.03 | **21.0 / 6.60 / 81.10** | 91.70% |

CHAIRs/CHAIRi 两套均显著改善且长度稳定，但 total generated mentions 均低于预注册 93% 下限，因此不进入 LLaVA 正式五种子，也不扫描参数。事后规范化对象诊断显示唯一正确对象仍保留 96.6%/95.3%，而幻觉 mentions 降低 27.1%/46.3%；应表述为“跨架构幻觉降低成立，但 coverage-preserving 结论未过预注册 gate”。完整记录见 `experiment/SC-VHR-LLAVA15.md`。

2026-07-17 进一步在完全未使用的新 100 图上比较 VHR、标准 SC 和 Positive-Delta SC。标准 SC 再次显著改善：39.0/11.45 → **28.0/8.86**，CHAIRs 95% CI `[-19,-3]`；unique correct objects 保留 **98.66%**。只保留正 delta 的 PD-SC 恢复 total mentions 到 94.86%，但 CHAIRs/CHAIRi 为 39.0/11.46，和 VHR 无差异，因此否决且不运行锁定 validation。机制结论是：对 base 偏好 token 的负向外推是 SC 的核心有效成分；total mentions 下降主要混合了重复减少，不能直接等价为语义覆盖下降。完整记录见 `experiment/PD-SC-VHR.md`。

三套互斥 LLaVA 划分 pooled 共 300 图：CHAIRs 38.67→**28.00**（95% CI `[-15.00,-6.33]`），CHAIRi 12.66→**9.25**，修复/新增 39/7，McNemar `p=1.83e-6`。hallucinated mentions 下降 34.72%，repeated mentions 下降 11.35%；unique correct objects 716→693（保留 96.79%），下降约 3.2% 且 pooled CI 不跨 0。论文中应报告为“显著 hallucination reduction + 小幅但真实的语义覆盖代价”，而不是完全无损。

Disagreement-Gated SC-VHR 只在 VHR/base argmax 不一致时使用 contrast。新 dev 上实际仅 11.46% token 步触发；unique correct 224→224、mentions 保留 97.8%，但 CHAIRs 39→38（CI `[-5,3]`），远差于标准 SC 的 28。它只显著改善 CHAIRi 到 10.06，因此否决且不运行 validation。结论是标准 SC 的关键收益大量来自“两分支 argmax 相同、外推后第三 token 上升”的步骤，简单二元路由不足。

随后对标准 SC 进行 100 图逐 token 诊断重跑，并与历史输出 100/100 逐字等价。9955 步中 branch argmax 分歧占 11.40%，但标准 SC 实际 switch 只有 5.87%；584 次 switch 中 508 次（86.99%）发生在两分支 argmax 原本相同的步骤。修复与持续幻觉在 switch rate、selected delta、score margin 上没有可用于可靠阈值的显著分离。因此停止两分支标量 gate 扫描，下一候选转向引入在线 text-only 视觉证据的 VEG-SC-VHR。

VEG-SC-VHR 保留在线 text-only 第三分支，并只允许视觉支持不低于 VHR top 的标准 SC switch。新 dev 结果为 33.0/9.53/79.85，mentions 保留 93.98%、unique correct 保留 98.21%，处于 VHR 39.0/11.45 与标准 SC 28.0/8.86 之间；但 CHAIRs 未达到预注册 ≤31，paired CI `[-14,2]` 跨 0，且比标准 SC 慢 21.5%。视觉 support margin 对修复有信号，却不能区分新增幻觉，因此不扫描 margin、不运行 validation。下一方向转为 response-type routing 解决 POPE 回归。

Response-Routed SC-VHR 已完成。POPE 9000/9000 binary 覆盖、CHAIR 非 binary；同进程严格等价和完整 11,500 行物化审计均通过。RR CHAIR 保持标准 SC 的 30.40/8.42，POPE 精确恢复 VHR 的 Acc/F1/YesRate 85.91/85.47/46.55，同时 binary 推理关闭第二分支、成本约减半。当前建议论文实用主方案为 RR-SC-VHR，创新核心仍是 SC，routing 作为任务依赖 calibration 组件；纯 SC POPE 负结果必须保留。

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
