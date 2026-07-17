# SAVER：由自对比切换触发的视觉区域主动复核

> 开始日期：2026-07-17
> 分支：`research/saver-vhr`
> 起始检查点：`8258862`（tag: `care-scvhr-dev-negative-20260717`）
> 状态：工程、局部前向和100图grounding诊断已完成；selected-head空间图未通过grounding gate，SAVER效果实验停止，未创建或运行新dev/validation/test。

## 1. 为什么停止 logits gate 路线

CARE dev 的点估计表明全词表反事实对齐能够改善 CHAIRi、幻觉 mentions、正确对象覆盖和重复，但 CHAIRs 仅从32降到30，95% CI `[-11,7]`。这与 VEG、DG 和 switch diagnostics 共同说明：

1. 只利用 branch logits、margin、符号或候选 support，无法可靠判断候选对象是否真的存在；
2. text-only 分支可以识别语言先验强度，却无法区分“场景中常见但当前图像不存在”的对象；
3. 新增幻觉主要是强场景共现对象，需要空间级视觉证据，而不是更复杂的词表阈值。

CARE 新增错误的代表模式：

- beach/bench → `boat`；
- park/street → `car`；
- zoo/giraffe/zebra → `elephant`；
- baseball field → `sports ball`；
- desk/classroom → `bottle/cup`。

这些 token 在语言和全局视觉语境上都合理，但没有对应局部实体。下一方法必须回答“候选对象在哪个区域被图像支持”。

## 2. 方法概述

暂定名称：**SAVER**（Switch-Activated Visual Evidence Reinspection）。

核心思想：标准 SC 只在少量 token 步真正把 VHR 候选切换为另一个候选。SAVER 不对所有 token 运行第三分支，而只对这些**干预诱发 switch**进行局部视觉复核。

基础分支：

- $I$：启用 VHR 的 intervention 分支；
- $B$：同图像、关闭 VHR 的 base 分支；
- 标准 SC proposal：$\ell_{SC}=\ell_I+\alpha(\ell_I-\ell_B)$。

令 $y_I=\arg\max\ell_I$，$y_{SC}=\arg\max\ell_{SC}$。当 $y_I=y_{SC}$ 时不增加计算；当两者不同时启动主动复核。

## 3. Head-guided 区域定位

VHR 已在 prefill 中根据有图/无图 head 输出差选择视觉敏感 heads。SAVER 复用这些已选 heads，而不是引入检测器或外部 grounding 模型。

在 switch 步，对目标层的 selected heads 计算当前 query 与图像 token keys 的局部相似度：

$$
a_{l,h,p}=\operatorname{softmax}_{p\in\mathcal V}
\left(\frac{q_{l,h}^{\top}k_{l,h,p}}{\sqrt d}\right)
$$

其中 $\mathcal V$ 为图像 token 位置。跨层、跨 selected heads 聚合：

$$
A_p=\frac{1}{|\mathcal H|}\sum_{(l,h)\in\mathcal H}a_{l,h,p}
$$

首版不学习权重，避免将 VHD magnitude 再作为可调门槛。工程诊断必须先确认：

1. LLaVA-1.5 的576个视觉 token 可稳定恢复为24×24网格；
2. selected-head map 非全零、非均匀，且不同图像/生成步存在变化；
3. 计算 map 不改变原 VHR/SC logits 与缓存。

## 4. 主动局部复核

从 $A$ 生成局部区域：选择高于 `mean(A)+std(A)` 的 patch；为空时选择最大 patch；只保留包含全局最大 patch 的4连通分量，取其最小包围框并固定扩张一个 patch。该统计规则在效果实验前固定，不扫描 top-k 或 crop ratio。使用峰值连通分量是为了避免多个空间离散高响应点的总包围框退化成近似全图。

将原图按包围框裁剪并 resize 回模型输入大小，运行一个**仅当前 switch 步使用**的 local base verification forward，得到 $\ell_L$。

候选对验证：

$$
m_L=\ell_L(y_{SC})-\ell_L(y_I)
$$

- 若 $m_L\ge0$：局部证据不弱于 VHR 候选，接受 SC switch；
- 若 $m_L<0$：局部证据更支持 VHR 候选，本步回退 $I$ logits。

该规则没有新增连续阈值，不需要 text-only 全程第三分支。binary 问题继续完整回退 VHR。

## 5. 与现有方法的区别

### 与 VHR/SC

- VHR 只确定视觉敏感 heads，不定位候选实体区域；
- SC 外推干预 effect，但不验证 switch 是否有局部视觉支持；
- SAVER 将“内部因果 switch”作为稀疏触发条件，再使用同一组可靠 heads 定位证据。

### 与 SADI/VISTA

SADI/VISTA 对内部 attention/representation进行连续或广泛重标定。SAVER 不持续修改所有 token，而只复核 SC 实际改变决策的步骤；其目标是保留 SC 修复能力，同时阻止场景共现对象被引入。

### 与 Active-Look

Active-Look 使用全局/局部冲突触发视觉重看。SAVER 的触发来自 intervention-isolated self-contrast，区域来自每样本 VHR-selected heads，并对 $y_I/y_{SC}$ 候选对直接验证。论文必须通过统一实现比较两者，不能只在概念上声称不同。

## 6. 计算与实现方案

### 6.1 Attention map 捕获

当前 `vhr_forward` 使用 SDPA，不返回 attention weights。为避免切换全模型到 eager attention：

1. 仅在 `vhr_saver_diagnostic/verify` flag 启用时，保存 selected-head 的当前 query 和图像 key slices；
2. 只在实际 SC switch 后计算 image-only softmax map；
3. 默认路径完全不创建这些张量；
4. map 张量在当前步使用后立即释放。

需要新增模块状态：visual token start/end、selected heads、last-step image map；必须逐样本重置，避免历史污染。

### 6.2 Local verification forward

首次实现优先做正确性版本：

- 保留原始 pixel tensor 和无 cache 的初始 model kwargs；
- 发生 switch 时，对 crop 重新运行 vision encoder，并以完整当前 prefix 做单步 base forward；
- local branch 不启用 VHR，避免验证信号与被验证干预循环依赖；
- 只取最后 token logits，不维护 local KV cache。

若正确性版本有效，再研究高效版本：缓存少量 crop vision embeddings，或将区域裁剪替换为视觉 token mask。不能先用近似高效实现替代核心假设验证。

### 6.3 预期成本

标准 SC 约2次 decoder forward/token。SAVER 额外成本与 switch rate 和 prefix replay 成正比。历史 InstructBLIP 诊断中 SC switch rate 为5.87%，但 LLaVA 需重新测量。

若 local replay 导致总时延超过 SC 的1.5倍，即使精度通过也必须作为高成本版本报告，并开发缓存/稀疏近似；不得隐藏 replay 成本。

## 7. 首个工程诊断，不是效果实验

在创建 SAVER 新 dev 之前，只允许使用 CARE dev 的1个已用图像验证以下工程不变量：

1. 开启 map diagnostic 但不启用 verification 时，VHR/SC 输出逐字不变；
2. 每个生成步 map shape 与视觉 token网格一致；
3. map finite、sum=1、标准差>0；
4. 不同 selected heads 和生成步的 map 不完全相同；
5. crop 坐标合法，裁剪后 processor/model 可单步前向；
6. 所有临时状态在样本结束后清理。

该诊断不得产出或比较 CHAIR 指标，也不能用于选择区域阈值。

## 8. 后续实验隔离

若工程诊断通过，再创建全新的 SAVER dev/locked-validation/reserved-test，全部排除 CARE 和更早历史 ID。CARE validation/test 不得挪作 SAVER 开发。

必要对照：VHR、SC、RR、SAVER，以及至少一个可复现的区域/主动验证方法。首阶段优先 LLaVA-1.5；通过后移植 Qwen2.5-VL 或 Qwen3-VL。

核心成功条件应覆盖：

- CHAIRs/CHAIRi 显著优于 VHR，并不劣于 SC；
- 阻止的 switch 中新增幻觉比例显著高于修复比例；
- unique correct objects 和 recall 不低于 SC；
- POPE、重复、cycle 和一般能力不回归；
- 报告 switch rate、verification rate、crop面积、时延和显存。

具体数值门槛在工程诊断完成后、创建新划分与运行效果实验前固定。

## 9. 主要风险

1. selected-head attention未必能定位实体，VHR衡量的是视觉敏感性而非空间grounding；
2. image-only softmax忽略文本 token竞争，可能产生看似尖锐但无意义的热图；
3. crop可能遗漏上下文，使真实关系/场景词被误拒；
4. subword switch不一定对应完整实体词；
5. 完整prefix local replay成本可能过高；
6. 与Active-Look/SADI的创新边界必须通过实现和消融证明。

任何工程不变量失败时先记录原因，不进入效果实验，也不把随机attention map包装为视觉解释。

## 10. 输出与回滚

- 新代码使用独立 `vhr_saver_*` flags，默认关闭；
- 工程输出目录：`saver_diagnostic_results/`；
- 后续效果目录：`saver_dev_results/`、`saver_validation_results/`；
- 不修改或覆盖任何 `care_*`、`scvhr_*`、`rrscvhr_*` 历史结果；
- 恢复起点：`care-scvhr-dev-negative-20260717`。

## 11. 工程诊断日志

- 2026-07-17：实现仅诊断的 selected-head image-token map 捕获；默认关闭，不执行local verification。576个视觉token稳定恢复为24×24。
- 2026-07-17：CARE dev首图上，诊断版SC与标准SC的ID/text/confidence逐字段一致；73个解码步均得到finite且归一化的map，平均std=0.0080，72/72相邻步不完全相同。
- 2026-07-17：初版将全部 `mean+std` 高响应patch共同取包围框，平均覆盖81.68%图像，未形成有效局部区域。该结果判定为区域构造工程失败，不进入效果实验。
- 2026-07-17：在不查看任何CHAIR效果的前提下，将区域定义修正为“包含全局最大patch的4连通高响应分量+一patch扩张”；阈值仍为`mean+std`，不扫描。需重新验证输出等价性和区域面积。
- 2026-07-17：第二次同图诊断再次与标准SC逐字段一致；73/73 map finite、归一化误差均值 `1.31e-8`、平均std仍为0.0080、72/72相邻步不完全相同。峰值连通分量在该图为单patch，扩张后固定3×3，平均bbox面积降至1.5625%。区域现在空间连贯且足够局部，但尚未验证raw-image坐标映射、crop processor和完整prefix local forward。
- 2026-07-17：首次 raw-coordinate/local-forward 工程诊断通过。24x24 的 `[0,18,3,21]` 映射到原始 `640x412` 图像的 `[113,309,165,361]`，得到 `52x52` crop；processor 正常产生 `[1,3,336,336]`，完整4-token生成前缀的full/crop base forward均返回finite的32064维logits，生成和local forward后所有VHR临时层状态均为空。该早期步骤不是SC switch且crop落在背景，不能作为grounding有效性证据，因此后续诊断改为只抽取真实switch步骤。
- 2026-07-17：第一次 switch-gated 诊断暴露首个prefill步骤没有map：原捕获条件只接受 `q_len=1`，而第一个候选来自完整prompt prefill。运行在写出文件前失败，没有效果数据。修正为clean prefill也使用最后一个prompt query构图，使每个候选步骤都有对应map；需重新验证输出等价性和switch-local forward。
- 2026-07-17：修正后真实switch-local forward通过。首个switch位于step 11，VHR/base候选均为`,`，SC候选为`of`；原图和local crop的base logits都更支持VHR候选，selected-minus-VHR分别为`-0.625/-0.953`。该例同时显示switch可能只是语法路径变化，并非实体token。
- 2026-07-17：增加all-head控制、去RoPE的content map和attention×value-norm contribution map；保持默认关闭。单图中value map可落到bird实例，但单图不作为结论。
- 2026-07-17：在CARE dev 100图上进行纯grounding诊断，不计算CHAIR、不使用CARE validation/test。6个GPU分片共100行、100个唯一ID、8779个map步骤、504个SC switch；结果目录保持未跟踪。同进程1图标准SC与诊断SC的ID/text/confidence完全一致。

## 12. Grounding gate结果与停止判定

使用COCO instance segmentation mask，只评估诊断map是否具备空间定位能力，不评估幻觉效果。对每个生成步骤比较四种固定构图：

- `selected`：VHR selected heads的RoPE后QK map；
- `all_head`：同层全部heads平均；
- `content`：去除RoPE位置项的content QK map；
- `value`：selected QK attention乘图像value norm后重新归一化。

共得到130个“生成token正好对应图中真实COCO类别”的候选事件。结果如下：

| map | 任意实例pointing | 真实候选pointing | 真实候选crop命中 |
|---|---:|---:|---:|
| selected | 20.29% | 10.00% | 18.46% |
| all-head | 20.77% | 10.00% | 18.46% |
| content | 20.41% | 6.92% | 16.15% |
| value | 16.22% | **16.15%** | 16.15% |
| raw-image center control | - | **33.85%** | - |

配对检验：

- selected vs center：selected-only/center-only=`2/33`，McNemar exact `p=3.67e-8`；
- value vs center：value-only/center-only=`6/29`，`p=1.17e-4`；
- selected与all-head的真实候选命中逐事件完全一致；
- value vs selected为`16/8`，`p=0.152`，没有显著优势。

此外，504次switch中只有6次（1.19%）SC候选本身是COCO对象类别（boat/clock/person/train/sink）；绝大多数是冠词、介词、形容词、动词或subword。说明SC的收益大量来自先改变语法/轨迹、再间接改变后续实体，而不是直接切换对象token。

**停止判定：当前SAVER假设失败。** selected-head视觉敏感性不能转化为可靠空间grounding，且“只在SC switch时复核实体”与真实switch类型不匹配。根据第9节预设风险规则，不创建SAVER效果划分，不运行CHAIR/POPE效果实验，也不扫描crop大小、层权重、top-k或连续阈值。

该负结果支持新的机制结论：VHR-selected heads主要提供全局视觉条件通路，而不是可直接作为检测器使用的候选实体定位图。下一方法应利用它们在隐藏空间中的全局视觉残差，校正与其冲突的语言先验通路，而不再把attention peak包装为空间解释。
