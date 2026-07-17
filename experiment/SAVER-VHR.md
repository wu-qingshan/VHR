# SAVER：由自对比切换触发的视觉区域主动复核

> 开始日期：2026-07-17
> 分支：`research/saver-vhr`
> 起始检查点：`8258862`（tag: `care-scvhr-dev-negative-20260717`）
> 状态：完成方法假设与工程可行性设计；尚未实现、创建新划分或运行实验。

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

从 $A$ 生成局部区域：选择高于 `mean(A)+std(A)` 的 patch；为空时选择最大 patch；取其最小包围框并固定扩张一个 patch。该统计规则在实现前固定，不扫描 top-k 或 crop ratio。

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
