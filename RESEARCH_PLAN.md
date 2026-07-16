# VHR 项目研究笔记与改进实验方案

> 更新日期：2026-07-15  
> 目标：在 ACL 2025 论文 *Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence* 的官方实现上形成可验证、可投稿的后续工作。  
> 注意：仓库当前是脏工作区，包含复现修复、学习文档和本地 A-VHR 扩展；本文将论文原方法与本地修改分开记录。

## 1. 项目结论摘要

- 官方方法 VHR 是一个 training-free 的生成期干预：构造“有图”和“去图”两次 prefill，比较每层各注意力头最后一个 token 的输出差异，将差异高于层内中位数的头视为视觉敏感头，并在后续解码中把这些头的输出固定放大 2 倍。
- 仓库已经成功复现三个 7B 模型的 VHR 主结果。最可靠的本地结果为：LLaVA-1.5 CHAIRs 33.40、LLaVA-NeXT 25.16、InstructBLIP 37.84；均与论文接近。
- InstructBLIP 必须使用 `text_config.eos_token_id=2`，且论文配置为最后 18 层；错误 EOS 会令开放式描述生成到长度上限，严重污染 CHAIR。
- 本地 A-VHR 不是原论文方法。它将硬选择改为连续 sigmoid 加权，并加入深层衰减。默认配置在 InstructBLIP 上失败：CHAIRs 从 37.84 恶化到 47.00，CHAIRi 从 9.72 恶化到 13.16，而 POPE 基本不变。
- 这项负结果表明，VHR 的收益很可能依赖“稀疏、足够强的视觉头干预”，简单平滑和整体弱化会丢失效果。后续不应把主要预算投入 A-VHR 大网格调参。

## 2. 代码与方法映射

| 文件 | 作用 | 研究时的关键点 |
|---|---|---|
| `main.py` | 数据循环、方法选择、层配置、输出 | CHAIR 固定 500 图并跑 5 个随机 seed；POPE 跑三个分割 |
| `generation.py` | 修改 Transformers 生成入口，实现有图/去图双 prefill | VHR 只在第一次生成迭代构造 text-only 分支；之后复用 KV cache 和选头状态 |
| `vhr.py` | 替换 LLaMA SDPA attention，计算 `head_vary` 并重加权 | 原版分支是 `head_vary > median`、固定倍率；本地文件还含 A-VHR 分支 |
| `utils.py` | 加载 LLaVA、LLaVA-NeXT、InstructBLIP 和模型适配 | InstructBLIP 的视觉 query token 与 LLaVA 图像 token 路径不同 |
| `datasets.py` | CHAIR、POPE、LLaVA-Wild 数据封装 | CHAIR 的 500 图由 seed 控制随机采样，不同 seed 不是同一图集 |
| `eval/eval.py` | CHAIR、COCO caption 指标和 POPE 评测 | 当前已补充空 glob 显式报错；POPE 文本判定规则较粗糙 |
| `experiment/A-VHR.md` | 本地 A-VHR 方案、运行记录和负结果 | 不是官方结果；是重要的失败实验与新方法动机 |
| `reproduce.md` | 复现数字与 InstructBLIP EOS 故障分析 | 当前最完整的复现记录 |

## 3. 原始 VHR 的实际执行流程

1. 正常输入图像与文本，准备第一次生成。
2. 复制一份输入并移除视觉信息：LLaVA 删除图像 token/像素输入；InstructBLIP 删除前部视觉 query embedding。
3. text-only 分支先执行一次 prefill，各目标层缓存最后一个位置的逐头 attention output。
4. 正常图文分支执行 prefill，逐头计算：

   \[
   v_{l,h}=\lVert o^{text}_{l,h}-o^{vision}_{l,h}\rVert_2^2
   \]

5. 每层选择 `v` 高于中位数的头，并将其 attention output 放大 `alpha=2`。可选过滤逻辑会去掉部分强离群头。
6. 后续单 token decoding 复用同一组选头，不再重新估计视觉敏感性。

论文默认层配置：LLaVA 系列使用第 2 层（代码索引 1）和最后 14 层；InstructBLIP 使用最后 18 层。该方法不训练参数，但相比 greedy 多一次 text-only prefill。

## 4. 原方法的核心局限

### 4.1 变化幅度不等于有益视觉贡献

`head_vary` 只测量移除图像后某个头变化了多少，不判断变化方向是否与最终视觉证据一致。一个对图像扰动敏感但把残差推向错误语义的头，也可能被放大。

### 4.2 固定选一半头，缺少样本自适应预算

中位数阈值使每层几乎恒定选择 50% 的头。清晰图像、模糊图像、是非问答和长描述得到相同干预预算；这不是由视觉证据强度决定的。

### 4.3 一次选头，全程复用

选头发生在首个生成 token 前，后续描述不同物体、属性和关系时仍使用相同头集合。它无法处理“某些 token 需要视觉增强、功能词和确定性 token 不需要”的差异。

### 4.4 text-only 对比并非纯粹的视觉反事实

删除视觉 token 同时改变序列长度、位置编码和注意力上下文。测得的差异混合了视觉信息缺失与输入结构变化。InstructBLIP 还使用固定 32 个 query token 的路径假设，架构泛化有限。

### 4.5 当前证据集中在老模型与物体存在性

三个主模型均为 7B 级较早架构；CHAIR 和 POPE 主要衡量物体存在性，不能充分覆盖属性、计数、空间关系、知识冲突及长文本幻觉。LLaVA-Bench 的裁判评测也需要更新和复核。

## 5. 推荐论文主线：Causal Sparse VHR（CS-VHR）

### 5.1 核心假设

VHR 真正有效的部分不是“把所有视觉敏感头平滑放大”，而是“从少量强视觉头中选择对当前视觉残差有正向贡献的头，并用足够强的倍率干预”。A-VHR 的失败为该假设提供了初步反证证据。

### 5.2 模块一：方向校准的头得分（Directional Calibration）

在现有两次 prefill 中定义逐头视觉差分：

\[
d_{l,h}=o^{vision}_{l,h}-o^{text}_{l,h}
\]

将所有头经各自输出投影后的差分聚合为该层视觉残差 `D_l`，再计算每个头与整体视觉残差的对齐程度：

\[
s_{l,h}=\frac{\langle W^O_{l,h}d_{l,h},D_l\rangle}
{\lVert W^O_{l,h}d_{l,h}\rVert\lVert D_l\rVert+\epsilon}
\cdot \log(1+\lVert d_{l,h}\rVert_2)
\]

只增强 `s>0` 且排名靠前的头。这样把“变化大”改为“变化大且与整体视觉方向一致”。第一版可用 attention 输出空间中的近似对齐，验证有效后再实现分块 `o_proj` 的严格版本。

### 5.3 模块二：稀疏预算选择（Adaptive Sparse Budget）

不再固定选 50%，而根据层内有效得分分布决定 head 数：

\[
k_l=\operatorname{clip}\left(\left\lceil H\cdot
\frac{\sum_h \max(s_{l,h},0)}{\sum_h |s_{l,h}|+\epsilon}\right\rceil,
k_{min},k_{max}\right)
\]

然后仅选择 top-`k_l`。更简单、适合首轮实验的版本是固定预算比例 `{12.5%, 25%, 50%}`；先证明稀疏性，再引入自适应公式。

### 5.4 模块三：保强度的归一化重加权（Strength-Preserving Reweighting）

A-VHR 的主要失败来自有效头倍率下降。CS-VHR 应保持被选头的平均倍率等于原版 2.0，只改变头之间的相对权重：

\[
r_{l,h}=1+(\alpha-1)\cdot
\frac{\operatorname{softplus}(s_{l,h}/\tau)}
{\operatorname{mean}_{j\in S_l}\operatorname{softplus}(s_{l,j}/\tau)}
\]

未选头倍率为 1。该归一化保证选择集合内平均倍率仍为 `alpha`，避免再次把“自适应”误写成“整体减弱”。倍率可裁剪到 `[1, r_max]`。

### 5.5 可选模块：步级门控

首篇实现不建议直接做复杂双前向门控。可先使用上一生成步的归一化熵或 top-1/top-2 margin 控制下一步干预强度，并报告它只是近似在线策略。若静态 CS-VHR 已超过 VHR，再加入此模块作为扩展；否则会令失败原因难以定位。

## 6. 为什么这条主线比继续 A-VHR 更合适

- 它直接针对 VHR 得分的语义缺陷：幅度无方向。
- 它保留原版已验证有效的稀疏、强干预，而不是同时改变选择、倍率和层深三件事。
- 每个模块都有独立、可证伪的消融：方向、预算、强度归一化。
- 额外计算仍主要发生在一次 prefill，可维持 training-free 和低生成开销的卖点。
- A-VHR 的负结果可成为方法动机与消融，而不是被丢弃的调参记录。

## 7. 分阶段实验方案

### 阶段 0：锁定可信基线

使用同一代码提交、相同数据、相同 seed 和相同生成长度重新确认：Greedy、VHR、A-VHR A4。优先使用已有结果做开发筛选，但最终论文表应在统一环境重跑。

必须记录：模型配置哈希、Transformers/PyTorch 版本、GPU、每样本时延、峰值显存、输出长度、EOS 配置和结果文件校验和。

### 阶段 1：低成本机制验证（InstructBLIP）

先只跑 CHAIR 单 seed 做方向筛选；候选配置：

| ID | 得分 | 预算 | 倍率 | 目的 |
|---|---|---|---|---|
| B0 | 原 `head_vary` | 50% | 固定 2 | VHR 基线 |
| B1 | 原 `head_vary` | 25% | 固定 2 | 测试稀疏预算 |
| B2 | 方向校准分数 | 50% | 固定 2 | 只验证方向 |
| B3 | 方向校准分数 | 25% | 固定 2 | 方向 + 稀疏 |
| B4 | 方向校准分数 | 自适应 | 固定 2 | 加自适应预算 |
| B5 | 方向校准分数 | 自适应 | 保强度连续倍率 | 完整 CS-VHR |

筛选条件：相对 B0，CHAIRs 至少下降 1.0，CHAIRi 不上升，长度变化在 ±5%，且单样本生成时延增加不超过 10%。单 seed 只用于淘汰，不能作为论文结论。

### 阶段 2：完整复现协议

在 InstructBLIP 上对进入候选的 1–2 个配置跑 CHAIR 5 seeds 和 POPE 三分割。使用配对 bootstrap 给 CHAIRs/CHAIRi 的 95% 置信区间，并报告每个 seed 的原始值。除均值外，还要报告相对 VHR 的逐图胜/平/负比例。

### 阶段 3：跨架构验证

- LLaVA-1.5-7B：与原论文最强提升模型对齐。
- LLaVA-NeXT-7B：验证长描述场景。
- InstructBLIP-7B：验证 Q-Former 架构。
- 至少增加一个较新的开源模型，如 Qwen2.5-VL-7B 或 LLaVA-OneVision-7B。若其注意力结构无法直接替换，应把“通用 hook/API 适配”作为工程任务单列，不可只在老模型上声称广泛泛化。

### 阶段 4：扩展幻觉评测

最低推荐组合：

- CHAIR：物体级开放描述，与原论文直接比较。
- POPE：物体存在性问答。
- AMBER：生成与判别幻觉，覆盖更丰富属性。
- HallusionBench 或同类诊断集：视觉错觉与推理错误。
- MMHal-Bench：使用更新的多模态裁判模型，并人工复核一个随机子集。

需要同时报告通用能力指标，防止通过少说话获得低幻觉：caption 的 CIDEr/SPICE、回答质量评分、平均长度、拒答率和信息覆盖率。

### 阶段 5：解释性与效率实验

- 选头稳定性：同一图不同 prompt、同一 prompt 不同图之间的 Jaccard 相似度。
- 因果性：逐个屏蔽 top 正向头、top 负向头和随机头，比较 logit/CHAIR 变化。
- 头预算曲线：12.5%、25%、37.5%、50%，画质量—幻觉—时延 Pareto 曲线。
- 层消融：浅层、最后 14/18 层、论文层配置、自动预算。
- 倍率消融：`alpha={1.5,2.0,2.5,3.0}`，检验方向校准是否降低超参敏感性。
- 对比构造：删除视觉 token、零视觉 embedding、保持长度的 learned/null embedding；确认收益不是位置变化伪影。

## 8. 统计与实验规范

- 开发集与最终测试分离。不能反复在 CHAIR 500 测试图上调参后仍把它当无偏测试结果。
- 所有方法使用相同图像集合、prompt、seed、`max_new_tokens` 和停止条件；最好保存图像 ID 列表，而不是仅依赖 Python `random.sample`。
- CHAIR 必须与长度和信息覆盖共同解读。更短的 caption 天然更少提及物体，可能造成虚假的去幻觉提升。
- 对逐图二值 CHAIRs 使用配对 bootstrap 或 McNemar 检验；对 CHAIRi 使用逐图 bootstrap。跨 seed 标准差不能替代逐样本显著性。
- 至少人工审查 100 个样例，按物体、属性、关系、计数、重复、拒答分类，并采用双人标注及一致性统计。
- 报告所有失败配置和选择规则，避免只展示最佳超参。

## 9. 工程风险与优先修复

1. `generation.py` 复制自 Transformers 4.45 私有生成逻辑，版本升级容易失效；实验环境应固定依赖，并增加最小生成回归测试。
2. VHR attention 内部保存 `attn_output_tc`、`aug_heads` 等可变状态；批量推理、异常中断或并发复用模型时可能串样本。每次生成开始与结束都应显式清理状态。
3. 当前实现多处默认 batch size 为 1，如 `attn_output[0, :, -1, :]`；论文实验可保持 batch=1，但需在文档中声明，或改为 batch-safe。
4. InstructBLIP 对比路径硬编码删除 32 个 embedding；应改用模型的 query token 数，并设计保持序列结构的反事实。
5. `eval/eval.py` 的 POPE yes/no 解析仅检查少量否定词；应使用官方评测脚本或更稳健的规范化解析，并保留原始结果供复核。
6. 运行脚本仍存在占位模型路径、GPU 数硬编码和不同脚本默认层数不一致的问题。正式实验前应统一为配置文件或显式 CLI 清单。

## 10. 投稿叙事建议

可采用如下论文论点：

> Existing VHR identifies vision-sensitive heads by counterfactual magnitude, but sensitivity alone does not imply faithful visual contribution. We show that indiscriminate smoothing weakens the sparse intervention and can erase its gains. We therefore introduce causally calibrated sparse routing, which selects directionally aligned heads under an adaptive budget while preserving intervention strength.

贡献应控制在三点：

1. 通过 A-VHR 负结果和因果屏蔽实验揭示“视觉敏感性不等于视觉忠实性”。
2. 提出方向校准、稀疏预算、保强度重加权的 training-free CS-VHR。
3. 在旧三架构与至少一个新模型、多个幻觉维度上验证，并给出质量—幻觉—效率 Pareto 分析。

如果方向校准无法稳定超过 VHR，不应继续包装成新算法。届时更有价值的备选论文是“VHR 机制审计”：系统证明收益来自长度、强度、对比构造或特定模型版本，并提出可靠评测协议；这比无效模块叠加更可信。

## 11. 下一步执行顺序

1. 冻结并标记官方 VHR 基线代码，避免本地 A-VHR 修改继续污染对照。
2. 为 `head_vary`、选中 head、每头倍率、每层输出差分增加可选日志，不改变默认推理结果。
3. 在 50–100 张独立开发图上完成 B0–B3，先验证方向校准是否有信号。
4. 只有 B2/B3 稳定优于 B0 后，再实现自适应预算与连续倍率。
5. 冻结方法和超参后，执行完整 5-seed、跨模型和扩展基准实验。

