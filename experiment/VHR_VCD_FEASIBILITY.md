# VHR 与 VCD 联合方法可行性评估（未进行实验）

> 日期：2026-07-15  
> 状态：仅完成论文方法与代码级评估；没有启动推理、调参或评测。此前遗留的 delayed DC-VHR 进程已终止。

## 1. 结论

VHR 可以与 VCD（Visual Contrastive Decoding）结合，但当前仓库不能通过同时传入 `vhr_layers` 和 `images_cd` 直接实现。两者在 `get_generation_mode` 中是互斥的 generation mode，而且 VHR 将选头和加权状态保存在共享 attention 模块中；若直接把 VCD 的 clean/noisy 两次 forward 放入 VHR 解码循环，noisy 分支会继承甚至改写 clean 分支的 VHR 状态。

从论文创新性看：

- **机械串联 VHR + VCD：工程可行，研究价值一般。** 它容易被审稿人视为两个现有 training-free 方法的简单叠加，并且无法说明收益来自选头还是 logits 对比。
- **推荐方向：反事实一致性选头 + head-aware VCD。** 使用 clean、text-only/null-image、noisy-image 三个视图，选择对两种视觉反事实呈现一致响应的 head，再执行分支对称的 VCD。这不是简单叠加，而是用 VCD 的图像反事实修正 VHR “视觉敏感性不等于视觉忠实性”的缺陷。
- **总体可行性：中高；直接超过 VHR 的把握：中等，不能在实验前承诺。** 最大风险是 VHR 和 VCD 都在增强视觉条件，可能重复校正并令模型过度保守。现有 POPE 分析已表明 VHR 会降低 yes/假阳性，同时增加部分假阴性；联合方法可能进一步放大这一权衡。

## 2. 两种方法的互补性

### 2.1 VHR 的作用位置与不足

原 VHR 在 prefill 阶段比较有图与无图输入的每头输出：

$$
D^{text}_{l,h}=\lVert o^{clean}_{l,h}-o^{text}_{l,h}\rVert_2^2
$$

每层选择高于中位数的 head，并在之后所有 decoding step 固定放大。它是**模型内部、head 级、稀疏干预**。

主要不足：

1. 差分模长只能说明“对是否有图敏感”，不能说明该 head 提供的是正确视觉证据。
2. remove-image 会改变序列长度和位置，尤其在 LLaVA 上可能把位置/结构变化混入视觉差分；本地 LLaVA 开发与验证结果反转与此风险一致。
3. 每层固定选择 50%、固定倍率、整段生成固定路由，没有根据反事实可靠性进行校准。
4. mutable state 隐式依赖“第一次长序列 forward 是 text、第二次是 clean”，不适合三分支解码。

### 2.2 VCD 的作用位置与不足

当前 VCD 对 clean image 与 diffusion-noised image 分别前向，在词表 logits 上计算：

$$
\ell^{VCD}_t=(1+\alpha)\ell^{clean}_t-\alpha\ell^{noisy}_t
$$

并用 clean logits 的 plausibility cutoff 屏蔽低概率 token。它是**模型输出端、token 级、全模型对比**。

主要不足：

1. 它不知道 clean/noisy 差异由哪些内部 head 产生，真实视觉证据、噪声伪影和全局分布漂移被统一处理。
2. 强扩散噪声不一定只删除对象证据，也可能改变属性、场景与语言流畅性，造成过度惩罚。
3. 每个 decoding step 都执行 clean/noisy 两次 forward，并维护两套 KV cache，计算和显存成本明显高于 VHR。
4. 当前仓库没有固定 torch 随机种子，`add_diffusion_noise` 的噪声不能由 CHAIR 的 Python seed 完整复现。

因此两者具有真实互补性：VHR 能定位内部稀疏通路，VCD 能提供与 text-only 不同的视觉反事实和 token 级校正；但二者也可能对同一视觉信号重复放大。

## 3. 四种联合方式

| 方案 | 做法 | 优点 | 主要问题 | 建议 |
|---|---|---|---|---|
| A. clean-only VHR + VCD | clean 分支启用 VHR，noisy 分支保持原模型，再做 logits contrast | 最直观，可能产生较强改善 | 两分支模型本身不同，logit 差异混入 VHR 干预，因果解释不干净；易过校正 | 只作为激进消融 |
| B. symmetric VHR + VCD | clean/noisy 使用同一组选头和同一倍率，再做 VCD | 对比两侧模型变换一致，解释最干净 | 同时增强 noisy 激活可能放大噪声，也可能抵消部分对比收益 | 机械联合中首选基线 |
| C. noisy-aware VHR | noisy 视图只参与选头，生成时只保留单 clean 分支 | 长文本 decoding 仍为一次 forward，效率好 | 严格说借用了 VCD 反事实，但没有执行完整 VCD logits 解码 | 推荐的高效主消融 |
| D. consensus heads + symmetric VCD | 三视图一致性选头；clean/noisy 对称增强；再做 VCD | 内部定位与输出对比统一，方法完整、论文创新性最好 | 三次 prefill、两次逐 token forward，工程最复杂 | 推荐论文主候选 |

## 4. 推荐方法：Counterfactual-Consensus Head-aware VCD

不建议使用下面这个表面直观的分数：

$$
D(o^{clean},o^{text})-\lambda D(o^{clean},o^{noisy})
$$

原因是 VCD 使用的 step-500 强噪声旨在破坏视觉信息；真正看图的 head 本来就应该对它敏感。直接惩罚 clean-noisy 差异，可能反而选中忽略图像的 head。

更合理的做法是比较两个视觉反事实的**方向一致性**。对每层每头定义：

$$
\Delta^{text}_{l,h}=o^{clean}_{l,h}-o^{text}_{l,h},\qquad
\Delta^{noise}_{l,h}=o^{clean}_{l,h}-o^{noisy}_{l,h}
$$

$$
C_{l,h}=\max\left(0,\cos(\Delta^{text}_{l,h},\Delta^{noise}_{l,h})\right)
$$

再用需要两种反事实都具有足够响应的调和强度：

$$
M_{l,h}=\frac{2\lVert\Delta^{text}_{l,h}\rVert_2
\lVert\Delta^{noise}_{l,h}\rVert_2}
{\lVert\Delta^{text}_{l,h}\rVert_2+\lVert\Delta^{noise}_{l,h}\rVert_2+\epsilon}
$$

最终分数：

$$
S_{l,h}=C_{l,h}\,M_{l,h}
$$

含义是：只有当“完全移除视觉”和“破坏视觉内容”令该 head 沿相近方向变化，并且两种变化都不弱时，才把它视为反事实一致的视觉 head。这比原 VHR 的单一差分模长更能排除位置变化、单一噪声伪影和偶然大幅响应。

选中 head 集合 $H_l$ 后，推荐主版本对 clean/noisy 两个分支使用完全相同的 head 变换 $T_H$：

$$
\ell_t^*=(1+\alpha)\ell_t(T_H(x^{clean}))-
\alpha\ell_t(T_H(x^{noisy}))
$$

并沿用 VCD 的 clean plausibility constraint。分支对称非常重要：否则最终差异同时包含“图像反事实”和“模型干预不同”两个变量。

建议暂定名称：

- Counterfactual-Consensus VHR（CC-VHR）：只做三视图一致性选头、单分支生成。
- Head-aware Visual Contrastive Decoding（H-VCD）：一致性选头后执行对称 VCD。

## 5. 三视图 prefill 的合理顺序

当前 VHR 通过 `q_len > 1` 和 `attn_output_tc is None` 隐式推断分支，必须改为显式 phase。建议顺序：

1. `text_capture`：保存 text/null-image 的每头末 token 输出，不保留其 KV cache。
2. `noisy_capture`：保存 noisy 输出并保留 noisy KV cache；此时不加权。
3. `clean_select`：逐层读取 text/noisy 摘要，计算 $S_{l,h}$、选择 head，并立即对 clean 当前层输出加权；保留 clean KV cache。
4. decoding：`decode_clean` 和 `decode_noisy` 交替执行，两者复用同一 $H_l$；若做 CC-VHR，则只执行 clean。

这样只需要三次 prefill，不需要选头后重新跑第四次 clean prefill。

结构保持也很重要。text counterfactual 应优先使用与原输入等长的 null-image/zero-query 表示，而不是删除图像 token；否则 $\Delta^{text}$ 会继续混入位置偏移。对 LLaVA，简单的全零像素经过 vision tower 后未必等于真正 null image，后续应比较 zero embedding、均值图像或 token mask，但本阶段不做实验选择。

## 6. 当前代码必须解决的问题

1. **generation mode 冲突**：`vhr_layers` 先把模式改成 `VHR_GENERATION`，随后 `images_cd` 只接受 greedy/sample，因而会报错。需要新增独立的 `VHR_VCD_GENERATION` 或统一解码器。
2. **attention 状态污染**：`attn_output_tc`、`aug_heads`、`aug_ratio_per_head` 存在模型模块上，clean/noisy 共用。必须增加显式 `branch/phase`，分别保存 `text_output`、`noisy_output`，并禁止任一分支意外重选 head。
3. **异常清理**：当前 VHR 只在正常返回后恢复 attention；联合模式应使用 `try/finally`，否则 OOM/异常后下一样本可能串状态。
4. **VCD 参数读取不一致**：`GenerationConfig` 注册了 `self.cd_alpha/self.cd_beta`，但 `_evolve_vcd_decoding` 从 `model_kwargs` 读取；正常 generate 参数解析后它们可能已被移入 generation config，当前代码实际很可能总落到 `alpha=0.5, beta=0.1`。联合实现应统一从 `generation_config` 读取。
5. **噪声可复现性**：数据循环只调用 `random.seed(seed)`，没有固定 torch generator；应为每个 image/seed 派生独立 torch generator，并记录 noise step 与 noise seed。
6. **batch 假设**：VHR 使用 `attn_output[0, :, -1, :]`，只正确支持 batch size 1；联合实现需明确保持 batch=1，或将状态扩展到 batch 维。
7. **InstructBLIP 硬编码**：text/noisy embedding 路径写死 32 个 query token，应改用模型配置中的 query token 数。

## 7. 计算与显存开销估计

令一次 prompt prefill 成本为 $P$，完整单分支 autoregressive decoding 成本为 $D$：

| 方法 | 近似计算量 | KV cache | 特点 |
|---|---:|---:|---|
| Greedy | $P+D$ | 1 套 | 基线 |
| VHR | $2P+D$ | 1 套生成 cache | 额外 text prefill |
| VCD | $2P+2D$ | 2 套 | 长输出时接近 2 倍解码成本 |
| CC-VHR | $3P+D$ | 1 套生成 cache | 三视图只用于选头，长输出较省 |
| H-VCD | $3P+2D$ | 2 套 | 比 VCD 多一次 text prefill；长输出仍主要由双分支 decoding 主导 |

每层只需临时保存 text/noisy 的末 token 每头向量，摘要内存为 $O(LHd_h)$，相对 KV cache 很小。H-VCD 的主要显存压力仍是 clean/noisy 两套 KV cache。

## 8. 论文价值与主要风险

### 可形成的核心论点

> Vision sensitivity measured by a single counterfactual is not sufficient for faithful grounding. A reliable visual head should respond consistently to multiple interventions that remove or corrupt visual evidence. We use counterfactual consensus to localize VCD to reliable internal pathways and perform branch-symmetric contrastive decoding.

相比“VHR + VCD”，这个叙事具有明确的新问题、新判据和新机制：

1. 揭示单反事实 VHR 把结构敏感性、噪声敏感性与忠实视觉贡献混在一起。
2. 提出多反事实方向一致性，而不是再扫描 head fraction 或倍率。
3. 将内部 head 路由与输出端 VCD 统一，并可提供 CC-VHR/H-VCD 的质量—效率 Pareto。

### 主要失败模式

1. **重复校正**：两者都削弱语言先验，联合后可能少说、拒答或增加 POPE false negative。
2. **强噪声语义失真**：step 500 的差异可能不只代表视觉证据被移除。
3. **对称增强抵消**：clean/noisy 同时增强相同 heads 后，logit 差异可能没有增加；但这是比非对称方案更可信的负结果。
4. **架构依赖**：InstructBLIP 的 32 个视觉 query 与 LLaVA 的大量 image token 机制不同，统一三视图构造并不自然。
5. **计算代价**：若 H-VCD 仅带来很小 CHAIR 改善，双倍 decoding 成本难以形成有吸引力的 Pareto。

## 9. 若之后获准实验，最低限度的验证矩阵

本节只是预先冻结比较关系，本次未执行。

1. Greedy、VHR、VCD：必要独立基线。
2. naive clean-only VHR+VCD：检验简单叠加上限。
3. symmetric VHR+VCD：控制分支模型一致性。
4. CC-VHR：检验多反事实选头本身是否有效。
5. H-VCD：完整方法。
6. 选头分数消融：仅 $D^{text}$、仅 $D^{noise}$、两者幅度和、方向一致性 $C$、完整 $C\times M$。

必须同时报告 CHAIRs、CHAIRi、Len、每图对象提及数、POPE F1/yes ratio/FP/FN、生成时延和峰值显存。正式结论仍需逐图配对 bootstrap/McNemar、跨 LLaVA/InstructBLIP 验证，并避免在已有正式 CHAIR 结果上继续调参。

## 10. 当前决策

- 保留 VHR+VCD 作为下一阶段候选。
- 不把机械叠加作为论文主方法。
- 优先级：先设计并审查三视图显式状态机与反事实一致性分数；若后续允许实验，先比较低成本 CC-VHR，再决定是否承担 H-VCD 的双分支解码成本。
- 当前不修改联合方法代码、不启动实验。

## 11. 后续执行记录（研究目标恢复后）

> 2026-07-15 更新：此前“只评估、不实验”的阶段已经结束；根据持续研究目标，开始验证低成本 CC-VHR。H-VCD 的双分支 logits 解码仍未实现或运行。

### 11.1 首个冻结候选

- 模型：InstructBLIP-Vicuna-7B。
- 数据：`experiment/dev_chair_ids.json` 的 100 张独立开发图，不使用正式 CHAIR 测试结果调参。
- 层与干预：最后 18 层、top-50%、倍率 2、保留原 VHR filter。
- text counterfactual：首轮使用原 VHR 的 remove-image，避免同时改变多个因素。
- noisy counterfactual：VCD diffusion step 500；noise seed 由 CHAIR seed 和 image id 唯一确定。
- score：两种反事实差分的 cosine alignment × harmonic strength；首轮保持固定 top-50% 预算。
- 生成：noisy 只参与 prefill 选头，之后仍为单 clean 分支，即 CC-VHR，而非完整 H-VCD。

开发门槛保持为：相对同图 VHR，CHAIRs 至少下降 1.0，CHAIRi 不上升，长度变化在 ±5%。不达门槛则放弃当前分数，不进入正式五 seed。

### 11.2 实现状态

- `vhr.py`：增加 `consensus` score mode，以及显式 `text/noisy/clean/decode` phase；三分支不再依靠 forward 次序猜测状态。
- `generation.py`：增加 noisy prefill，clean prefill 逐层选头；VHR attention 恢复改为 `try/finally`。
- `main.py` / `utils.py`：增加 `vhr_noise_step`、确定性 torch noise generator 和 InstructBLIP noisy query embedding。
- `scripts/run_instructblip_ccvhr_dev.py`：固定开发运行入口，可在同一模型加载中运行候选。
- 静态检查通过；1 张真实 InstructBLIP 冒烟成功，生成耗时约 14.7 秒，无 shape/cache/phase 错误。
- 100 图 remove-image 开发实验已完成：VHR 为 44.0/15.84/107.36，CC-VHR-remove 为 42.0/15.31/100.59（CHAIRs/CHAIRi/Len）。配对转移修复 14、引入 12，CHAIRs 差值 95% CI `[-12, 8]`。
- 虽然两个 CHAIR 指标方向有利，但长度下降 6.31%、generated object mentions 从约 1010 降到 960，超过预设 ±5% 门槛，且改善不显著。因此放弃 `remove + step500 + fixed top50` 这一精确版本，不进入正式实验。
- zero-query 版本得到 48.0/18.04/100.89，句级与词级均明显差于 VHR，长度仍下降约 6%。因此结构保持没有修复问题，说明当前 consensus score 本身没有识别出可靠视觉通路。
- **CC-VHR 路线已停止**：不再扫描 noise step、head fraction、倍率或正值门控，避免在同一开发集上追逐偶然配置。
- 下一独立假设改为“原 VHR 可靠选头 + 分支对称 VCD”。clean/noisy 使用同一 VHR 变换，先验证输出端视觉反事实是否与原 VHR 互补；若该受控组合仍不过门槛，则放弃 VCD 联合方向。
- 分支对称 VHR-VCD 开发结果为 42.0/21.43/107.41；虽然 CHAIRs 比 VHR 低 2 点，但 CHAIRi 从 15.84 大幅恶化到 21.43，幻觉提及约从 160 增至 222，且平均生成耗时从约 3.7 秒增至 8.2 秒/图。
- 最大失败样本 image 124279 出现长循环，单图 hallucinated mentions 从 VHR 的 57 增到 123；其余样本也出现餐具、球、车辆等共现物体密度上升。强噪声对比与 head 放大不是互补，而是在少数轨迹上共同放大错误对象链。
- **VCD 联合方向整体否决**：不再扫描 alpha、beta 或 noise step，也不进入正式实验。

## 12. VCD 否决后的新路线：Self-Contrast VHR

VHR-VCD 的失败表明图像噪声不是合适的第二分支，但“分支对比”本身仍可用于提取 VHR 干预方向。新方法在相同图像和相同生成历史下维护两个分支：一个启用原 VHR，另一个保持原模型 attention；使用

$$
\ell^*=(1+\alpha)\ell_{VHR}-\alpha\ell_{base}
$$

并以 VHR clean logits 设置 plausibility cutoff。两个分支唯一差别是 head intervention，因此差异比 clean/noisy VCD 更接近 VHR 路由的因果效应。

冻结的首轮配置为最后 18 层、原 magnitude top-50%、倍率 2、filter、`alpha=0.5`、`beta=0.1`。开发 100 图结果：

| 方法 | CHAIRs | CHAIRi | Len | hallucinated/generated mentions |
|---|---:|---:|---:|---:|
| VHR | 44.0 | 15.84 | 107.36 | 160 / 1010 |
| Self-Contrast VHR | **29.0** | **8.97** | 111.03 | 84 / 936 |

CHAIRs 配对差值 -15.0，95% CI `[-25,-5]`；修复 22、引入 7。长度增加 3.42%，而不是靠缩短输出取胜。对象提及减少 7.3%，但幻觉提及减少 47.5%，初步表明 precision 的提升远大于 coverage 损失。该结果只用于通过开发门槛，参数已冻结，下一步为独立 validation 100 图。

独立 validation 结果继续为正：VHR 42.0/17.63/110.05，Self-Contrast VHR **26.0/6.39/108.93**。CHAIRs 差值 -16.0，95% CI `[-25,-7]`；CHAIRi 差值的 95% CI `[-24.92,-1.55]`；修复 20、引入 4。对象提及仅下降 4.3%，幻觉提及从 193 降到 67。开发与验证均通过后，方法和超参正式冻结，进入 CHAIR 500 图 × 5 seeds。
