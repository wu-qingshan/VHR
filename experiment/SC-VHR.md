# Self-Contrast VHR（SC-VHR）实验记录

> 日期：2026-07-15  
> 状态：InstructBLIP 正式 CHAIR 与 POPE 已完成；CHAIR 显著提升，POPE 小幅但稳定退化；跨模型验证待完成。

## 1. 方法

原 VHR 通过 clean/text-only prefill 选择视觉敏感 heads，并在后续解码固定放大。SC-VHR 在同一图像、同一 prompt、同一生成 token 历史下额外维护一个原模型分支：

- 主分支：启用原 VHR 选头与倍率；
- 反事实分支：同图像、同 KV 历史，但 attention 不应用 VHR head intervention；
- 两分支唯一变量是 VHR 干预本身。

最终 logits：

$$
\ell^*=(1+\alpha)\ell_{VHR}-\alpha\ell_{base}
$$

并以 $\ell_{VHR}$ 的最大值设置 plausibility cutoff。冻结配置：InstructBLIP 最后 18 层、原 magnitude top-50%、倍率 2、原 filter、`alpha=0.5`、`beta=0.1`。

## 2. 形成过程与否决路线

1. DC-VHR 正式改善不显著：CHAIRs -1.56，95% CI 跨 0，放弃。
2. CC-VHR remove：42.0/15.31/100.59，长度下降 6.31%，放弃。
3. CC-VHR zero：48.0/18.04/100.89，明显退化，整个 consensus score 路线放弃。
4. symmetric VHR-VCD：42.0/21.43/107.41，CHAIRi 大幅退化且耗时翻倍，VCD 联合路线放弃。
5. SC-VHR 使用同图 base 分支替代 noisy-image 分支，直接提取 VHR 干预的因果 logit 方向。

上述失败后没有扫描 noise step、head fraction、倍率或 alpha/beta；SC-VHR 首轮配置通过开发集后即冻结。

## 3. 独立开发与验证

| 数据 | VHR CHAIRs/CHAIRi/Len | SC-VHR CHAIRs/CHAIRi/Len | CHAIRs 差值 95% CI | 转移（修复/引入） |
|---|---|---|---|---|
| dev 100 | 44.0 / 15.84 / 107.36 | **29.0 / 8.97 / 111.03** | `[-25,-5]` | 22 / 7 |
| validation 100 | 42.0 / 17.63 / 110.05 | **26.0 / 6.39 / 108.93** | `[-25,-7]` | 20 / 4 |

validation 的 CHAIRi 差值 95% CI 为 `[-24.92,-1.55]`。两套图像 ID 完全不重叠，且 validation 前没有改参数。

## 4. 正式 CHAIR：InstructBLIP-Vicuna-7B

500 图 × 5 seeds，prompt、EOS、最大长度和原 VHR 完全一致。

| seed | VHR CHAIRs | SC-VHR CHAIRs | VHR CHAIRi | SC-VHR CHAIRi | VHR Len | SC-VHR Len |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 34.0 | **30.0** | 8.90 | **8.28** | 105.73 | 110.30 |
| 1 | 36.8 | **30.8** | 8.84 | **7.50** | 104.95 | 114.09 |
| 2 | 38.4 | **28.8** | 10.17 | **10.07** | 108.52 | 112.87 |
| 3 | 39.4 | **27.8** | 9.73 | **7.21** | 105.98 | 111.92 |
| 4 | 40.6 | **34.6** | 10.94 | **9.05** | 106.93 | 114.67 |
| mean | 37.84 | **30.40** | 9.72 | **8.42** | 106.42 | 112.77 |

逐图配对统计（2500 图）：

- CHAIRs 差值：**-7.44**，95% CI `[-9.48,-5.40]`；McNemar exact `p=1.10e-12`。
- CHAIRi 差值：**-1.30**，95% CI `[-2.26,-0.08]`。
- 状态转移：正确保持 1305；新增幻觉 249；修复 435；持续幻觉 511。
- Len 差值：+6.34 words，95% CI `[4.56,8.17]`，相对增长 5.96%。
- generated object mentions：23790 → 25267（+6.21%）。
- hallucinated mentions：2312 → 2128（-7.96%）。

因此 SC-VHR 不是靠减少信息量或缩短回答取得收益：输出更长、对象提及更多，同时句级和词级幻觉都显著下降。代价是约两倍解码计算和约 6% 的文本长度增长。

## 5. CHAIR 阶段判断

这是目前第一个在正式 2500 图上同时满足以下条件的改进：

1. CHAIRs 显著优于原 VHR；
2. CHAIRi 显著优于原 VHR；
3. 五个 seed 方向一致；
4. 信息覆盖没有下降；
5. 开发、独立验证和正式测试方向一致。

## 6. 正式 POPE：存在性判断代价

三分割各 3000 题，使用与 CHAIR 完全相同的冻结 SC-VHR 配置，没有二元问题回退。

| 方法 | Acc | Precision | Recall | F1 | YesRate |
|---|---:|---:|---:|---:|---:|
| VHR | **85.91** | 88.86 | **82.47** | **85.47** | 46.55 |
| SC-VHR | 85.14 | **89.58** | 79.78 | 84.34 | 44.63 |

分割明细：

| 分割 | VHR F1 | SC-VHR F1 | 差值 | bootstrap 95% CI |
|---|---:|---:|---:|---|
| popular | 84.99 | 84.05 | -0.94 | `[-1.54,-0.36]` |
| adversarial | 82.88 | 82.22 | -0.66 | `[-1.31,-0.04]` |
| random | 88.55 | 86.76 | -1.79 | `[-2.41,-1.19]` |

SC-VHR 的 precision 增加 0.72，但 recall 下降 2.69，yes rate 下降 1.92。说明 self-contrast 会强化 VHR 的保守方向：更少把不存在的物体答为 yes，同时也漏掉更多真实存在物体。POPE 不是“基本持平”，而是平均 F1 下降 1.13；其中 random 分割 accuracy 下降 1.50，McNemar `p=3.74e-8`。

## 7. 最终评价

当前证据支持以下窄结论：

> SC-VHR 显著改善 InstructBLIP 的开放式对象幻觉，并且不是通过缩短回答或减少对象覆盖实现；但它进一步推低存在性决策阈值，在 POPE 上牺牲 recall 和 F1。

优点：

1. 开发、独立验证、正式测试方向完全一致。
2. 正式 CHAIRs/CHAIRi 均显著，五个 seed 全部改善。
3. 反事实只改变 VHR intervention，因果解释比 noisy-image VCD 更干净。
4. 输出更长、对象提及更多，排除了“少说少错”的主要混杂。

不足：

1. POPE 三分割 F1 全部下降，方法存在明确的保守偏置。
2. 每步需要 VHR/base 两次 decoder forward 和两套 KV cache，实测约为 VHR 的 2.1–2.3 倍时延。
3. 正式平均长度增加 5.96%，需补充 caption quality、重复率和人工质量评估。
4. 目前只有 InstructBLIP，不能声称跨架构泛化。
5. 与一般 contrastive/self-contrast decoding 的关系需要在论文中清晰区分，并补 VCD、M3ID/ICD 类基线讨论。

因此 SC-VHR 应保留为论文主候选，但不能把实验阶段视为完全结束。InstructBLIP 核心阶段已结束；投稿前至少还需 LLaVA 跨模型、质量与效率评测。若采用“开放式问题用 SC-VHR、二元问题回退 VHR”的通用响应形式路由，POPE 可理论上恢复原 VHR，但必须作为明确的方法组件和消融，而不能隐藏当前纯 SC-VHR 的负结果。
