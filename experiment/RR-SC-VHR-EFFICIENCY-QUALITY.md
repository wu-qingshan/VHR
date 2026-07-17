# RR-SC-VHR 效率与文本质量评测

> 日期：2026-07-17
> 分支：`research/rr-scvhr-eval`
> 起始检查点：`c18c015`（tag: `rrscvhr-official-20260717`）
> 状态：统一 benchmark、等价性审计、完整 COCO/CHAIR evaluator 和文本稳定性评测均已完成。

## 1. 目的

补齐论文仍缺少的两类证据：

1. VHR、纯 SC、RR 在统一硬件和同一模型加载下的时延、吞吐和峰值显存；
2. 正式 CHAIR 输出的 COCO caption quality、重复率与异常循环，验证 SC/RR 的提升没有明显破坏文本质量。

## 2. 统一效率协议

- 模型：InstructBLIP-Vicuna-7B，同一次加载、单 GPU、同一 Python 进程；
- 配置：最后 18 层、magnitude top-50%、ratio 2、filter、`alpha=0.5`, `beta=0.1`；
- 开放描述：`experiment/dev_chair_ids.json` 前 10 图；
- 二元任务：POPE popular 前 100 题；
- 先分别 warm up VHR、SC、RR 各 1 样本，不计入结果；
- 两轮测量采用正序 VHR→SC→RR 和反序 RR→SC→VHR；
- 每次测量前 `empty_cache/reset_peak_memory_stats/synchronize`；
- 记录：wall seconds、seconds/sample、samples/s、baseline allocated/reserved、peak allocated/reserved、增量 peak；
- 模型加载时间不计入单方法 latency。

完整性要求：

- 同方法两轮输出逐字一致；
- CHAIR RR 与 SC 逐字一致；POPE RR 与 VHR 逐字一致；
- 输出目录全新且拒绝覆盖；
- 报告均值及两轮原始数值，不只报告最佳轮。

## 3. 文本质量协议

正式 500×5 CHAIR，比较 VHR 与 SC/RR：

- METEOR、CIDEr、Bleu（原 COCO/CHAIR evaluator）；
- 平均长度、对象 mentions、unique correct objects（已有记录）；
- corpus/per-caption distinct-1/2；
- repeated 2/3/4-gram ratio；
- 连续 exact cycle（周期≤16、至少重复3次）命中率。

RR CHAIR 与 SC 完全一致，因此质量指标只需评测 VHR 与 RR/SC 两组，但必须明确该复用关系。

## 4. 停止与披露规则

- 若 benchmark 输出不等价，整批效率数据作废并定位状态污染；
- 若两轮时延差异超过 20%，报告波动并增加一轮，而不是选择较快结果；
- 峰值显存为 PyTorch allocator 指标，不等价于 `nvidia-smi` 进程总占用，论文中注明；
- COCO caption metrics 与详细描述任务可能相关性有限，只作为质量补充，不替代人工评估。

## 5. 执行日志

- 2026-07-17：审计现有日志，发现纯 SC POPE 四 GPU chunk 为 0.90–0.97 秒/题，但 VHR/RR 仅有小样本或不同环境记录，不能公平比较；建立上述统一协议。
- 2026-07-17：统一 benchmark 完成，共 12 次正式测量。VHR、SC、RR 在 CHAIR/POPE 上各两轮输出逐字段一致；CHAIR RR=SC、POPE RR=VHR。所有两轮 latency relative range 均小于 2%，未触发 20% 的第三轮规则。
- 2026-07-17：正式 RR CHAIR 五个 seed 运行完整 COCO/CHAIR evaluator，结果文件写入 `rrscvhr_official_results/` 的独立 `chairmetrics_*`，未改写生成文件。
- 2026-07-17：新增 `scripts/analyze_caption_quality.py`，完成 2500 对 VHR/RR caption 的 distinct、repeated n-gram 和 exact cycle 配对统计。

## 6. 统一效率结果

### 6.1 CHAIR 开放描述（10 图/轮）

| 方法 | 两轮秒/图 | 均值秒/图 | 吞吐图/秒 | peak allocated GiB | 增量 peak GiB |
|---|---:|---:|---:|---:|---:|
| VHR | 4.2578 / 4.2269 | **4.2424** | 0.2357 | 15.0453 | 0.2632 |
| SC | 8.7860 / 8.8276 | **8.8068** | 0.1135 | 15.1805 | 0.3983 |
| RR | 8.6336 / 8.7871 | **8.7104** | 0.1148 | 15.1805 | 0.3983 |

- SC/VHR latency 为 **2.076×**；RR/VHR 为 **2.053×**。
- RR 与 SC 使用相同开放描述路径，均值差约 1.1%，属于运行波动；显存峰值完全相同。
- 最大两轮 relative range 为 RR 的 1.76%，不增加第三轮。

### 6.2 POPE binary（popular 前100题/轮）

| 方法 | 两轮秒/题 | 均值秒/题 | 吞吐题/秒 | peak allocated GiB | 增量 peak GiB |
|---|---:|---:|---:|---:|---:|
| VHR | 0.4974 / 0.4938 | **0.4956** | 2.0178 | 14.8598 | 0.0776 |
| SC | 0.9067 / 0.9202 | **0.9135** | 1.0948 | 14.8998 | 0.1176 |
| RR | 0.4937 / 0.5001 | **0.4969** | 2.0126 | 14.8598 | 0.0776 |

- 纯 SC 相对 VHR latency 为 **1.843×**。
- RR 与 VHR latency 比为 **1.003×**，实测等价；相对纯 SC **加速 1.838×**，即只使用其 54.4% 的时延。
- RR 的 total peak allocated 与 VHR 完全相同；相对 SC 少约 0.0400 GiB。由于 7B 模型权重占主导，总峰值差小；增量 peak 从 0.1176 降到 0.0776 GiB。
- 最大两轮 relative range 为 SC 的 1.49%，不增加第三轮。

### 6.3 等价性

比较字段为 `question_id/text/gt/ans_conf/question`：

1. CHAIR/POPE 的 VHR、SC、RR 各自 cycle0 与 cycle1 全部一致；
2. 两轮 CHAIR 均为 RR 与 SC 逐字段一致；
3. 两轮 POPE 均为 RR 与 VHR 逐字段一致。

因此本次效率数字对应的确是预期函数路径，不存在用不同输出换取速度的情况。原始汇总为 `rrscvhr_benchmark_results/benchmark_summary.json`，确定性审计为 `benchmark_analysis.json`。

## 7. COCO caption quality

完整 evaluator 的五 seed 均值（×100）：

| 方法 | BLEU-1 | BLEU-4 | geometric BLEU | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|---:|---:|
| VHR | 14.86 | 3.90 | **7.84** | **16.37** | **16.61** | 0.13 |
| RR/SC | 13.32 | 3.16 | **6.71** | **14.84** | **15.89** | 0.25 |
| RR−VHR | −1.53 | −0.74 | −1.13 | −1.53 | −0.72 | +0.13 |

CHAIR 的 COCO reference 平均长度仅约 13 token，而生成的详细描述约 106–113 token，二者任务形式并不完全匹配，导致绝对 BLEU/CIDEr 很低。但相对变化仍显示：SC/RR 的 CHAIR 幻觉下降伴随传统 caption overlap 指标下降，不能声称文本质量全面提升。

## 8. 长度、多样性与重复

采用小写 Unicode word token；per-caption 指标先对每条 caption 计算再求均值。

| 方法 | words/caption | distinct-1 ↑ | distinct-2 ↑ | repeated-2 ↓ | repeated-3 ↓ | repeated-4 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| VHR | 106.56 | **0.5096** | **0.8203** | **0.1797** | **0.0717** | **0.0348** |
| RR/SC | 112.87 | 0.4973 | 0.7944 | 0.2056 | 0.0992 | 0.0592 |
| RR−VHR | +6.31 | −0.0124 | −0.0259 | +0.0259 | +0.0275 | +0.0244 |

corpus-level distinct-1/2 由 0.01419/0.11816 变为 0.01427/0.12864，说明更长输出覆盖了更多全局词组；但单条 caption 内的 distinct 降低、重复率上升。两种统计回答的问题不同，论文应以 per-caption 结果判断局部退化，不能用 corpus-level 多样性掩盖重复。

连续 exact cycle 定义为：case-sensitive whitespace-token block、周期≤16、连续出现至少3次。

| 方法 | cycle captions | hit rate |
|---|---:|---:|
| VHR | 6/2500 | 0.24% |
| RR/SC | 33/2500 | 1.32% |

配对列联为 neither=2462、VHR-only=5、RR-only=32、both=1；McNemar exact `p=7.43e-6`。绝对比例仍低，但增幅显著，构成真实的生成稳定性负结果。

## 9. 综合评价

RR-SC-VHR 的结论应拆成三层：

1. **幻觉与任务适配成立**：开放描述 CHAIR 显著改善，binary POPE 精确恢复 VHR；
2. **效率主张成立**：开放描述成本约 2× VHR，binary 路由恢复 VHR 成本并比纯 SC 快 1.84×；
3. **文本质量并非无损**：输出更长，COCO overlap 指标下降，per-caption 重复和 exact cycle 增加。

因此 RR 是当前最可靠的任务路由方案，但下一阶段应研究只在 SC 引入语言不稳定时回退 VHR 的 token-level stability routing。该方向必须在新的开发/锁定验证划分上评测，不能根据本次正式 CHAIR 测试集继续调阈值。
