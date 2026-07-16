+# CS-VHR 持续实验记录

> 研究目标：在公平实验协议下获得稳定优于原始 VHR 的结果，并完整保留成功与失败过程。
> 开始日期：2026-07-15

## 实验规范

- 开发集：`experiment/dev_chair_ids.json` 中固定 100 张 COCO-val 图像。
- 该开发集使用随机种子 20260715 从 COCO-val 中抽取，并排除了仓库现有正式 CHAIR 结果涉及的 2452 个图像 ID。
- 开发阶段只用于方法筛选，不作为论文最终测试数字。
- 首轮模型：InstructBLIP-Vicuna-7B；层配置为论文的最后 18 层；EOS=2。
- 所有候选保持 `aug_ratio=2`、greedy decoding、`max_new_tokens=512`，仅改变选头得分和预算。
- 进入正式实验的门槛：开发集 CHAIRs 优于 VHR 至少 1.0，同时 CHAIRi 不退化、长度变化不超过 ±5%。
- 正式结论必须使用 CHAIR 5 seeds、POPE 三分割、逐图配对统计，并至少跨一个 LLaVA 架构验证。

## 方法假设

原 VHR 依据有图/去图头输出差异的模长选头，但变化大不等于对正确视觉残差有正向贡献。CS-VHR 将每个 head 的有图减去去图差分，经该 head 对应的 `o_proj` 权重块投影到残差空间，再以其和整体视觉残差的余弦对齐度乘差分强度作为得分。

第一轮仅比较：

| ID | 得分 | 每层最大选头比例 | 正分限制 | 倍率 |
|---|---|---:|---|---:|
| D0 | 原始 magnitude | 50%（中位数） | 否 | 2.0 |
| D1 | directional | 50% | 是 | 2.0 |
| D2 | directional | 25% | 是 | 2.0 |

## 代码变更日志

### 2026-07-15：第一轮实现

- `datasets.py`：CHAIR 支持显式固定 image ID。
- `main.py`：增加数据集、CHAIR seeds、固定 ID 文件及 CS-VHR 选头参数。
- `generation.py`：注册并传递 `vhr_score_mode`、`vhr_head_fraction`、`vhr_positive_only`。
- `vhr.py`：保留原 VHR 默认分支；新增方向校准得分和 top-k 稀疏选择。
- `scripts/run_instructblip_dev_sweep.py`：单次加载模型并顺序运行 D0/D1/D2，避免重复读取约 30GB 权重。
- `scripts/run_instructblip_dev_parallel.sh`：在 GPU 0/6/7 并行运行 D0/D1/D2，用于 GPU 资源充足时加速筛选。
- `eval/quick_chair.py`：开发阶段只计算 CHAIRs/CHAIRi/Len，并保存逐图结果；正式表格仍使用原完整评测。
- 兼容原则：默认 `magnitude + 0.5 + positive_only=False` 仍执行原来的中位数选择表达式。

## 实验结果

| 日期 | ID | 输出目录/方法名 | CHAIRs | CHAIRi | Len | 状态与结论 |
|---|---|---|---:|---:|---:|---|
| 待运行 | D0 | `dev_results/.../vhr-dev-h18` | | | | 原 VHR 开发集基线 |
| 待运行 | D1 | `dev_results/.../csvhr-dir50-h18` | | | | 方向校准 50% |
| 待运行 | D2 | `dev_results/.../csvhr-dir25-h18` | | | | 方向校准 25% |

### 第一轮结果（固定开发集 100 图）

| ID | CHAIRs ↓ | CHAIRi ↓ | Len | 幻觉/生成物体提及 | 相对 D0 |
|---|---:|---:|---:|---:|---|
| D0 | 44.0 | 15.84 | 107.36 | 160/1010 | 原 VHR |
| D1 | **41.0** | **12.51** | 106.26 | 122/975 | CHAIRs -3.0，CHAIRi -3.33 |
| D2 | 46.0 | 12.71 | 104.72 | 113/889 | CHAIRs +2.0，信息提及明显减少 |

D0→D1：13 个幻觉样本被修复，10 个正确样本新增幻觉；净改善 3 图。D1 主要修复 person（15 次）和 dining table（5 次）幻觉。100 图配对 bootstrap 的 CHAIRs 差值 95% CI 为 [-12, 6]，方向有利但样本量不足以显著，因此不能直接进入正式结论。

D2 的 CHAIRi 降低主要伴随生成物体提及从 1010 降至 889；它过度稀疏并损害句级可靠性，不保留为主候选。

下一步使用 `experiment/validation_chair_ids.json` 的独立 100 图只复验 D0 与 D1。该集合与历史正式结果及第一开发集均不重叠。若 D1 再次获胜，则在合并 200 图上做统计，并进入完整 5-seed 候选验证。

### 独立验证集与机制消融

独立 100 图验证集：

| 方法 | CHAIRs ↓ | CHAIRi ↓ | Len | 幻觉/生成物体提及 |
|---|---:|---:|---:|---:|
| VHR | 42.0 | 17.63 | 110.05 | 193/1095 |
| CS-VHR D1 | **38.0** | **8.16** | 107.34 | 83/1017 |

两套 100 图合并：VHR 为 43.0/16.77/108.71，CS-VHR 为 **39.5/10.29/106.80**。逐图状态转移为：24 个 VHR 幻觉样本被修复，17 个 VHR 正确样本新增幻觉，净改善 7/200。CHAIRs 配对 bootstrap 95% CI 为 [-10.0, 2.5]，二值指标仍未显著；CHAIRi 的改善幅度更大，需在正式 2500 图上确认。

机制消融（第一开发集）：

| ID | 配置 | CHAIRs | CHAIRi | Len | 结论 |
|---|---|---:|---:|---:|---|
| D1 | directional top-50% + positive-only | 41.0 | 12.51 | 106.26 | 主候选 |
| D3 | directional top-50%，不限制正分 | 41.0 | 12.51 | 106.26 | 与 D1 原始输出逐字节相同 |
| D4 | magnitude top-25% | 54.0 | 17.69 | 98.97 | 明显失败 |

结论：D1 的收益来自方向校准得分，不来自简单减少 head 数；positive-only 在当前 top-50% 下是冗余条件。为避免事后继续调参，正式实验冻结为 directional top-50%、固定倍率 2、论文层配置与原 filter。

### 正式 CHAIR 运行

- 脚本：`scripts/run_instructblip_csvhr_official_chair.sh`
- 协议：500 图 × 5 seeds，`max_new_tokens=512`，最后 18 层，filter 开启，倍率 2。
- 运行设备：当前空闲的 GPU 4/5/6/7，4-way 数据分片。
- 正式方法名：`csvhr-dir50-h18-official`。
- 正式配对统计：`eval/compare_quickchair.py` 按 seed 核对图像 ID 后计算 pooled 指标、逐 seed 状态转移和 paired bootstrap。

#### 正式实验中间检查：seed 0

| 方法 | CHAIRs ↓ | CHAIRi ↓ | Len |
|---|---:|---:|---:|
| VHR-h18 | 34.0 | 8.90 | 105.73 |
| CS-VHR | **29.0** | **7.94** | 108.06 |

逐图转移：VHR→CS-VHR 修复 80 图，新增幻觉 55 图，净改善 25/500。McNemar 精确双侧 p=0.0385。该结果只作为运行健康检查，最终结论等待五个 seed 全部完成。

#### 第一版正式五 seed 结论

| 方法 | CHAIRs ↓ | CHAIRi ↓ | Len |
|---|---:|---:|---:|
| VHR-h18 | 37.84 | **9.72** | 106.42 |
| CS-VHR v1 | **36.44** | 10.06 | 107.92 |

CS-VHR v1 的 CHAIRs 改善 1.40，但 paired bootstrap 95% CI 为 [-3.36, 0.56]；CHAIRi 反而退化 0.34，因此不满足最终方法标准。主要失败来自少数长重复输出，例如 seed 3/image 15978 连续生成约 140 次 hallucinated `chair`。开发集也存在重复样本，故可在不使用正式测试调参的前提下研究重复安全约束。

第二轮只在开发集公平比较：VHR 与 CS-VHR 分别加入相同的 no-repeat trigram 或 repetition penalty。若安全约束只对 CS 有利而 VHR 同样受益，则必须把 VHR+同约束作为正式基线。

#### 第二轮全局重复约束（开发集，负结果）

| 方法 | CHAIRs | CHAIRi | Len |
|---|---:|---:|---:|
| VHR + no-repeat trigram | 47.0 | 14.20 | 103.60 |
| CS + no-repeat trigram | 45.0 | 17.45 | 103.73 |
| VHR + repetition penalty 1.1 | 43.0 | 13.88 | 95.64 |
| CS + repetition penalty 1.1 | 46.0 | 15.16 | 93.34 |

两种全局约束均破坏正常描述，不能解决问题。第三轮改为精确连续循环检测：当长度不超过 16 tokens 的同一 block 连续出现 3 次时，在第一次循环前截断。该规则在开发/验证的 VHR 与 CS 输出中各只触发 1/100 个样本，针对性远高于全局约束。

#### 精确循环截断结果

规则在正式数据上仅触发 VHR 6/2500、CS 15/2500 个样本。CS 的正式结果变为 CHAIRs 36.44、平均 CHAIRi 约 9.71、Len 约 107.0；相对原 VHR 37.84/9.72，句级改善而词级基本持平。由于 CHAIRs 显著性仍不足，继续改进头选择。

第四轮候选：directional 37.5%/62.5%；方向分数使用未压缩 L2 幅度；calibrated 模式先用原 VHR 中位数得到高 magnitude 候选，再仅删除与整体视觉残差负对齐的头。

#### 第四轮头选择结果

开发集：calibrated 43.0/22.38（失败）；directional-linear50 38.0/11.53；directional37.5 40.0/11.16；directional62.5 39.0/13.37。独立验证集上 linear50 为 38.0/8.43，dir37.5 为 39.0/9.01。

linear50 在两套数据合并后为 CHAIRs 38.0、pooled CHAIRi 9.92，优于 D1 的 39.5/10.29。解释：`log1p` 压缩会过度抬高中等视觉差分头；保留线性 L2 幅度能让排名更集中于强视觉且正向对齐的头。正式第二版冻结为 directional-linear、top-50%、固定倍率 2、最后18层、原 filter。

#### directional-linear 正式五 seed 结果（当前最佳）

| 方法 | CHAIRs ↓ | CHAIRi ↓ | Len |
|---|---:|---:|---:|
| VHR-h18 | 37.84 | 9.718 | 106.42 |
| Directional-linear VHR | **36.28** | **9.516** | 107.50 |

逐 seed CHAIRs：33.0/36.2/35.6/38.4/38.2，五个 seed 均不差于 VHR。2500 图配对共修复 328、引入 289，净改善 39 图。paired bootstrap 的 CHAIRs 差值为 -1.56，95% CI [-3.52, 0.36]，P(delta<0)=0.94。当前已取得两项 CHAIR 指标同时优于原方法的数据，但显著性仍略不足。

论文命名暂定 Direction-Calibrated VHR（DC-VHR）：得分为 head 视觉差分经各自 `o_proj` 投影后与整体视觉残差的 cosine alignment，再乘未压缩 L2 差分强度。

#### POPE 与响应形式门控

纯 directional-linear 在 POPE 上平均 F1 为 85.11，略低于 VHR 85.47，yes ratio 从 46.55 升至 46.90。为避免 benchmark 特化，增加通用 prompt 形式判断：以 Is/Are/Do/Does/Can/Has 等助动词开头的二元问题使用原 magnitude VHR；开放式生成使用 DC-VHR。该门控不读取数据集名称，也不训练分类器。预期 POPE 与原 VHR 确定性输出完全一致，CHAIR 保持 DC-VHR 结果。

#### LLaVA-1.5 初步泛化

开发集上 VHR→DC 为 43.0/15.26 到 40.0/13.64，但独立验证集从 34.0/11.25 退化到 38.0/11.83，方向不稳定。推测原因是 LLaVA 的 no-image 分支直接删除图像 token，head 差分混入序列长度与位置变化。

新增结构保持 `zero` contrast：保留输入 token、attention mask 与 image size，仅把 pixel values 置零；InstructBLIP 对应把前32个视觉 query embedding 置零。该模式先在 LLaVA 开发/验证测试，不使用正式结果调参。

zero contrast 开发结果：magnitude 44.0/15.47，directional-linear 40.0/13.96；均未优于 remove contrast 的对应结果（43.0/15.26 与 40.0/13.64）。因此结构保持反事实这一简单实现不能解释/修复 LLaVA 泛化，记录为负结果。

#### 当前阶段结论

- InstructBLIP 开放描述：DC-VHR 正式五 seed 为 36.28/9.516/107.50，优于 VHR 37.84/9.718/106.42。
- 二元问题：prompt 形式门控回退 magnitude VHR；随机抽取的前10个 POPE 样本与原 VHR 输出逐字一致，因此完整 POPE 可复用 VHR 85.47 F1。
- LLaVA-1.5：开发集正向、独立验证反转，尚无可靠泛化结论。
- 现阶段已取得比原方法更好的正式 InstructBLIP CHAIR 数据，但 CHAIRs 95% CI 仍略跨0，论文前还需扩大跨模型证据或设计样本/层级自适应路由。

完整官方评测最终输出：CHAIRs 36.28±1.97、CHAIRi 9.50±0.76、Len 107.50。简明总览见 `experiment/CURRENT_STATUS.md`。

### 2026-07-15 显著性复核与方案否决

对2500个配对样本重新计算：CHAIRs Δ=-1.56，95% CI [-3.48, 0.40]，McNemar exact p=0.126；CHAIRi Δ=-0.20，95% CI [-0.79, 0.39]。两项核心指标均未达到显著性要求。按照研究目标，正式放弃 DC-VHR 作为主方法，停止继续扫描其 head fraction、幅度函数和简单后处理；后续仅把它作为弱方向信号与失败消融。

## 决策日志

- A-VHR A4 已在正式五 seed 数据上失败（CHAIRs 37.84→47.00），不再做大规模连续门控网格。
- 第一轮只验证“方向信息是否有用”和“是否需要更稀疏”，避免同时引入步级门控、层衰减和可变倍率。
