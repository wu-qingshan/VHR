# Qwen 对象证据校准与 caption 修订路线

> 日期：2026-07-17
> 分支：`research/qwen-object-evidence`
> 起点：`qwen-srr-dev-negative-20260717`
> 当前状态：query构造与 verifier 工程门槛通过；尚未运行617-query feasibility

## 1. 为什么必须离开 VHR 默认轨迹

Qwen dev 已证明：

- vanilla CHAIRs/CHAIRi/Len 为 `23.67/6.46/147.48`；
- VHR/SC 为 `29.67/8.26/315.60` 与 `27.00/7.93/345.63`；
- SRR 虽把 exact cycle 降为 0，仍只有 `27.00/7.15/223.26`。

这说明现代 Qwen 的主要问题不是“缺少更强视觉 head 放大”，而是视觉 steering 会破坏
已经较强的 base。下一方法以 vanilla caption 为主轨迹，只针对其中的候选对象建立
独立视觉证据，再做最小修订。

目标方法暂称 **OER（Object-Evidence Revision）**：

1. vanilla 生成完整 caption；
2. 用通用 noun-phrase parser 提取可见对象候选；
3. 同一 MLLM 对每个候选回答固定 Yes/No 存在性问题，以 logit margin 提供证据；
4. 只删除或修订缺少视觉支持的对象相关内容；
5. 保持其他句子、对象覆盖和 base 流畅性。

在实现修订前，必须先证明步骤3的信号能区分真实/幻觉对象。

## 2. verifier feasibility 数据与标签

只使用已经打开的 Qwen dev 300 图和封存 `qwen_dev_results/vanilla.json`，不读取 locked
validation/test。

- 使用 CHAIR evaluator 对 vanilla caption 提取 canonical object node words；
- 同图内相同 object 去重，形成 `(image_id, object)` query；
- 若 object 在该图 CHAIR ground-truth object set 中，label=1，否则 label=0；
- CHAIR 对象词只用于本阶段测量 verifier 信号，不作为最终 OER 的候选提取器；正式方法
  必须改用通用 noun phrase parser，避免 benchmark vocabulary 泄漏。

所有300图一次性评估，不按 verifier 结果选择子集。

## 3. 固定 verifier

对每个 query 使用原图和固定问题：

> `Is there a visible {object} in this image? Answer only Yes or No.`

模型为同一 `Qwen2.5-VL-7B-Instruct`、原生 vanilla、bfloat16/eager，不安装 VHR patch，
不生成长回答。取 assistant 位置首 token logits：

- `Yes` token id `9454`；
- `No` token id `2753`；
- score = `sigmoid(logit_yes - logit_no)`。

阈值固定为0.5，不使用 CHAIR label 调阈值，不聚合大小写或前导空格 token，不扫描 prompt。
每个样本后清除 `rope_deltas`，并检查所有 score finite。

## 4. 工程门槛

1. 已知正例 dog 与已知负例 elephant 的 score 均 finite，且 dog score > elephant score；
2. 同一 query 重复两次 score 绝对误差 `<=1e-6`；
3. 两张不同 image-grid 图顺序 A->B 与 fresh B 一致；
4. 首 token top candidates 与 Yes/No token id 审计写入结果；
5. verifier 不修改 vanilla/VHR/SC/SRR 原始输出。

工程门槛未通过，不运行300图 feasibility。

## 5. feasibility 统计门槛

固定报告：query总数、正/负数量、ROC-AUC、PR-AUC、阈值0.5的 accuracy、balanced
accuracy、precision/recall/F1、正负 score 均值/中位数、bootstrap mean gap 95% CI、
秒/query和峰值显存。

必须同时满足：

1. query总数 `>=500`；
2. hallucinated/negative query `>=60`；
3. ROC-AUC `>=0.75`；
4. threshold=0.5 balanced accuracy `>=0.70`；
5. `mean(score_positive)-mean(score_negative)` 的 95% CI 下界 `>0`。

任一失败：停止自验证路线，不实现 caption 修订、不扫描 prompt/threshold。全部通过后才
安装固定 spaCy English parser，预注册 OER 的通用候选提取与最小修订规则，并在新的
OER 输出目录运行 dev；locked validation 仍保持未查看。

## 6. 论文潜力与限制

该方向比继续 VHR gate 更有潜力，因为它：

- 从现代模型最强的 vanilla 输出出发；
- 将“视觉依赖强”改为“具体对象存在证据充分”；
- 可跨 attention 架构，不依赖 GQA head id 或视觉 token 布局；
- 能直接分析 false verification、coverage 和计算成本。

风险是同一模型可能自我确认其语言先验，Yes/No score 未必能区分幻觉；本阶段 gate 就是
为了在投入修订实验前否决这种情况。

## 7. 实验日志

- 2026-07-17：从 SRR dev negative tag 创建独立分支，冻结 query、token id、score、
  样本范围和停止门槛；尚未运行 verifier。
- 2026-07-17：从 vanilla 300 图构造 617 个同图去重对象 query，真实531、幻觉86，
  覆盖281图；数量门槛通过。query 文件保持在未跟踪
  `qwen_object_evidence_results/queries.json`。
- 2026-07-17：工程 smoke 通过。dog/elephant/giraffe score 分别为
  `0.9820/0.000031/0.99945`；dog 与 giraffe 重复 score 误差为0；动态 grid 为
  `1x34x46` 与 `1x26x36`；正例首 token top1 为 Yes，负例 top1 为 No。

工程 tag：`qwen-object-verifier-engineering-ready-20260717`。下一步只运行预注册的
617-query feasibility，不生成或修改 caption。

## 8. 617-query feasibility 结果

六个 GPU chunk 全部 exit 0，严格合并 617/617 query。结果目录保持 untracked：

- queries SHA256：`c8c6c59e67135e4308226d7b1ca2dcf7241c5ac5718b8062b9139b627af8e036`；
- verifier SHA256：`ded1cf9e1c1a308c8675320d6e05a6473b83e6c75a2145b6b5a47f64c37488f8`；
- analysis SHA256：`c0a941d812b4eef6770e8f090a64ca26684d17faf81e8d99e830f2d6d62779c5`。

| 指标 | 结果 |
|---|---:|
| queries（正/负） | 617（531/86） |
| ROC-AUC | **0.9260** |
| PR-AUC | **0.9874** |
| Accuracy | 0.8817 |
| Balanced accuracy | **0.7948** |
| Precision / Recall / F1 | 0.9455 / 0.9153 / 0.9301 |
| TN / FP / FN / TP | 58 / 28 / 45 / 486 |
| positive score mean/median | 0.8935 / 0.9933 |
| negative score mean/median | 0.3063 / 0.1261 |
| mean gap | 0.5872 |
| mean gap 95% CI | **[0.5128,0.6581]** |
| 秒/query | 0.2471 |

五项预注册 gate 全部通过，结论为 **verifier feasible，允许实现 OER**。

但 threshold=0.5 仍有45个真实对象 false negative 和28个幻觉 false positive，不能把
“score<0.5”机械等同于删除整个句子。错误集中于小物体/局部物体（traffic light、
laptop、tie）以及词义歧义（如 orange 可表示颜色或物体）。因此 OER 必须：

1. 使用通用 noun phrase 的完整短语和上下文，而不是仅 canonical 单词；
2. 将 verifier 作为证据而非唯一标签，避免低分直接删除含多个真实对象的整句；
3. 优先做最小 phrase/clause 修订，并重新验证修订后新对象；
4. 单独报告 false verification、对象 recall、句子删除率和额外计算。

本阶段固定 tag：`qwen-object-verifier-feasible-20260717`。下一阶段需另行预注册 OER
候选提取、证据使用和修订规则，不能根据 locked validation 选择 threshold。
