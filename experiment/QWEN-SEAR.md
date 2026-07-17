# SEAR：从 vanilla 出发的选择性对象证据句子保留

> 日期：2026-07-17
> 分支：`research/qwen-sear`
> 起点：`qwen-object-verifier-feasible-20260717`
> 当前状态：300图dev已完成；幻觉指标通过但语义覆盖门槛失败，按预注册停止，未运行locked validation/test

## 1. 方法动机

Qwen vanilla 已明显优于 VHR/SC；对象 verifier 又取得 ROC-AUC 0.926，说明可靠方向是
保留 vanilla 主轨迹，只处理缺少具体视觉证据的对象声明。

固定0.5 verifier阈值会误拒45/531真实对象，不适合硬过滤。SEAR（Selective
Evidence-Anchored Retention）采用高置信 absent 阈值 `score<=0.1`：在已完成的
canonical-query诊断上标记39/86幻觉对象和20/531真实对象。该数字只用于冻结一个
保守工作点，不再扫描阈值。

## 2. 通用候选提取

最终方法禁止调用 CHAIR vocabulary/synonym evaluator。使用独立 spaCy English parser
`en_core_web_sm==3.8.0`：

1. 对 vanilla caption 做 sentence segmentation 与 noun chunks；
2. 候选 phrase 为 noun chunk 原文，去除开头 determiner `a/an/the/this/that/these/those`；
3. 用 chunk root lemma 去重，但保留 phrase 和所属 sentence；
4. 排除 pronoun root，以及固定 discourse stoplist：
   `image, photo, picture, scene, view, background, foreground, setting, area, side, color,
   lighting, atmosphere, impression, focus, design, sense, way, part, center, top, bottom`；
5. 不依据 CHAIR 结果添加或删除词。

在进入效果实验前，用 CHAIR 仅做离线 extraction audit：canonical object 的原始 mention
若落在任一候选 phrase 内则视为 captured。必须满足 unique mention recall `>=90%`，
且平均候选数 `<=20/image`。失败则停止当前 extractor，不运行过滤，也不依据漏词列表
手工扩充 stoplist。

## 3. 固定 verifier 与过滤

每个候选使用已冻结问题和 Yes/No logits，score定义不变。对每个句子：

- 若所有候选 score `>0.1`，逐字保留原句；
- 若任一候选 score `<=0.1`，删除整个句子；
- 无候选句子逐字保留；
- 若所有句子均被删除，保留“句内最低 object score最高”的一句；若所有句子均无对象，
  保留第一句。

不调用生成式 rewrite，不修改保留句内部文本，不做二次阈值、对象白名单或数据集判断。
因此效果完全来自“高置信对象证据 + 最小句子选择”，而非重新生成更短caption。

## 4. 工程与提取门槛

1. spaCy版本与模型版本记录；
2. 保留句文本必须是原caption的精确substring；
3. 同一caption候选顺序与过滤结果确定；
4. verifier结果与独立617-query runner对相同 `(image,phrase)` 一致；
5. 过滤结果非空；
6. extraction audit recall `>=90%`，平均候选 `<=20/image`。

未全部通过，不运行300图效果。

## 5. dev 效果门槛

只处理封存 vanilla 300图，不重跑 vanilla/VHR/SC/SRR。输出到独立
`qwen_sear_dev_results/`，保持 untracked。必须同时满足：

1. CHAIRs `<=20.67`，相对 vanilla 至少下降3点；
2. CHAIRi `<=6.46`，不得差于 vanilla；
3. total unique correct objects 至少保留 vanilla 的95%；
4. Len 在 vanilla 的 `[0.75,1.05]` 倍；
5. exact cycle 不高于 vanilla；
6. 512-token hit 不高于 vanilla；
7. 平均 verifier+过滤时延低于标准 SC 的25.29秒/图；
8. CHAIRs paired bootstrap方向为负；
9. 报告删除句比例、触发图比例、被删除真实/幻觉对象数和失败案例。

任一失败：停止，不运行 locked validation，不扫描 threshold/stoplist/parser规则。全部
通过后冻结实现，才允许在未查看的 locked validation 500图运行一次。

## 6. 论文定位

SEAR 把 VHR 的“增强视觉依赖”改写成更直接的假设：只有具体对象获得独立视觉存在性
证据时才保留相关声明。它不依赖 attention head、GQA、mRoPE或视觉token布局，理论上
可迁移到不同 MLLM；同时保留 verifier error、coverage 和额外成本的可审计接口。

若 SEAR 只靠大幅删句降低 CHAIR、无法保留95%真实对象，则不具备较高质量论文价值。

## 7. 实验日志

- 2026-07-17：从 verifier feasible tag 创建独立分支，冻结 parser、stoplist、0.1阈值、
  非生成式句子保留规则和严格dev门槛；尚未安装parser模型、尚未运行SEAR。
- 2026-07-17：仅在`.venv-qwen`安装`spaCy 3.8.11`与
  `en_core_web_sm 3.8.0`。首次extraction audit在产生结果前因隔离环境缺少NLTK
  `punkt_tab`退出；资源随后只安装到`.venv-qwen/nltk_data`，原命令重跑。
- 2026-07-17：extraction audit通过。300图共720个图内唯一CHAIR lexical mention，
  捕获672个，recall为**93.33%**；617个唯一图像-对象对捕获581个，recall为
  94.17%。共5750个root-deduplicated通用候选，平均19.17/图，满足不超过20的门槛。
- 2026-07-17：发现5个由Qwen异常循环造成的超长noun chunk，最长472词。未根据结果
  裁剪候选或修改parser；工程smoke确认最长query logits/score有限、重复误差为0，且
  与封存canonical verifier共享样本的分数误差为0。
- 2026-07-17：六GPU完成5750/5750通用短语query，严格合并后应用冻结0.1整句删除规则。
  直接以文件路径调用quick-CHAIR时发生`eval.py`包名遮蔽，评测前退出；改用
  `python -m eval.quick_chair`运行同一历史评测器，没有修改评测代码。

## 8. Extraction与工程结果

| 项目 | 结果 | 门槛 |
|---|---:|---:|
| unique lexical mention recall | **93.33%** | ≥90% |
| unique image-object recall | 94.17% | 报告 |
| 平均候选/图 | **19.17** | ≤20 |
| P95 / 最大候选/图 | 38 / 55 | 报告 |
| verifier queries | 5750/5750 | 完整 |
| 平均extraction时延 | 0.0386秒/图 | 报告 |

审计中的CHAIR词表只用于测量捕获率；`queries.json`完全由spaCy noun chunks生成，manifest
明确记录`chair_vocabulary_used=false`。

## 9. 300图dev正式结果

| 方法 | CHAIRs ↓ | CHAIRi ↓ | Len | unique correct | exact cycles | 512 source hit |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 23.67 | 6.46 | 147.48 | 531 | 50 | 29 |
| SEAR | **17.00** | **5.69** | 120.05 | 486 | 37 | 29 |

SEAR相对vanilla的CHAIRs差值为**-6.67**，paired bootstrap 95% CI
`[-9.67,-4.00]`，方向概率为1.0。总pipeline为10.50秒/图，其中verifier+提取+过滤为
4.62秒/图，低于SC的25.29秒/图。

但核心coverage门槛失败：unique correct objects仅保留
`486/531=91.53%`，低于预注册95%。170/300图触发删除，共删除311/2459句
（12.65%）；这些删除只移除28次幻觉object mention，却同时移除243次正确mention；按
unique object计为23个幻觉对象与45个正确对象。整句删除的主要作用不是精准移除错误，
而是连带清除同句大量真实内容。

| dev gate | 结果 |
|---|---|
| CHAIRs ≤20.67 | 通过 |
| CHAIRi不差于vanilla | 通过 |
| unique correct ≥95% vanilla | **失败（91.53%）** |
| Len为vanilla的0.75–1.05倍 | 通过 |
| cycle/max-hit不恶化 | 通过 |
| SEAR-only时延低于SC | 通过 |
| paired CHAIRs方向为负 | 通过 |

总判定：**失败，停止在dev；不运行locked validation/test，不扫描threshold、stoplist或
parser规则。**

## 10. 机制结论与下一方向

对象verifier信号本身仍然成立，但“任一低分对象→删除整句”的决策粒度过粗。一句中常
同时包含一个低分候选和多个真实对象，因此高precision absent threshold仍会造成严重
语义连带损失。下一候选必须采用新的未使用dev划分，研究phrase/clause级局部修订，并将
vanilla作为安全锚点；本次300图只能用于确认整句粒度失败，不能继续选择阈值或语法规则。

结果SHA256：

- extraction audit：`6480e461ea90c2dbbd947c0bcf7f22dcc765ee0de1320055bb778f6d218a004b`
- queries：`9c8577d8117d399247a57b0c6089bc4c505de941c647b62f883cf72252027a20`
- verifier：`0a864b76712dbcd76f8c33d312496b9cbfddf3d04f5c163f66993d0a4fe7e50c`
- SEAR captions：`176f9ccd51cbb6553eb912904f21f437acfd508ce194842812cd4ab2f39ec424`
- analysis：`8a34dd39f56453506ba367425f0493ae4f40315b8238808036bdc34edd324711`
