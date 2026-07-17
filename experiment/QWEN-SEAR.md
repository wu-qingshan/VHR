# SEAR：从 vanilla 出发的选择性对象证据句子保留

> 日期：2026-07-17
> 分支：`research/qwen-sear`
> 起点：`qwen-object-verifier-feasible-20260717`
> 当前状态：方法预注册；尚未安装 parser、尚未运行 caption 过滤

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
