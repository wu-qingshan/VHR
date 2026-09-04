# MGR-CEILING 预注册：E0.1 oracle 上界、E0.2 人工触发审计、E0.3 grounding smoke

> 日期：2026-09-04
> 分支：`research/qwen-mgr`（自 `qwen-vera-dev-negative-20260718` 创建）
> 状态：预注册冻结，未运行任何 E0.1/E0.2/E0.3 代码
> 上游计划：`nextplan.md` 第 3 节（阶段 0）

## 0. 目的与纪律

VERA 失败点被诊断为通用 noun-chunk 触发与真实幻觉错位（336 触发句仅 39 句含 CHAIR
幻觉）。本预注册在投入任何 MGR 方法代码之前，用一周内的三个测量决定修订路线的天花板：

- E0.1：完美触发条件下 mention 级修订的 CHAIR/覆盖天花板（oracle，explanation-only）；
- E0.2：noun-chunk 触发的人工 precision 审计（本人标注，不可委托）；
- E0.3：Qwen2.5-VL 原生 grounding 的工程与定位质量 smoke。

纪律约束（全阶段适用）：

1. 只使用已打开数据：`qwen_vera_dev_results/`（300 图 vanilla 物化输出）及其拆分
   `experiment/qwen_vera_dev_chair_ids.json`。locked validation/test 不生成任何输出。
2. E0.1 使用 CHAIR GT 标签构造理想触发，属于 `posthoc_explanation_only=true` 与
   `must_not_tune_current_dev=true`，与 VERA 对齐诊断同级：结果只用于天花板判断与
   阶段 2 门槛校准，不得反推任何触发阈值、prompt、集成权重或算子选择参数。
3. MGR 方法本身（阶段 1 起）禁止使用 CHAIR 词表参与触发、生成、接受或排序；本文件
   中一切 CHAIR 用法均为离线审计。
4. 不修改任何历史结果文件；新输出全部写入 `mgr_ceiling_results/`。
5. 任何门槛数值只引用本文件与 `nextplan.md`，不得事后新增或改写。

## 1. E0.1 oracle 上界实验

### 1.1 输入

- 300 图 vanilla caption：`qwen_vera_dev_results/vanilla.json`（JSONL，
  `question_id` + `text`，与 VERA 正式 dev 分析同源，不重新生成）。
- 提取器：SEAR/VERA 已冻结的 `spaCy 3.8.11 + en_core_web_sm 3.8.0`，
  `qwen_sear.extract_caption` + `qwen_vera.sanitize_extraction`（root/phrase 至少
  含一个 ASCII 字母）。
- GT：`datasets.CHAIRBench` 冻结评测器（`eval/chair.py` + `eval/synonyms.txt`）。

### 1.2 oracle 触发构造（精确规则）

对每张图 i：

1. 记 `words_i, node_words_i = evaluator.caption_to_words(vanilla_text_i)`；
   `H_i = {n ∈ node_words_i : n ∉ gt_objects_i}`（CHAIR 幻觉 node word 集合）。
2. 对提取候选 c（root lemma 去重后的 query 级）定义其 CHAIR 映射：
   `node(c) = inverse_synonym_dict[root_lemma(c)]` 若 `root_lemma(c) ∈ mscoco_objects`，
   否则 `node(c) = None`。
3. **oracle 触发**：`node(c) ∈ H_i`。该 root 的全部 occurrence 一并触发。
4. 记录幻觉覆盖率：`captured_i = |{n ∈ H_i : ∃c, node(c)=n}| / max(1,|H_i|)|`，
   即 root 级提取对幻觉 node word 的召回上限，进天花板表。

oracle 是"完美分类器"假设：真实方法只能逼近，不能超过。所有幻觉判定基于 CHAIR
GT，永不进入方法组件。

### 1.3 三算子（固定，不新增、不扫描）

**算子 A（mention 删除 + 最小修复）**：对每张图，把全部 oracle 触发 occurrence 的
字符区间 `[start_char, end_char)`（不含前导限定词，与提取器一致）替换为单个空格，
随后按固定顺序应用唯一清理规则（不扫描、不替换为其他规则集）：

1. 水平空白折叠：连续空格/制表符 → 单空格（换行保持原样）；
2. 标点前空格修复：` ,`→`,`、` .`→`.`、` ;`→`;`、` :`→`:`、` !`→`!`、` ?`→`?`；
3. 删除遗留重复逗号：`,\s*,`→`,`（迭代至不动点）。

清理规则只影响可读性；CHAIR 指标对空白不敏感，覆盖损失全部来自区间删除本身。

**算子 B（子句删除）**：删除包含任一 oracle 触发 occurrence 的完整 spaCy 句子。
实现固定为：给每个 query 构造 oracle 分数（触发 root 为 0.0，其余为 1.0），调用冻结
`qwen_sear.filter_caption`（阈值 0.1，含"全删时保留最低分句"fallback）。

**算子 C（VERA 式重写）**：对包含任一 oracle 触发 occurrence 的句子，用冻结
`qwen_vera.REVISION_PROMPT`（trigger phrases 按 root 去重、取每组首个 occurrence 的
phrase）生成 base-revision 候选：Qwen2.5-VL-7B-Instruct 原生 greedy、
`repetition_penalty=1.05`、`max_new_tokens=128`、不安装 VHR（VHR-only 已证明无增益）。
候选接受规则（oracle 版六条，全过才接受，否则逐字 fallback）：

1. 冻结提取器解析出且只解析出一个非空句子；
2. 全部 protected roots 按 root lemma 精确保留；
   `protected = {源句全部 root} \ {触发 root}`（完美 verifier 语义）；
3. 不保留任一触发 root；
4. 候选中不存在 `node(c) ∈ H_i` 的 root（不引入 CHAIR 幻觉对象）；
5. word length ∈ 源句 `[0.5, 1.25]`；
6. 无 period≤16、连续≥3 的 exact whitespace cycle（`qwen_vera.exact_cycle`）。

通过后用 `qwen_vera.apply_sentence_revisions` 替换句级 span，未触发句子逐字保留。
每个触发句只修订一次，不递归。

### 1.4 指标

对 vanilla 与 A/B/C 各输出完整 300 图 caption，用冻结评测器计算：CHAIRs、CHAIRi、
Len、hallucinated/generated mentions、unique correct objects（逐图集合求和，与
`analyze_qwen_vera_dev.py` 同口径）、exact cycle 命中数。A/B/C 相对 vanilla 各做
paired bootstrap（20000 次、seed `2026071701`、复用冻结 `bootstrap_delta`）。
物化：逐图输出 JSON、逐触发 occurrence/接受决策记录、oracle 天花板表 JSON。

### 1.5 决策规则（沿用 nextplan.md 第 3 节，数值不变）

以**算子 A**（mention 删除）为准：

- 若 CHAIRs 相对 vanilla 28.33 改善 < 5 点（即 CHAIRs > 23.33）或 unique correct
  < 96%（< 535）：修订路线天花板不足，转方案 B（审计论文），阶段 1–2 取消；
- 若改善 ≥ 8 点（CHAIRs ≤ 20.33）且 unique correct ≥ 98%（≥ 546）：按计划继续；
- 中间区间：继续，但阶段 2 dev 门槛的 CHAIRs 下降量按
  `max(3.0, 0.5 × oracle 改善点数)` 设定。

算子 B/C 只报告粒度阶梯，不参与决策。若算子 C 因 GPU 故障无法完成，决策仅依据
算子 A，并在 MGR-CEILING.md 中如实记录。

## 2. E0.2 二十图人工触发审计

### 2.1 抽样（固定）

- 总体：`qwen_vera_dev_results/tasks.json` 中出现触发句的 187 个 image_id（VERA
  verifier 触发记录，已物化）。
- 抽样：`random.Random(2026090401).sample(sorted(image_ids), 20)`，一次定死。
- 材料：每个触发句列出全部 trigger phrase 及其 verifier 分数、原始句子、图片路径
  （`val2014/COCO_val2014_{id:012d}.jpg`）。预计 25–40 个触发句。

### 2.2 标注协议（本人执行，三分类，逐 phrase）

对每个 trigger phrase 独立判断"图中是否真的不存在该 phrase 指代的实体"：

1. **真幻觉**：图中确无该实体（删掉它会让 caption 更符合图像）；
2. **真实物体误触发**：图中存在该实体，verifier 误报；
3. **非实体词误触发**：phrase 是抽象词/属性/短语边界错误等，不是可指认的物体
   mention（含 Markdown/parser artifact）。

标注时只看图与 phrase，不看 CHAIR 标签（与 CHAIR 独立的人工信号）。

### 2.3 产出与决策

precision = 真幻觉数 / 总标注数；同时报告三类分布。**决策规则**：若
precision < 0.3，阶段 1 的 mention 提取 v2 必须以"模型自列实体 + span 对齐"为主
方案，spaCy noun chunk 降为回退；若 ≥ 0.3，两级方案按阶段 1 预注册的固定组合继续。

## 3. E0.3 grounding 工程 smoke

### 3.1 mention 集合与调用

- 图：与 E0.2 相同的 20 图（同一 seed 抽样）。
- mention：这 20 图 vanilla caption 的全部提取候选（root 去重、合法性过滤后），
  预计约 100–120 个。
- 模型：本地 `models/qwen2.5-vl-7b-instruct`，bf16、eager attention、单 GPU。
- 固定 grounding prompt（一次冻结，不扫描措辞）：

  ```text
  Point to the {phrase} in the image.
  ```

- 解析：从输出中提取 `<box>[[x1,y1,x2,y2]]</box>`（绝对像素坐标，Qwen2.5-VL 原生
  约定）；无 `<box>` 或坐标不可解析（NaN、x2≤x1+1、y2≤y1+1）记为 ungrounded。
  每次调用后 `model.rope_deltas = None`。

### 3.2 测量项

1. 有效 bbox 率、ungrounded 率、越界率（超出图像宽高）；
2. 对 root 的 node word ∈ 该图 CHAIR GT 的 mention：取该图全部同 node word GT
   框（pycocotools 读 `instances_val2014.json`，category 经同一 synonyms 表映射），
   计算最大 IoU，报告 IoU≥0.5 命中率（含 iscrowd 框，如实报告）；
   node word ∉ GT 的 mention 单列为 GT-absent（可能是幻觉，也可能标注缺失）；
3. 单 mention 时延（含图像预处理）；
4. 确定性：全部 mention 重复运行两次，bbox 输出必须逐字符一致；
5. rope_deltas 泄漏检查：grounding 批次前后各跑一次冻结 verifier query
   （同一图同一对象），分数必须完全一致；
6. 人工抽查清单：随机 10 个 mention 的 bbox 叠图导出，供本人目检（L3 练习）。

### 3.3 门槛

E0.3 无淘汰门槛（工程 smoke）：确定性（第 4 项）与 rope 泄漏（第 5 项）失败必须
修复后重跑；其余为报告型，进入 MGR-CEILING.md 与阶段 1 预注册的输入。另检查
`GroundingDINO`/`OWLv2` 本机可得性（可选消融，不进主方法）。

## 4. 交付物与 tag

- `mgr_ceiling_results/`：E0.1 三算子逐图输出、oracle 天花板表、触发/接受记录、
  E0.2 标注材料与（本人完成后的）标注结果、E0.3 grounding 逐 mention 输出与分析；
- `experiment/MGR-CEILING.md`：三表 + 逐项门槛判定 + 决策结论；
- tag：本预注册 `mgr-ceiling-preregistered-20260904`；阶段 0 完成后
  `mgr-ceiling-20260904`。

## 5. 实验日志

- 2026-09-04：自 `qwen-vera-dev-negative-20260718` 创建 `research/qwen-mgr`；
  冻结本预注册。尚未创建任何 E0.1/E0.2/E0.3 代码或输出。
