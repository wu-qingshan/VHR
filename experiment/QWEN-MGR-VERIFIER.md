# QWEN-MGR-VERIFIER：mention 级组合证据验证器 feasibility（阶段 1 预注册）

> 日期：2026-09-05
> 分支：`research/qwen-mgr`
> 起点：`mgr-ceiling-20260905`（阶段 0 决策：MGR 主线继续）
> 状态：预注册冻结；尚未编写方法代码、未运行任何 feasibility 数据
> 上游：`nextplan.md` 第 4 节；`experiment/MGR-CEILING.md` 第 4 节校准输入

## 1. 目的

VERA 的失败点是触发与真实幻觉错位（336 触发句仅 11.6% 含 CHAIR 幻觉）。阶段 0
证明：(i) 完美触发下 mention 删除天花板为 CHAIRs −17.0 点且覆盖无损；(ii) 错位来自
提取层——noun-chunk root 93.2% 不是 CHAIR 词汇、幻觉召回上限 68.6%；(iii) 原生
grounding 可用（IoU≥0.5 命中 85.7%，1.4 s/mention）。本阶段回答一个问题：

> 用"实体级提取 v2 + 整图 Yes/No verifier + grounding 条件区域 verifier"的固定组合
> 证据，能否在已打开的 600 图上把触发对齐 precision 提到 ≥0.5、recall ≥0.6、
> mention 级 ROC-AUC ≥0.85，且成本可接受？

任一门槛失败即停止，不扫描 prompt、阈值、margin、集成规则。

## 2. 数据与纪律

- 图像与 caption（全部已打开，不重新生成）：
  - 旧 Qwen dev 300：`experiment/qwen_dev_chair_ids.json`
    （`e723f34b…77f8c4`），vanilla `qwen_dev_results/vanilla.json`
    （`f51a8f86…64cf7b`，CHAIRs 23.67 / 106 幻觉 mention）；
  - VERA dev 300：`experiment/qwen_vera_dev_chair_ids.json`
    （`fa423498…63ddf`），vanilla `qwen_vera_dev_results/vanilla.json`
    （`699aead7…24932`，CHAIRs 28.33 / 129 幻觉 mention）。
- 可复用物化资产（同一冻结 verifier，prompt `Is there a visible {object} in this
  image? Answer only Yes or No.`，token 9454/2753，`sigmoid(yes−no)`）：
  - `qwen_sear_dev_results/extractions.json`（`ba673a2e…5f5b`）与
    `verifier.json`（`0a864b76…e50c`，5750 query）；
  - `qwen_vera_dev_results/source/extractions.json`（`7c24978d…fb75a`）与
    `source/verifier.json`（`2d94b28d…0d5b3`，5964 query）。
- 工程 smoke 只用 COCO 42、136 与上述已打开图；locked validation/test 与未来的
  `mgr_{dev,validation,test}` 不产生任何输出。
- **CHAIR 词表与 GT 只允许出现在分析器**（参考标签与门槛计算）。提取、verifier、
  grounding、裁剪、集成规则的代码路径禁止导入 `datasets`/`eval.chair`/synonyms。
- 新输出全部写入 `mgr_verifier_results/`；不修改任何历史结果文件。
- GroundingDINO/OWLv2：transformers 原生 zero-shot 检测类可导入，但本机无权重
  缓存且无外网；判定为**当前不可用**，仅保留为未来可选消融，不进入本阶段。

## 3. 固定方法（一次定死）

### 3.1 mention 提取 v2

**(a) 模型自列实体（text-only）**。用 Qwen2.5-VL-7B-Instruct 以纯文本（**不输入
图像**，避免模型按可见性过滤掉正是要抓的幻觉实体）处理 vanilla caption，HF
`generate` greedy、原生 `repetition_penalty=1.05`、`max_new_tokens=160`，固定 prompt：

```text
The following text is a description of an image. List every concrete physical thing the text mentions as being in the image: objects, people, animals, vehicles, food, furniture, plants, and similar visible entities.
Rules:
- Copy each item exactly as it is written in the text (same words, same spelling), without articles such as "a", "an", or "the".
- One item per line, no numbering, no bullets, no explanations.
- Do not list abstract words (such as atmosphere, mood, scene, setting, design, detail), colors, materials, actions, or whole places such as room or street.
- Do not add, merge, or paraphrase items. Do not list anything the text does not mention.

Text:
{caption}
```

解析与 span 对齐（固定规则）：逐行取项；去掉行首 `-`/`*`/`•`/编号与行尾句点；
丢弃空行、超过 8 个词的行、以及第 40 项以后的项（各自计数）；去掉前导限定词
（`a an the this that these those`）。对齐：在 caption 中做大小写不敏感的**整词
子串匹配**，取全部出现位置为 occurrence；若无匹配，允许**仅末词**的复数容错
（项末词 +`s`/+`es`，或项末词去尾 `s`）；仍无匹配 → `unaligned`，丢弃并计数。
mention 字符串取 caption 中匹配 span 的原文；同一 caption 内按小写去重。

**(b) spaCy noun chunk（冻结 3.8.11 + en_core_web_sm 3.8.0）**：直接复用两套
dev 的物化 `extractions.json`（经 `qwen_vera.sanitize_extraction`）。

**E0.2 条件分支（由人工 precision 机械决定，与本阶段任何结果无关）**：

- 若 E0.2 noun-chunk 人工 precision < 0.3 → **分支 S**：inventory = 自列对齐
  mention；某 caption 自列对齐为空时，该 caption 回退为 spaCy 候选；
- 若 ≥ 0.3 → **分支 U**：inventory = 自列对齐 mention ∪ 与任一自列 span 无字符
  重叠的 spaCy 候选。

预注册消融行（无论分支，均报告，不参与门槛）：`selflist_only`、`spacy_only`
（VERA 等价 inventory）、`union`。feasibility 计算在 union 超集上进行，各行为其子集，
mention 分数不依赖 inventory 归属。

### 3.2 整图 verifier `s_whole`

对每个 (image, mention 字符串) 运行冻结 verifier（`qwen_object_verifier.score_object`
不改动），object = mention 字符串。若 (image_id, 小写 object) 与物化 SEAR/VERA
query 的 `object` 精确相同，直接复用物化分数（报告复用率）；否则新建 query。

### 3.3 grounding 与区域 verifier `s_region`

- grounding prompt（E0.3 冻结）：`Point to the {phrase} in the image.`，greedy、
  `max_new_tokens=64`，每次调用后 `model.rope_deltas=None`；
- 解析（E0.3 v2）：首个 `<box>[[x1,y1,x2,y2]]</box>` 或首个 `"bbox_2d": [x1,y1,x2,y2]`；
  仅点输出（`point_2d`/`<points`）、文本拒绝或无法解析 → `ungrounded`，原因分别记为
  `point_only` / `no_box_output`；
- 裁剪：框各边按框宽（x 方向）/框高（y 方向）的 10% 外扩，裁到图像边界；任一边
  < 2 px → `ungrounded`（`degenerate_box`）；
- 区域 verifier：对裁剪图运行**同一**冻结 verifier、同一 object 字符串，得 `s_region`。
- feasibility 对 inventory 中**全部** mention 运行 grounding 与区域 verifier（不用
  GT 选算，便于诊断）；方法成本按 3.4 的级联口径从逐调用时延重算。

### 3.4 固定集成规则："双证人缺席"

```
s = max(s_whole, s_region)   若存在有效框
s = s_whole                  若 ungrounded
触发 ⇔ s ≤ 0.1
```

选择依据（一次写死）：区域证据只能**否决**整图触发、不能制造触发；整图说"在"就不
触发；整图说"不在"而 grounding 能框出且区域 verifier 说"在"，视为整图 verifier
漏检小物体，撤销触发。这是 coverage-first 的最保守组合。阈值 0.1 沿用 SEAR/VERA
冻结值，不扫描。由此，方法只需对 `s_whole ≤ 0.1` 的 mention 发起 grounding 与区域
query（**级联等价**：`s_whole > 0.1` 时 `s ≥ s_whole > 0.1` 必不触发），成本核算
按此口径。消融分数 `s_whole` 单独报告（等价 VERA 触发器换新 inventory）。

## 4. 参考标签与指标（仅分析器）

对每个 mention，用冻结 CHAIR 评测器对 mention 字符串运行 `caption_to_words` 得
node words：

- 任一 node word ∉ 该图 GT → **幻觉 mention**（正类）；
- 全部 node word ∈ GT 且 ≥1 个 → **正确 mention**（负类）；
- 无 CHAIR node word → **unlabeled**（不进 P/R/AUC，单独统计）。

指标（各 inventory 行均报告；门槛只看 E0.2 选定分支）：

- `precision_chair` = 触发∧幻觉 / 触发∧有标签；`precision_all` = 触发∧幻觉 / 全部
  触发（unlabeled 视为非幻觉，悲观下界）；unlabeled 触发占比；
- `recall` = 触发∧幻觉 / 全部幻觉 mention；
- `H_coverage`：全 caption 层面 CHAIR 幻觉 node word 集合中被 ≥1 个触发 mention 覆盖
  的比例（衔接 E0.1 的 68.6% 提取上限）；
- ROC-AUC 与 PR-AUC：有标签 mention 上以 `s` 排序（低分=幻觉）；
- 分数分布、ungrounded 率与原因、grounding 对幻觉 mention 的行为（框/点/拒绝）、
  错误类型（小物体、词义歧义、群体名词）；
- 成本：每图 `s_whole` query 时延 + 级联区域 query（grounding+region）时延之和；
  另报告全管线（自列生成+全部调用）每图时延与各组件分解。

## 5. 预注册门槛（进入实现前冻结；数值来自 nextplan 4.1）

1. 有标签 mention ≥ 1500，且幻觉 mention ≥ 200；
2. `precision_chair` ≥ 0.5；
3. `recall` ≥ 0.6；
4. ROC-AUC ≥ 0.85；
5. verifier query 成本（整图 + 级联区域，含 grounding 生成）≤ 8 秒/图；全管线时延
   另报告，受阶段 2 门槛 6（≤ SC-VHR 26.54 秒/图）约束；
6. 工程门槛：
   - E1 自列生成确定性：同 caption 两次 token 序列一致；
   - E2 verifier 确定性：整图与区域重复分数 |Δ| ≤ 1e-6；
   - E3 grounding 确定性：raw 输出逐字一致；
   - E4 顺序无关：同一 verifier query 在 grounding 调用之前/之后分数一致；
   - E5 状态清理：每次调用后 `rope_deltas=None`，批前/批后固定 verifier 分数一致；
   - E6 span 完整性：每个 occurrence 的 caption 切片与 mention 字符串（含复数容错
     记录）一致；裁剪框在图像边界内；
   - E7 输入哈希与第 2 节一致，不修改历史文件；
   - E8 manifest：全部 query/输出带 SHA256，可追溯到 mention。

任一门槛失败：停止，不扫描 prompt/阈值/margin/规则，转方案 B 或回阶段 0（需新决策
文档）。工程门槛失败只修实现，不看效果数据。

分析器在缺少 E0.2 完整标注（38 个 phrase 全部填写）时**拒绝输出门槛判定**，只输出
工程统计；分支选择日志必须记录 E0.2 precision 数值与所选分支。

## 6. 实现与运行顺序

1. `qwen_mgr.py`：自列 prompt、解析/对齐、裁剪、集成规则、grounding 解析（无 CHAIR）；
   `scripts/test_qwen_mgr.py` 单元测试；
2. `scripts/run_mgr_selflist.py`（GPU，text-only 生成 + 对齐，输出 inventory）；
3. `scripts/build_mgr_inventory.py`（合并自列与 spaCy，生成 union query 清单，标记
   可复用整图分数）；
4. `scripts/run_mgr_verifier.py`（整图新 query）与 `scripts/run_mgr_grounding.py`
   （grounding + 裁剪 + 区域 verifier，逐 mention 记录时延）；
5. `scripts/run_mgr_engineering_smoke.py`（E1–E6）→ tag `mgr-verifier-engineering-ready`；
6. 600 图 feasibility 运行（4 GPU 分块）；
7. `scripts/analyze_mgr_verifier.py`（唯一导入 CHAIR 的脚本；读 E0.2 CSV 决定分支；
   输出全部 inventory 行与门槛判定）→ tag `mgr-verifier-feasible-YYYYMMDD` 或
   `mgr-verifier-negative-YYYYMMDD`。

## 7. 实验日志

- 2026-09-05：冻结本预注册。此时未编写 `qwen_mgr.py` 或任何 runner，未运行任何
  自列/grounding/区域 query。GroundingDINO/OWLv2 可得性检查完成（不可用，离线）。
- 2026-09-06（裁剪下界工程修正，v2）：首个区域 query 即触发
  `smart_resize` 硬约束：输入图像两边均须 ≥ 28 px（factor=28），预注册第 3.3 节
  的"任一边 < 2 px → ungrounded"在 2–28 px 区间物理上无法送入 processor。修正：
  `MIN_CROP_SIDE` 由 2 提为 28，原因仍记 `degenerate_box`；margin 外扩与 clamp
  规则、其余全部定义不变。该修正在看到任何效果数据之前、由工程约束强制发生。
  整图 verifier（1749 新 query）已在此修正前完成，不受影响。
- 2026-09-06（阶段 1 feasibility 数据完成，正式判定待 E0.2）：
  - 工程：selflist 600 图完成（对齐 4295 mention、未对齐率 34.3%、31 个空 caption）；
    union inventory 11,986 query（整图分数复用 10,237、新跑 1,749）；
    grounding+区域 verifier 全量 11,986/11,986 完成，chunk1 双进程产生 2,756 条
    重复行，全部仅时延字段不同、结果字段逐字一致（确定性成立）；
    verifier 泄漏检查通过；级联口径成本 6.17 秒/图（≤8 达标），加自列生成约
    8.1 秒/图。
  - 首轮指标（分析器在无 E0.2 标注时按预注册抑制门槛判定）：

    | inventory | P_chair | recall | AUC(s) | AUC(s_whole) | 幻觉 mention | H_cov |
    |---|---:|---:|---:|---:|---:|---:|
    | branch_S | 0.714 | 0.103 | 0.776 | 0.771 | 97 | 0.047 |
    | branch_U | 0.565 | 0.185 | 0.773 | 0.767 | 189 | 0.166 |
    | selflist_only | 0.714 | 0.112 | 0.781 | 0.774 | 89 | 0.047 |
    | spacy_only | 0.507 | 0.188 | 0.750 | 0.750 | 186 | 0.171 |

  - 门槛预测：G2（P≥0.5）与 G5（成本）通过；G1（幻觉 mention 189<200，且
    branch_S 仅 97）、G3（recall 0.185≪0.6）、G4（AUC 0.773<0.85）失败。
    两分支结论一致，**判定结果不依赖 E0.2 的分支选择**；正式 negative tag 在
    E0.2 标注完成、分析器输出判定后冻结。
  - 失败机制（`mgr_verifier_results/score_diagnosis.json`，explanation-only）：
    幻觉 mention 的 `s_whole` 中位数 0.818，86/189 > 0.9（正确 mention 中位
    0.993）；组合分数只把触发内的幻觉捕获从 40 降到 35（region 证据几乎无增量，
    AUC +0.005）。即**同模型 verifier 系统性继承生成轨迹的幻觉信念**，且区域
    verifier 同样如此（幻觉 mention 的 region 分数 0.85–0.99）。object-evidence
    阶段的 AUC 0.926 是固定对象清单上的信号，不迁移到"模型自己幻觉出的对象"。
    触发精度问题（VERA 失败点）确已被提取 v2 修复，但召回瓶颈转移到了证据源
    本身。E0.3 观察到的 "There are none." 拒绝形态是少数例外（point_only 率
    38–49%，多数幻觉 mention 仍得到高置信框或点）。
  - 按第 5 节停止规则：不扫 prompt/阈值/margin/规则；阶段 2 取消，进入
    "转方案 B 或回阶段 0 重新评估（需新决策文档）"的分叉，等待本人对 E0.2 的
    标注与后续方向决策。
- 2026-09-11（E0.2 标注完成，正式判定冻结）：
  - E0.2 人工标注 38/38 完成：真幻觉 6、真实物体误触发 9、非实体词误触发 23。
    **noun-chunk 触发人工 precision = 6/38 = 0.158 < 0.3** → 按预注册分支规则，
    主 inventory 为 branch_S（模型自列实体为主方案）。84.2% 触发为误报
    （60.5% 非实体词 + 23.7% 真实物体），独立人工证据确认 VERA 触发错位诊断。
  - 正式门槛判定（branch_S）：

    | 门槛 | 数值 | 要求 | 判定 |
    |---|---|---|---|
    | G1 样本量 | 有标签 1037 / 幻觉 97 | ≥1500 / ≥200 | **失败** |
    | G2 precision_chair | 0.714 | ≥0.5 | 通过 |
    | G3 recall | 0.103 | ≥0.6 | **失败** |
    | G4 ROC-AUC | 0.776 | ≥0.85 | **失败** |
    | G5 verifier 成本 | 6.17 秒/图 | ≤8 秒/图 | 通过 |

  - **判定：`mgr-verifier-negative`。** 停止，不扫描任何 prompt/阈值/margin/
    集成规则；阶段 2（MGR 完整方法）取消。失败机制维持 2026-09-06 诊断：
    同模型 verifier 对幻觉 mention 的中位打分 0.818（86/189 > 0.9），区域证据
    无增量；提取 v2 已修复触发精度（G2 通过、P_chair 0.714 对比 VERA 期的
    11.6% 句级命中），但证据源继承生成轨迹幻觉信念，召回与排序均不可达门槛。
  - 关键工件 SHA256：

    | 文件 | SHA256 |
    |---|---|
    | analysis.json（正式判定） | `97884512…2bec2` |
    | e02_annotation_form.csv（人工标注） | `c0862092…f1836` |
    | inventory.json | `8c6e643c…f3c012` |
    | whole_queries.json | `75348f58…53df4b8` |
    | whole-verifier-chunk00.jsonl | `34345c97…8c8cd10` |
    | region-chunk00.jsonl | `4c8334a7…c81e9e8` |
    | region-chunk01.jsonl | `243fd21c…796b2a8` |
    | region-chunk02.jsonl | `4c4635b9…0688c5c` |
    | region-chunk03.jsonl | `ec3b0795…dfdb076b` |
    | selflist-old.jsonl | `26f43645…b53618ed` |
    | selflist-vera.jsonl | `e00adf81…6e3aff2` |
    | score_diagnosis.json | `3fa2bc57…2fbbcff8` |
    | engineering_smoke.json | `01bb064e…b2515fab` |

## 8. 阶段结论

MGR 的"同模型组合证据"前提在 mention 级不成立：触发对齐问题已由实体级提取
解决（G2 通过），但 Yes/No logit verifier 与 grounding 区域验证都会继承生成
轨迹对幻觉对象的置信，任何同模型证据组合的召回上限（实测 recall 0.10–0.19、
AUC ≤0.78）远离门槛。阶段 1 按预注册判负封存；后续走 nextplan 停止规则分叉
（方案 B 审计论文，或新决策文档评估跨模型/外部证据源），由本人决策。
