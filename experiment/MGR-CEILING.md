# MGR-CEILING：阶段 0 天花板测量结果与决策

> 日期：2026-09-05
> 分支：`research/qwen-mgr`
> 预注册：`experiment/MGR-CEILING-PREREG.md`（tag `mgr-ceiling-preregistered-20260904`）
> 数据纪律：全部基于已打开 VERA dev 300 图；CHAIR GT 只用于 explanation-only
> oracle 审计（`posthoc_explanation_only=true, must_not_tune_current_dev=true`）；
> locked validation/test 未产生任何输出。

## 0. 执行摘要

**决策：E0.1 命中"改善 ≥ 8 点且覆盖 ≥ 98%"分支，MGR 主线（方案 A）按计划继续。**
阶段 2 dev 门槛的 CHAIRs 下降量按预注册公式校准为
`max(3.0, 0.5 × 17.0) = 8.5 点`（在新 dev 的 vanilla 基线上相对计算）。

三个附带的结构性发现：

1. **删除算子是唯一天花板充分的修订算子**：oracle mention 删除把 CHAIRs 从
   28.33 降到 11.33 且 unique correct 全保留；oracle 触发下的 VERA 式受约束重写
   仅降到 26.00——80 个重写候选中 54 个保留了被显式标记的幻觉 root。
2. **root 级 noun-chunk 提取的幻觉召回上限只有 68.6%**（CHAIR 幻觉 node word
   能被某 chunk root 覆盖的比例），且 483 个 root 去重候选中 450 个（93.2%）
   不是 CHAIR 词汇根——触发错位问题在提取层，验证 E0.2/提取 v2 的必要性。
3. **Qwen2.5-VL 原生 grounding 工程可用**：GT-present mention 上 IoU≥0.5
   命中 18/21（85.7%）、中位 IoU 0.917；确定性与 `rope_deltas` 泄漏检查通过；
   且观察到对幻觉 mention 的显式拒绝输出（"There are none."）。

## 1. E0.1 oracle 上界（300 图全量）

### 1.1 主表

| 方法 | CHAIRs | CHAIRi | Len | unique correct | 幻觉 mention | exact cycle 图 |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 28.33 | 8.31 | 153.85 | 557 | 129 | 51 |
| **A：oracle mention 删除** | **11.33** | **2.94** | 153.33 | **557（100%）** | 43 | 51 |
| B：oracle 句删除 | 10.67 | 2.93 | 146.19 | 543（97.5%） | 41 | 49 |
| C：oracle VERA 式重写 | 26.00 | 7.53 | 153.65 | 557（100%） | 116 | 51 |

paired bootstrap（CHAIRs，相对 vanilla，20000 次，seed 2026071701）：

- A：−17.00 点，95% CI `[−21.33, −12.67]`，P(方向为负)=1.000；
- B：−17.67 点，95% CI `[−22.33, −13.33]`，P=1.000；
- C：−2.33 点，95% CI `[−4.33, −0.67]`，P=0.9995。

### 1.2 决策判定（预注册第 1.5 节）

- 算子 A 改善 = 28.33 − 11.33 = **17.0 点 ≥ 8**；
- 算子 A unique correct 保留 = 557/557 = **100% ≥ 98%**；
- → **继续 MGR 计划**；阶段 2 dev 门槛 CHAIRs 下降 ≥ 8.5 点。

### 1.3 机制诊断（explanation-only）

- 85/300 图含 CHAIR 幻觉（幻觉 node word 去重 107 个、mention 129 次）；
- oracle 触发句子 80 句、覆盖 63 图；root 捕获率均值 0.686——即使完美分类器，
  31.4% 的幻觉词无法作为 noun-chunk root 被删除（属性位/从句/复合词等）；
- 算子 C 接受率 13/80（16.25%）：失败主因是 `all_trigger_roots_removed`
  （54/80）与 `no_chair_hallucinated_root`（54/80，部分重叠）——受约束重写
  在被明确告知"不支持"时仍复述原对象；`protected_roots` 保持失败 25/80、
  长度越界 10/80。B 的覆盖损失（−14 unique correct）复现 SEAR 结论。
- 算子 A 删除 90 个 mention：86 幻觉 + 4 个重复正确 mention（unique 无损），
  平均每图修改 0.3 句、句法残留（如 "interior of the"）已知且已预注册声明。

## 2. E0.2 人工触发审计（状态：材料就绪，待本人标注）

- 冻结抽样 `random.Random(2026090401).sample(sorted(187 触发图), 20)` →
  33 个触发句、38 个触发 phrase；
- 材料：`mgr_ceiling_results/e02_annotation_material.md`（含图）+
  `e02_annotation_form.csv`（label 列填 1/2/3）；
- 判定规则（预注册第 2.3 节）：人工 precision < 0.3 → 阶段 1 提取 v2 以
  "模型自列实体"为主方案。**本项为阶段 1 预注册前必须完成的输入。**

## 3. E0.3 grounding smoke（20 图 483 mention，已完成）

| 测量项 | 结果 |
|---|---|
| 有效 bbox 率 | 271/483 = 56.1% |
| ungrounded 构成 | point_only 102、no_box_output 110 |
| 越界框 | 80 个，越界幅度全部 ≤ 4px（smart-resize 舍入，良性） |
| GT-present（CHAIR 词根 ∈ GT）| 30 个 mention；其中 21 个有框 |
| **IoU≥0.5 命中** | **18/21 = 85.7%**，中位 IoU 0.917 |
| IoU<0.5 失败例 | 人群/多人（0.38、0.42）与小物体（banana 0.38） |
| 单 mention 时延 | 均值 1.41s、中位 1.40s、最大 2.24s |
| 确定性（双 GPU 全量重跑） | 483/483 raw_text 与 token_ids 逐一致，通过 |
| `rope_deltas` 泄漏 | 批次前后冻结 verifier 分数完全一致（0.5927），通过 |
| bucket 分布 | gt_present 30、gt_absent 3、non_chair_root 450（93.2%） |

- 解析契约修正（v2，已记入预注册日志）：冻结 prompt 下模型输出
  `"bbox_2d": [x1,y1,x2,y2]` JSON（绝对像素）而非 `<box>` 标签；首轮仅
  `<box>` 解析的分析留档 `e03_analysis_v1_boxparser.json`。
- 负证据样本：image 220732 的幻觉 mention "train"（E0.1 oracle 判幻觉）得到
  文本拒绝 "There are none." 而非框——原生 grounding 可提供显式负信号；
  另有 1 例 GT-absent mention 仍给出框（标注缺失或误定位，待 E0.2 人工对照）。
- 人工抽查：10 张 bbox 叠图已导出至 `mgr_ceiling_results/e03_overlays/`
  （seed 2026090402，清单见 `e03_analysis.json` 的 `human_overlay_index`）。
- GroundingDINO/OWLv2 可得性未检查（可选消融，不阻塞，移交阶段 1 前补查）。

## 4. 对阶段 1 的校准输入（预注册时冻结，不回扫）

1. 阶段 2 dev 门槛：CHAIRs 相对 vanilla 下降 ≥ **8.5 点**（0.5×17.0）；
2. 提取 v2 必须解决 31.4% 的 root 捕获缺口与 93.2% 非 CHAIR 词根占比
   （候选方案已在 nextplan 2.1：模型自列实体 + span 对齐 + spaCy 回退）；
3. 修订算子菜单以删除为主算子；受约束重写保留为次选（其 oracle 天花板仅
   −2.33 点，且 67.5% 的候选删不掉触发 root）；
4. grounding 区域条件验证可行（85.7% IoU 命中、1.4s/mention、含负证据形态），
   支撑 nextplan 4.2 的区域 verifier 设计；
5. 全 caption 验证成本预算：20 图 483 mention ≈ 11.3 分钟单 GPU，约
   1.4s/mention——阶段 1 门槛 5（≤8s/图）可满足（平均 24 mention/图需
   批量共享前缀或只验证实体清单，进入阶段 1 预注册细化）。

## 5. 工件与 SHA256

输入（未改动，校验通过）：

- `experiment/qwen_vera_dev_chair_ids.json`
  `fa4234988eccb3588ee0b3e27e9005f6ce4507a6e2c16dea55fc380696a63ddf`
- `qwen_vera_dev_results/vanilla.json`
  `699aead712eacdeb83227a518abbce75a5df6a324ca1504bca7d69bd20224932`
- `qwen_vera_dev_results/source/extractions.json`
  `7c24978dddfaa4db2f1ee1faea3507bfdf654fc467ba350aef1427df67efb75a`
- `eval/synonyms.txt` `b685537b36fe21b791c87151b96a991ceb582edff19f3da204dcc7a7905502f2`

输出（`mgr_ceiling_results/`）：

| 文件 | SHA256 |
|---|---|
| oracle_triggers.json | `d6103b28…f44ca7` |
| operatorA.jsonl | `26f7431…620485` |
| operatorB.jsonl | `1624701…4a3133d` |
| operatorC_tasks.json | `08979f1…20fe4d3` |
| rewrite_candidates.jsonl | `3a35b5f…672680f` |
| rewrite_candidates_repeat.jsonl | `bdf5975…5f089713` |
| operatorC.jsonl | `79df7f6…4202424` |
| operatorC_audits.json | `76468e5…8a0537c0` |
| ceiling_analysis.json | `2ba1ea4…634760b` |
| ceiling_analysis_final.json | `79e844f…99aa75bf` |
| e02_sampling.json | `99ae42a…7d752d77` |
| e02_annotation_material.md | `b75c387…2d109f7` |
| e02_annotation_form.csv | `d8f6279…ac8d88df` |
| e03_grounding.jsonl | `8a32c43…d7ccdb3ca` |
| e03_grounding_repeat.jsonl | `c7b08a6…599af13f` |
| e03_analysis.json | `c60e3b0…0fea52b` |
| e03_analysis_v1_boxparser.json | `c508fa7…e873b6` |

离线确定性：oracle 构建（A/B/tasks/triggers）双次运行 4 文件哈希一致；
GPU 确定性：算子 C 与 grounding 各双 GPU 全量重跑 token 级一致。

## 6. 审计抽查指引（L6，10 分钟）

1. 打开 `e02_annotation_material.md` 抽 3 个触发句，对照图片预填标注，再与
   E0.1 oracle 触发集合交叉看是否一致；
2. 从 `operatorC_audits.json` 抽 3 个 `passed=true` 记录，逐字验证
   protected roots 确实在候选中保留、触发 root 确实消失；
3. 打开 `e03_overlays/` 任选 3 张，目检 bbox 是否套住 phrase 所指；
4. 用 `eval/quick_chair.py` 对 `operatorA.jsonl` 重算一次 CHAIRs，应得 11.33；
5. 重跑 `build_mgr_ceiling_oracle.py` 前先移走输出目录，比对哈希与本表。

## 7. 结论

E0.1/E0.3 完成且全部工程门槛通过；E0.2 待本人标注（38 phrase，预计 1.5–2
小时）。阶段 0 的决策为：**不转方案 B，按 nextplan 阶段 1（mention 级验证器
feasibility，预注册先行）继续**。E0.2 标注结果与（可选）GroundingDINO 可得性
检查是进入阶段 1 预注册前仅剩的两个输入。
