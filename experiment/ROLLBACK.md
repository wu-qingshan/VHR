# 研究版本回滚说明

## 当前稳定版本

- 标签：`scvhr-instructblip-stable-20260716`
- 含义：SC-VHR 在 InstructBLIP 上完成开发、独立验证、正式 CHAIR 五种子和 POPE 三分割后的稳定代码。
- 权威结果：`experiment/SC-VHR.md`。
- 结果目录：`scvhr_dev_results/`、`scvhr_validation_results/`、`scvhr_official_results/`。

该检查点只固化源代码、评测脚本、运行脚本、固定数据划分和研究文档；模型、数据和生成结果不进入 Git。

## 推荐恢复方式

不覆盖当前工作树，创建一个新的恢复分支：

```bash
git switch -c restore/scvhr-stable scvhr-instructblip-stable-20260716
```

如果只需要恢复核心实现文件，可从标签取回：

```bash
git restore --source scvhr-instructblip-stable-20260716 -- main.py generation.py vhr.py utils.py
```

执行第二种方式前应先保存当前未提交修改。不要对包含新实验的工作树执行 `git reset --hard`。

## 后续实验约束

1. 新方法必须使用新 method 名和新 output directory，不能覆盖 `scvhr_*_results/`。
2. 新改动先跑静态检查与 1 图冒烟，再进入固定开发集。
3. 未通过预设门槛时，在实验记录中标为负结果；核心实现恢复到本标签或保持默认 SC-VHR 路径不变。
4. 正式测试数据不用于调参。

## LLaVA-1.5 跨模型诊断检查点

- 标签：`scvhr-llava15-crossval-20260716`
- 分支：`research/scvhr-llava15`
- 内容：冻结 LLaVA dev/validation 运行脚本、覆盖诊断脚本和完整实验记录。
- 结果目录保持未跟踪：`llava15_scvhr_dev_results/`、`llava15_scvhr_validation_results/`、`llava15_scvhr_smoke_results/`。

恢复该诊断阶段代码而不覆盖当前工作树：

```bash
git switch -c restore/scvhr-llava15 scvhr-llava15-crossval-20260716
```

## PD-SC-VHR 预注册实现检查点

- 标签：`pdscvhr-preregistered-20260717`
- 分支：`research/pd-scvhr`
- 内容：全新未使用 dev/validation IDs、预注册门槛、Positive-Delta 独立 generation flag、单次加载 runner 和成功烟雾测试记录。
- 不包含生成结果；`pdscvhr_*_results/` 始终保持未跟踪。

恢复命令：

```bash
git switch -c restore/pdscvhr-preregistered pdscvhr-preregistered-20260717
```

PD 新 dev 负结果完成后的检查点使用标签 `pdscvhr-dev-negative-20260717`。该标签保留实现与完整记录，但生成结果仍未跟踪；若继续新方法，应从该标签创建新分支，不修改历史输出。

三套 LLaVA pooled 分析与通用多配对脚本的检查点标签为 `scvhr-llava-pooled-20260717`。

DG-SC-VHR 预注册实现与成功烟雾测试的检查点标签为 `dgscvhr-preregistered-20260717`；生成结果保持未跟踪。

DG dev 负结果与机制结论的检查点标签为 `dgscvhr-dev-negative-20260717`。锁定 validation 未运行。

标准 SC token-switch 诊断实现与一图逐字等价验证的检查点标签为 `scvhr-switch-diagnostic-ready-20260717`。

100 图逐字等价诊断与 VEG 下一方向结论的检查点标签为 `scvhr-switch-diagnostic-complete-20260717`。

VEG 三分支预注册实现与完整 EOS 烟雾测试的检查点标签为 `vegscvhr-preregistered-20260717`。

VEG dev 部分正/门槛失败结果的检查点标签为 `vegscvhr-dev-negative-20260717`。锁定 validation 未运行。

RR-SC-VHR 完整等价验证、物化脚本与正式 CHAIR/POPE 结果记录的检查点标签为 `rrscvhr-official-20260717`。正式结果目录 `rrscvhr_official_results/` 保持未跟踪。

RR-SC-VHR 统一效率、COCO caption quality、重复/循环诊断和可复现分析脚本的检查点标签为 `rrscvhr-efficiency-quality-20260717`。该标签新增：

- `scripts/benchmark_instructblip_rrscvhr.py`：同一次模型加载下的正反顺序效率测量；
- `scripts/summarize_rrscvhr_benchmark.py`：输出等价性、两轮波动、均值和时延比审计；
- `scripts/analyze_caption_quality.py`：distinct、repeated n-gram 和 exact cycle 配对统计；
- `experiment/RR-SC-VHR-EFFICIENCY-QUALITY.md`：完整效率与质量结论，包括重复/循环负结果。

`rrscvhr_benchmark_results/`、`rrscvhr_quality_results/` 和 `rrscvhr_official_results/` 均保持未跟踪，不进入标签。恢复方式：

```bash
git switch -c restore/rrscvhr-efficiency-quality rrscvhr-efficiency-quality-20260717
```

## Qwen2.5-VL 迁移预注册检查点

- 分支：`research/qwen-scvhr`
- 起点：`rrscvhr-efficiency-quality-20260717`
- 内容：独立环境版本、Qwen 特有工程等价性门槛、全新数据隔离方案、dev 停止规则。
- `.venv-qwen/`、模型权重和 `qwen_*_results/` 均被忽略，不进入提交。
- 该阶段没有携带 CIRD/SAVER/VCRP 后续分支代码，也没有运行 validation/test。
- 预注册 tag：`qwen-scvhr-preregistered-20260717`。

恢复时应从该 tag 新建分支，不能升级历史环境：

```bash
git switch -c restore/qwen-scvhr-preregistered qwen-scvhr-preregistered-20260717
```

Qwen 原生 smoke、专用 GQA/mRoPE attention adapter、独立双 cache generation loop、
ratio=1/HF generate/A->B/RR 等价性全部通过后的检查点标签为
`qwen-scvhr-engineering-ready-20260717`。该标签包含模型 manifest，但不包含权重、
`.venv-qwen/`、`qwen_native_smoke_results/` 或 `qwen_vhr_smoke_results/`。

```bash
git switch -c restore/qwen-scvhr-engineering qwen-scvhr-engineering-ready-20260717
```

排除 3,146 个历史/机制/smoke COCO id 后的 Qwen 300/500/500 互斥划分标签为
`qwen-scvhr-dev-split-20260717`。完整 exclusion manifest 与三个 split SHA256 均在
该标签中；该标签尚未运行任何 Qwen 效果样本。

Qwen 固定 CHAIR chunk runner、严格 merger 和完整 512-token 单图 smoke 后的标签为
`qwen-scvhr-dev-runner-ready-20260717`。从该标签运行 dev，不允许再修改层数、倍率、
alpha/beta、repetition penalty 或最大长度。

Qwen 300 图 dev 的完整负结果、coverage/cycle/runtime/bootstrap 分析和停止决定标签为
`qwen-scvhr-dev-negative-20260717`。locked validation/test 未运行，所有
`qwen_dev_results/` 保持 untracked。该标签是后续 base-anchored stability rollback
方法的机制起点，不应从结果目录复制或修改历史输出。

SRR-SC-VHR 的固定 exact-cycle 风险定义、不可逆 base rollback、停止 VHR forward
规则和严格 dev gate 的预注册标签为 `qwen-srr-preregistered-20260717`。该标签不含
SRR 实现或结果。

SRR CPU cycle/safe-token 单元测试、无风险 SRR=SC、真实 cycle rollback、停止 VHR
forward、base cache 连续与重复运行等价性全部通过后的标签为
`qwen-srr-engineering-ready-20260717`。工程结果保持在未跟踪
`qwen_srr_smoke_results/`。

SRR 300图 dev、完整 CHAIR/coverage/cycle/runtime/bootstrap 分析与停止结论标签为
`qwen-srr-dev-negative-20260717`。SRR 消除全部 exact cycle 并改善 SC 的效率/覆盖，
但没有恢复到 vanilla Pareto；locked validation/test 未运行，结果目录保持 untracked。

Qwen 对象 Yes/No logit verifier 的固定 prompt、token ids、全300图 query 构造和可行性
门槛预注册标签为 `qwen-object-evidence-preregistered-20260717`。该标签不含 verifier
实现或结果。

617-query 构造、dog/elephant/giraffe 方向、重复确定性和动态 image-grid 状态清理通过
后的标签为 `qwen-object-verifier-engineering-ready-20260717`。query/smoke 结果保持
在未跟踪 `qwen_object_evidence_results/`。

617-query ROC-AUC/PR-AUC/balanced-accuracy/bootstrap 分析和 verifier feasibility
通过后的标签为 `qwen-object-verifier-feasible-20260717`。该标签允许进入 OER 修订
设计，但不包含任何 caption 修订或 locked validation 结果。

SEAR 的 spaCy 通用 noun-chunk、固定 stoplist、score≤0.1、高置信句子删除和严格dev
门槛预注册标签为 `qwen-sear-preregistered-20260717`。该标签不含 parser模型、实现或
过滤结果。

SEAR extraction工程、5750-query verifier、300图dev负结果与严格停止决定的检查点标签为
`qwen-sear-dev-negative-20260717`。结果目录`qwen_sear_dev_results/`保持untracked；
locked Qwen validation/test从未读取。核心结果为CHAIRs 23.67→17.00，但unique correct
objects仅保留91.53%，因此否决整句删除并转向新的phrase/clause级方法。

VERA-VHR 的vanilla安全锚点、证据触发局部修订、base/VHR双候选、protected-root二次
验证、全新数据隔离和严格dev停止规则的预注册标签为
`qwen-vera-preregistered-20260717`。该标签不含新split、实现或VERA结果。

排除4446个历史/保留ID后的VERA 300/500/500互斥划分标签为
`qwen-vera-dev-split-20260717`。该标签包含split与完整manifest；三个新集合均尚未运行，
并与原Qwen dev/validation/test完全无重叠。

旧dev工程smoke发现Markdown `*`被spaCy误识别为noun root后、在任何新VERA dev运行前，
加入“root与phrase必须含ASCII英文字母”的通用输入合法性规则。修订预注册标签为
`qwen-vera-preregistered-v2-20260717`；v1 tag保留且不得用于效果主张。

VERA核心策略、任务/双候选/二次query/严格物化脚本、vanilla cache回归修复、HF/ratio=1/
顺序/重复等价性和10-task接受/fallback完整链路通过后的标签为
`qwen-vera-engineering-ready-20260717`。所有smoke结果保持untracked；新VERA dev与两个
locked集合尚未运行。

VERA固定dev分析器与baseline/source-query/revision/candidate-query/materialization完整
执行接口冻结后的标签为`qwen-vera-dev-runner-ready-20260717`。该标签固定所有效果、
coverage、CI、效率和接受率门槛；不含完整dev结果。

VERA新300图dev、完整门槛分析、事后trigger/CHAIR对齐诊断和严格停止决定的检查点标签为
`qwen-vera-dev-negative-20260718`。完整VERA保持vanilla的全部unique correct objects，
但CHAIRs仅28.33→28.00，paired 95% CI `[-1.67,0.67]`，未过预注册效果门槛。
locked validation/test从未运行；`qwen_vera_dev_results/`继续保持untracked。该标签可用于
复现代码、split、manifest、诊断逻辑和实验结论，但不包含生成结果。后续方法必须使用
新的开发集研究mention级视觉蕴含或区域grounding，不得回看当前dev扫描VERA参数。
