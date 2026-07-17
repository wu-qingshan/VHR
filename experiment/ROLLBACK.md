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

## CARE-SC 检查点

- `care-scvhr-preregistered-20260717`：CARE 公式、门槛、停止规则和全新100/100/500隔离划分；
- `care-scvhr-implementation-ready-20260717`：反事实对齐、稳定性保护、runner、纯函数测试与成功烟雾验证；
- `care-scvhr-dev-eval-ready-20260717`：结果生成前冻结的统一 dev evaluator；
- `care-scvhr-dev-negative-20260717`：100图部分正但门槛失败的完整记录，locked validation 未运行。

所有 `care_*_results/` 目录保持未跟踪。恢复 CARE 负结果检查点而不覆盖当前工作树：

```bash
git switch -c restore/care-dev-negative care-scvhr-dev-negative-20260717
```

SAVER 区域级主动视觉复核的纯设计与工程可行性检查点为 `saver-feasibility-20260717`。该标签不包含 SAVER 实现或效果实验，只固定 switch-trigger、selected-head 区域图、local candidate-pair verification、成本披露和新数据隔离要求。
