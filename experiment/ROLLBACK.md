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
