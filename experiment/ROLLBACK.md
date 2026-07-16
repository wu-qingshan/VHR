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
