# SC-VHR：LLaVA-1.5 跨模型验证记录

> 开始日期：2026-07-16  
> 工作分支：`research/scvhr-llava15`  
> 起始检查点：`39f9650`（tag: `scvhr-instructblip-stable-20260716`）  
> 状态：冻结 dev/validation 已完成。CHAIRs/CHAIRi 在两套数据上显著改善，但对象提及覆盖率均低于预注册的 93% 下限；因此总体门槛未通过，停止正式五种子与 LLaVA 参数扫描。

## 1. 目的

检验在 InstructBLIP 上冻结的 Self-Contrast VHR 解码公式是否能跨架构迁移到 LLaVA-1.5，而不是继续针对新模型调参：

$$
\ell^*=(1+\alpha)\ell_{VHR}-\alpha\ell_{base}
$$

LLaVA-1.5 只沿用原论文/项目对该架构的 VHR 层配置；SC-VHR 的对比分支、强度和 cutoff 不变。

## 2. 冻结配置

- 模型：`models/llava-1.5-7b-hf`
- VHR 层：layer 1 + 最后 14 层
- head score：原 magnitude
- head fraction：0.5
- augmentation ratio：2
- VHR filter：启用
- SC-VHR：`alpha=0.5`, `beta=0.1`
- decoding：greedy，`max_new_tokens=512`
- 固定 dev IDs：`experiment/dev_chair_ids.json`（100 图）
- 固定 validation IDs：`experiment/validation_chair_ids.json`（100 图，与 dev 不重叠）
- seed：0（固定 ID 下只用于保持运行协议一致）

除模型对应的 VHR 作用层外，以上 SC-VHR 参数来自 InstructBLIP 阶段，不允许根据 LLaVA validation 结果调整。

## 3. 运行前基线

| 数据集 | VHR CHAIRs ↓ | VHR CHAIRi ↓ | Len | 生成对象提及 | 幻觉提及 |
|---|---:|---:|---:|---:|---:|
| dev 100 | 43.0 | 15.26 | 80.14 | 701 | 107 |
| validation 100 | 34.0 | 11.25 | 82.03 | 711 | 80 |

权威基线文件：

- `llava15_dev_results/llava-1.5-7b-hf/0-chair_0-vhr-llava15-dev-512.quickchair.json`
- `llava15_validation_results/llava-1.5-7b-hf/0-chair_0-vhr-llava15-val-512.quickchair.json`

## 4. 预注册通过门槛

必须同时满足：

1. dev 和 validation 的 CHAIRs 都至少绝对下降 1.0 点；
2. 两个集合的 CHAIRi 均不得实质恶化：允许的最大绝对上升为 0.5 点；
3. 两个集合平均长度均保持在对应 VHR 基线的 ±7% 内；
4. 对象覆盖不得坍塌：generated mentions 不低于对应 VHR 的 93%；
5. 不能出现一个集合改善而另一个集合反转。

停止规则：任何一套固定集合未通过即记为“LLaVA-1.5 架构迁移负结果”，停止该架构的 `alpha/beta/head fraction` 搜索，不以 validation 调参。只有两套均通过，才进入 LLaVA 正式五种子 CHAIR/POPE。

## 5. 执行日志

- 2026-07-16：确认稳定 tag、工作分支和已有结果目录完整；GPU 4–6 空闲。
- 2026-07-16：完成上述冻结配置与通过/停止门槛预注册；此时尚未生成或评测 LLaVA SC-VHR 输出。
- 2026-07-16：LLaVA 一图烟雾路径通过。首次命令同时传入 100 个固定 ID，发现 `CHAIRBench` 在该情况下不会被 `--chair_samples 1` 截断；进程在成功生成并写入 9 条后由人工中断。该目录 `llava15_scvhr_smoke_results/` 明确视为不完整诊断输出，不参与任何指标计算。中断 traceback 为 `KeyboardInterrupt`，不是模型或 SC-VHR cache 错误。
- 2026-07-16：新增 `scripts/run_llava15_scvhr_crossval.py`。脚本在单次模型加载中依次运行固定 dev 和 validation；运行前检查各 100 个 ID 唯一、两集合无重叠，并在目标输出已存在时拒绝覆盖或静默跳过。
- 2026-07-16：冻结运行完成。dev 100 图耗时 822 秒，validation 100 图耗时 870 秒；两者在同一模型加载中顺序执行，进程正常退出。完整性检查确认各 100 行、ID 唯一、顺序与预注册文件完全一致、无空回答。
- 2026-07-16：直接执行 `python eval/quick_chair.py` 会被同目录 `eval.py` 遮蔽包路径；默认 Python 环境又缺少新版 NLTK 的 `punkt_tab`。最终严格复用历史正式脚本的 `/home/nlp/anaconda3/envs/vhr/bin/python -m eval.quick_chair`，不下载新 tokenizer 资源，不改变历史评测栈。

## 6. 冻结结果

| 数据 | 方法 | CHAIRs ↓ | CHAIRi ↓ | Len | 生成对象提及 | 幻觉提及 |
|---|---|---:|---:|---:|---:|---:|
| dev | VHR | 43.0 | 15.26 | 80.14 | 701 | 107 |
| dev | SC-VHR | **35.0** | **12.34** | 78.42 | 632 | **78** |
| validation | VHR | 34.0 | 11.25 | 82.03 | 711 | 80 |
| validation | SC-VHR | **21.0** | **6.60** | 81.10 | 652 | **43** |

逐图配对 bootstrap（20,000 次，seed 20260716）：

| 数据 | CHAIRs 差值 | 95% CI | CHAIRi 差值 | 95% CI | 修复/新增 |
|---|---:|---|---:|---|---:|
| dev | -8.02 | `[-15,-2]` | -2.92 | `[-5.95,-0.11]` | 10 / 2 |
| validation | -12.99 | `[-21,-6]` | -4.66 | `[-6.94,-2.42]` | 15 / 2 |

## 7. 预注册门槛判定

| 门槛 | dev | validation | 判定 |
|---|---:|---:|---|
| CHAIRs 至少下降 1 点 | -8.0 | -13.0 | 通过 |
| CHAIRi 最大恶化不超过 0.5 | -2.92 | -4.66 | 通过 |
| Len 在基线 ±7% | -2.15% | -1.13% | 通过 |
| generated mentions ≥ 基线 93% | **90.16%** | **91.70%** | **失败** |
| 两集合方向一致 | 改善 | 改善 | 通过 |

总判定：**未通过全部预注册门槛**。虽然 LLaVA-1.5 上的幻觉改善比 InstructBLIP dev/validation 更稳定且显著，但两个集合都减少了对象提及，不能排除部分收益来自降低对象覆盖。因此：

1. 不进入 LLaVA 正式五种子 CHAIR/POPE；
2. 不扫描 `alpha/beta/head fraction`，尤其不根据 validation 调参；
3. 将结果记录为“跨架构 hallucination reduction 成立，但 coverage-preserving claim 未成立”的部分正结果，而不是无条件跨模型成功；
4. 下一步只先做现有输出的覆盖、唯一对象数和重复率诊断，以区分真实覆盖下降与重复提及减少。

## 8. 覆盖诊断（事后解释，不改变 gate）

使用 `eval/compare_object_coverage.py` 和与 quick-CHAIR 相同的 CHAIR canonical object mapping，对现有输出重新分解。这里的诊断用于解释失败门槛，不用于事后把本次实验改判为通过。

| 数据 | 指标 | VHR | SC-VHR | 保留比例 | 配对均值差 95% CI（每图） |
|---|---|---:|---:|---:|---|
| dev | 总对象 mentions | 701 | 632 | 90.16% | `[-1.12,-0.27]` |
| dev | 正确对象 mentions | 594 | 554 | 93.27% | `[-0.72,-0.08]` |
| dev | 唯一正确对象 | 235 | 227 | **96.60%** | `[-0.20,0.03]` |
| dev | 幻觉 mentions | 107 | 78 | 72.90% | `[-0.53,-0.07]` |
| dev | 重复 mentions | 392 | 348 | 88.78% | `[-0.80,-0.09]` |
| validation | 总对象 mentions | 711 | 652 | 91.70% | `[-1.04,-0.16]` |
| validation | 正确对象 mentions | 631 | 609 | 96.51% | `[-0.60,0.14]` |
| validation | 唯一正确对象 | 257 | 245 | **95.33%** | `[-0.25,0.00]` |
| validation | 幻觉 mentions | 80 | 43 | 53.75% | `[-0.55,-0.20]` |
| validation | 重复 mentions | 392 | 372 | 94.90% | `[-0.56,0.15]` |

解释：

1. 预注册的 total generated mentions gate 确实失败，且总 mention 下降显著，不能删除这一负结果。
2. 但真实语义覆盖下降远小于总 mention 下降：唯一正确对象仅下降 3.4%/4.7%，dev 的 CI 跨 0，validation 上界为 0。
3. dev 的总下降同时来自更少重复和更少幻觉；validation 主要来自幻觉减少，正确 mention 的差异不显著。
4. 因此“SC-VHR 只靠少说”不符合现有细粒度证据，但“完全不损失覆盖”同样证据不足。

下一方法假设：当前双边 logit 外推既增强 VHR 偏好的 token，也压低 base 偏好的 token；后者可能误伤部分真实对象。可在新的、未使用图像上预注册 Positive-Delta SC-VHR，只保留 $\max(\ell_{VHR}-\ell_{base},0)$ 的增强项，以测试是否能保留 hallucination reduction 同时恢复总/唯一正确对象覆盖。不得在本次 validation 上调其参数。

## 9. 回滚与输出隔离

- 核心稳定恢复命令见 `experiment/ROLLBACK.md`。
- 本阶段使用全新输出目录，不覆盖 `llava15_dev_results/`、`llava15_validation_results/` 或任何 InstructBLIP 正式结果。
- 烟雾测试、dev 和 validation 输出分别隔离；不完整输出不得进入统计。
- 本阶段源代码、运行脚本和记录使用标签 `scvhr-llava15-crossval-20260716` 建立独立检查点；原 `scvhr-instructblip-stable-20260716` 标签保持不变。
