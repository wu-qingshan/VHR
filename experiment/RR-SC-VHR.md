# Response-Routed SC-VHR 实验记录

> 开始日期：2026-07-17  
> 分支：`research/rr-scvhr`  
> 起始检查点：`8e30bbc`（tag: `vegscvhr-dev-negative-20260717`）  
> 状态：实现、同进程严格等价验证和完整 CHAIR/POPE 物化评测均已完成。RR 保留标准 SC 的 CHAIR 收益并精确恢复 VHR 的 POPE 与二元任务成本。

## 1. 已知问题

标准 SC-VHR 在 InstructBLIP 正式 CHAIR 500×5 上显著改善，但 POPE 三分割平均 F1 从 85.47 降至 84.34：precision +0.72、recall -2.69、yes rate -1.92。它将开放描述中的语言先验抑制转化成了二元存在性判断中的保守偏置。

连续 logit gate、PD、DG、VEG 均未在 LLaVA 上可靠超过标准 SC。当前最确定、且直接针对已知负结果的优化是按响应形式路由，而不是继续共享一组决策边界。

## 2. 方法

Response-Routed SC-VHR（RR-SC-VHR）：

- 开放描述/非二元问题：启用冻结标准 SC-VHR；
- 二元问题：完全关闭 `vhr_base_contrast`，使用原 VHR 单分支 logits，不应用 SC plausibility cutoff，也不建立第二套 base KV cache。

二元识别使用现有 `is_binary_question`，支持 Is/Are/Was/Were/Do/Does/Did/Can/Could/Has/Have/Will/Would 前缀，不引入学习器或数据集名称判断。

## 3. 数据形式审计

- POPE popular/adversarial/random 各 3000，共 9000 题；全部以 `Is` 开头，识别覆盖率 **100%**，无漏识别。
- CHAIR prompt 固定为 `Please describe this image in detail.`，不会被识别为 binary。

因此理论路由结果是：RR 的 CHAIR 与标准 SC 逐 token 相同；RR 的 POPE 与原 VHR 逐 token 相同。

## 4. 预注册验证要求

1. 实现必须在每题生成前显式设置 `vhr_base_contrast`，避免上一题状态泄漏；
2. 结果 JSON 写入 `response_route`，值只能为 `scvhr` 或 `vhr`；
3. 开放描述烟雾样本：RR caption、confidence 必须与冻结标准 SC 逐字一致；
4. POPE 烟雾样本：RR caption、confidence 必须与原 VHR 逐字一致；
5. 统计三分割全部 9000 问题的 classifier coverage 必须保持 100%；
6. 路由不改变 head selection、ratio、layer、filter 或解码参数；
7. 若任何等价性失败，停止，不使用“近似恢复”表述。

由于 POPE 路由后的生成参数与 VHR 完全相同，完整 POPE 指标应复用/恢复原 VHR：Acc 85.91、Precision 88.86、Recall 82.47、F1 85.47、YesRate 46.55。正式记录必须同时保留纯 SC 的 F1 84.34 负结果，不能用 RR 隐藏它。

## 5. 效率预期

- 开放描述：与标准 SC 相同，约 2× decoder forward；
- 二元问题：只运行 VHR 分支，恢复原 VHR 时延和 KV cache 占用；
- 与在所有任务强制 SC 相比，POPE 推理计算约减半。

## 6. 输出与回滚

- 烟雾输出：`rrscvhr_smoke_results/`
- 正式输出如需生成：`rrscvhr_official_results/`
- 不覆盖 `results/`、`scvhr_official_results/` 或任何 LLaVA 输出。
- 回滚到 `vegscvhr-dev-negative-20260717` 可完全移除 RR 实现。

## 7. 执行日志

- 2026-07-17：审计 POPE 三分割 9000 问题，全部以 `Is` 开头；binary classifier 覆盖 100%。完成方法、等价性、披露和停止规则预注册。
- 2026-07-17：实现逐题显式 `vhr_base_contrast` 路由和 `response_route` 记录；新增仅用于隔离烟雾输出的 `dataset_limit` 及单次加载 runner `scripts/run_instructblip_rrscvhr_smoke.py`。
- 2026-07-17：首次 RR 烟雾的 CHAIR 1/1 与历史标准 SC 文本/confidence 完全一致；POPE 30/30 文本与历史 VHR 一致，但历史 `ans_conf` 存在约 0–0.5 的浮点差异。按预注册要求不直接接受，runner 改为同一次当前模型加载中依次运行 fresh SC/VHR 与 RR，进行严格同进程等价验证。
- 2026-07-17：同进程严格验证通过：CHAIR RR vs fresh SC 为 1/1 完全一致；POPE RR vs fresh VHR 为 30/30 完全一致，比较字段包括 question ID、caption 和 confidence，route 标签全正确。
- 2026-07-17：新增 `scripts/materialize_rrscvhr_results.py`。基于上述函数等价性，从权威 SC-CHAIR 与 VHR-POPE 原始输出物化完整 RR 文件，只增加 `response_route`，保留 source/output manifest，并拒绝覆盖。
- 2026-07-17：物化脚本首次直接执行因缺少项目根 `sys.path` 导入失败，未创建输出；已补与其他 scripts runner 相同的根路径初始化。
- 2026-07-17：修复后完整物化成功：8 个文件、11,500 行；manifest 逐文件记录权威 source。逐行审计确认除新增 `response_route` 外与源文件完全一致。

## 8. 完整结果

### CHAIR：与标准 SC 完全一致

| seed | CHAIRs ↓ | CHAIRi ↓ | Len |
|---:|---:|---:|---:|
| 0 | 30.0 | 8.28 | 110.30 |
| 1 | 30.8 | 7.50 | 114.09 |
| 2 | 28.8 | 10.07 | 112.87 |
| 3 | 27.8 | 7.21 | 111.92 |
| 4 | 34.6 | 9.05 | 114.67 |
| mean | **30.40** | **8.42** | 112.77 |

相对原 VHR 37.84/9.72/106.42，RR 继承标准 SC 的全部显著性、对象提及和长度结论：2500 图 CHAIRs 差值 -7.44，95% CI `[-9.48,-5.40]`，McNemar `p=1.10e-12`。

### POPE：与原 VHR 完全一致

| split | Acc | Precision | Recall | F1 | YesRate |
|---|---:|---:|---:|---:|---:|
| adversarial | 82.97 | 83.30 | 82.47 | 82.88 | 49.50 |
| random | 89.33 | 95.60 | 82.47 | 88.55 | 43.13 |
| popular | 85.43 | 87.67 | 82.47 | 84.99 | 47.03 |
| mean | **85.91** | **88.86** | **82.47** | **85.47** | **46.55** |

对照纯 SC：85.14/89.58/79.78/84.34/44.63。RR 将 recall 恢复 +2.69、F1 恢复 +1.13、yes rate 恢复 +1.92，同时保留 VHR precision。纯 SC 的 POPE 回归仍必须在论文消融中完整披露。

## 9. 等价性与来源证据

1. 问句覆盖：POPE 9000/9000 被识别为 binary；CHAIR 0/固定提示被识别为 binary。
2. 同进程严格 smoke：CHAIR RR vs fresh SC 1/1；POPE RR vs fresh VHR 30/30，ID/text/confidence 全部一致。
3. 完整 RR 结果由 `scripts/materialize_rrscvhr_results.py` 从权威分支输出确定性物化；8 文件 11,500 行逐字段审计通过。
4. 来源 manifest：`rrscvhr_official_results/instructblip-vicuna-7b/materialization_manifest.json`。

## 10. 效率

- 纯 SC 官方 POPE 四块 750 题分别耗时 683/675/712/726 秒，约 0.90–0.97 秒/题；
- 同进程 VHR/RR POPE 小样本均约 4 秒/10题/分割，即约 0.4–0.5 秒/题；
- RR 在 binary 路由下不创建 base contrast 分支，理论 forward 和 KV cache 从 2 套恢复为 1 套。实测符合约减半趋势，但正式论文应在相同 GPU、完整集上补统一 latency/peak-memory profiling。

## 11. 最终评价

RR-SC-VHR 是目前最可靠的实用主方案：

- 开放描述：使用有显著正式证据的标准 SC；
- 二元存在性：使用无回归的原 VHR；
- 无新增连续参数，无验证集调参，路由规则与响应形式直接对应；
- 消除了“同一强抑制策略跨任务通用”的错误假设。

局限：响应形式路由本身较简单，论文创新核心仍应是 SC-VHR 的 intervention-isolated self-contrast；RR 应作为 calibration/decision policy 组件，而不是单独包装为主要算法创新。
