# SRR-SC-VHR：以 base 为安全锚点的不可逆稳定性回退

> 日期：2026-07-17
> 分支：`research/qwen-stability-router`
> 起点：`qwen-scvhr-dev-negative-20260717`
> 当前状态：实现与工程门槛通过；尚未运行 300 图 SRR dev

## 1. 动机

Qwen2.5-VL-7B 的冻结 300 图 dev 显示固定 VHR head amplification 是负迁移：

- vanilla：`23.67/6.46/147.48`；
- VHR：`29.67/8.26/315.60`；
- SC：`27.00/7.93/345.63`；
- exact cycle：`50/187/233`；
- 512-token hit：`29/133/166`。

SC 沿 VHR-base logit residual 外推，能小幅减少 VHR 的对象幻觉，却进一步放大生成
不稳定。继续扫描层数、倍率或 alpha/beta 既违反停止规则，也不能解决“现代 base 本身
比 VHR 更好”的根本事实。

SRR（Stability-Risk Rollback）的核心不是继续挑 VHR token，而是把 vanilla/base
cache 作为安全锚点：SC 只在稳定时使用，一旦出现已定义的退化风险，当前样本永久
回退 base，并停止后续 VHR forward。

## 2. 固定算法

开放描述同时维护 VHR 与 base cache，沿用冻结的 SC `alpha=0.5,beta=0.1`。每步：

1. 计算 SC logits，并应用模型原生 repetition penalty 1.05；
2. 取 SC argmax，解码“已有生成 token + 候选 token”为文本；
3. 若候选会首次形成 case-sensitive whitespace exact cycle（周期 `<=16`、连续
   `>=3` 次），触发不可逆 rollback；
4. 触发步改从 base logits 选择最高分且不会形成同一定义 cycle 的 token；
5. 后续仅更新 base cache，不再运行 VHR 分支；base token 继续使用相同最小 cycle
   safety mask；
6. 未触发时输出标准 SC token，不增加其他 gate。

binary 问题沿用 RR，完整走 VHR，不启用 SRR。方法不读取 benchmark 名称。

cycle 定义来自 RR-SC-VHR 正式质量分析，早于 Qwen 结果冻结，不扫描 period、repeat、
top-k 或回退持续时间。rollback 是永久的，不比较软阈值。

## 3. 工程门槛

1. 在无风险的短序列上 SRR 与 SC 逐 token 一致；
2. 人工构造的 token/text cycle 必须在第三次完成前触发；
3. rollback 后 VHR forward count 不再增加；
4. rollback 后 base cache 继续接收同一共享历史；
5. base 首选 token 若仍成 cycle，必须选择下一最高安全 token；
6. A->B 与 fresh B 输出一致，样本间 rollback 状态清零；
7. binary route 与 VHR 逐 token 一致。

工程门槛未全部通过，不运行 300 图 dev。

## 4. 固定 dev 评测

只在已有 Qwen dev 300 图运行 SRR；vanilla/VHR/SC 原始输出和评测全部复用，不重跑、
不修改。SRR 写入独立 `qwen_srr_dev_results/`，保持 untracked。

为了达到“现代模型上至少不破坏强 base”的论文标准，SRR 必须同时满足：

1. CHAIRs `<=24.67`，即不比 vanilla 退化超过 1 点，并相对 VHR 至少改善 5 点；
2. CHAIRi `<=6.96`，即不比 vanilla 退化超过 0.5 点；
3. total unique correct objects 至少保留 vanilla 的 98%；
4. Len 在 vanilla 的 `[0.90,1.25]` 倍范围；
5. exact-cycle rate 不高于 vanilla +2pp；
6. 512-token hit rate 不高于 vanilla +5pp；
7. 平均时延低于标准 SC，且 rollback 样本实测 VHR forward 少于 base forward；
8. SRR 相对 VHR 的 CHAIRs paired bootstrap 方向为负。

任一门槛失败：停止 SRR，不运行 locked validation，不扫描规则。全部通过后提交固定
tag，才允许在未查看的 Qwen locked validation 500 图运行一次。

## 5. 论文定位

若通过，SRR 的贡献不是“用 no-repeat 美化输出”，而是：

- 证明视觉 head steering 在现代 GQA 主干上可能跨越语言稳定性边界；
- 以持续维护的 base cache 提供可恢复的信任区域；
- 只在可观测的退化事件发生时不可逆回退，同时停止昂贵分支；
- 将幻觉、稳定性、覆盖和实际计算放在同一个安全解码框架内。

仍需后续 AMBER/MMHal/MME/MMBench 和跨模型结果，单一 Qwen CHAIR 通过不代表论文目标
已经完成。

## 6. 实验日志

- 2026-07-17：从 Qwen SC dev negative tag 建立独立分支，冻结算法、工程门槛和
  dev/validation 停止规则；未运行 SRR。
- 2026-07-17：实现 `srrscvhr` 独立 mode。CPU 单元测试确认 period≤16、连续3次的
  cycle 在完成前被识别；base argmax 仍危险时会逐个屏蔽并选择最高安全 token。
- 2026-07-17：无 exact-cycle 风险的 dev 图 497826 上，SRR 不触发且完整等于 SC，
  说明规则不是无条件后处理。
- 2026-07-17：正式 SC cycle 图 220215 上 GPU 工程检查全部通过：前8 token SRR=SC；
  step 143 检测到 period 1、block `name` 的风险并永久 rollback；VHR/base forward
  为 144/185，触发后 VHR 不再增加；屏蔽 1 个仍会成 cycle 的 base 候选；最终无
  exact cycle；同图重复运行逐 token 一致。结果目录 `qwen_srr_smoke_results/`
  保持 untracked。

工程固定 tag：`qwen-srr-engineering-ready-20260717`。下一步只生成 SRR 300 图，
所有 vanilla/VHR/SC baseline 均复用封存结果。
