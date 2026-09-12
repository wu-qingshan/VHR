# AUDIT-PLAN：方案 B 转向决策与跨主干审计预注册

> 日期：2026-09-11
> 分支：`research/qwen-mgr`
> 触发：tag `mgr-verifier-negative-20260911`（nextplan 第 11 节停止规则分叉）
> 决策：**本人选定方案 B——审计论文**（nextplan 1.2 备选叙事，RESEARCH_PLAN 第 10
> 节预留定位）；阶段 2（MGR 完整方法）取消，预算转入跨模型对照与写作。
> 状态：LLaVA-1.6 实验预注册冻结，尚未实现 runner 或生成任何输出。

## 1. 论文叙事（三条主张）

1. **主张一（跨主干审计）**：解码期全局视觉 head 干预（VHR/SC-VHR 家族）的收益
   随主干现代化系统性消退。现有冻结证据：InstructBLIP 强正迁移（官方五种子）→
   LLaVA-1.5 弱正迁移（CHAIR 改善但覆盖门槛失败）→ **LLaVA-1.6（AnyRes 中间代，
   本文件预注册补测）** → Qwen2.5-VL 负迁移（架构级退化：长度爆炸、cycle、
   CHAIRs 恶化）。
2. **主张二（修订管线三层解剖）**：post-hoc mention 级修订的天花板与两个依次
   暴露的瓶颈。天花板：E0.1 oracle（完美触发下 mention 删除 −17.0 CHAIRs、
   unique correct 100% 保留）证明任务可解。瓶颈一（触发错位）：E0.2 人工审计
   noun-chunk 触发 precision 0.158。瓶颈二（证据自洽）：阶段 1 证明实体级提取
   把触发精度修复到 0.714，但同模型 Yes/No verifier 与 grounding 区域验证继承
   生成轨迹的幻觉信念（幻觉 mention 中位分 0.818、86/189>0.9、AUC ≤0.78），
   召回上限远离门槛。
3. **主张三（评测纪律）**：全程预注册、互斥划分、locked 集不读、冻结分析器、
   tag+SHA256 的可复现负结果协议（附录发布）。

## 2. 已冻结材料索引（tag）

| 材料 | tag |
|---|---|
| InstructBLIP VHR/SC-VHR 官方结果 | `scvhr-instructblip-stable-20260716` 等 |
| LLaVA-1.5 SC-VHR 跨模型验证 | `scvhr-llava15-crossval-20260716`、`scvhr-llava-pooled-20260717` |
| RR-SC-VHR 官方与效率-质量 | `rrscvhr-official-20260717`、`rrscvhr-efficiency-quality-20260717` |
| Qwen SC-VHR 负迁移链 | `qwen-scvhr-*`（20260717 系列） |
| Qwen object verifier 信号分析 | `bac6834`（AUC 0.926） |
| SEAR/VERA 粒度阶梯负结果 | `qwen-sear-dev-negative-20260717`、`qwen-vera-dev-negative-20260718` |
| MGR 阶段 0 天花板 + E0.2 人工审计 + E0.3 | `mgr-ceiling-20260905` |
| MGR 阶段 1 负结果（证据自洽机制） | `mgr-verifier-negative-20260911` |

## 3. 新增实验预注册：LLaVA-1.6 跨主干对照（报告型，无淘汰门槛）

### 3.1 目的

在"老架构（Q-Former/线性投影）→ 中间代（AnyRes）→ 现代架构（GQA/mRoPE）"的
主干谱系上补测中间点，检验 head 干预收益的消退是否在 AnyRes 代已经发生。审计
定位：任何结果都进论文，不构成淘汰。

### 3.2 冻结配置（逐字继承 LLaVA-1.5 记录，SC-VHR 公式，零新调参）

- 模型：`models/llava-v1.6-vicuna-7b-hf`（本地权重，LlavaNext 加载路径）
- VHR：layer 1 + 最后 14 层、magnitude top 50%、aug ratio 2.0、filter 启用、
  `tau=1.0`、`rho_max=2.0`、contrast_mode=remove、directional_delay=64
- SC-VHR：`vhr_logit_alpha=0.5`、`vhr_logit_beta=0.1`（其余开关全关，仅纯 SC）
- 解码：greedy、`max_new_tokens=512`、`repetition_penalty=1.0`、
  `no_repeat_ngram_size=0`、seed 0
- 数据：`experiment/dev_chair_ids.json`（100）与 `experiment/validation_chair_ids.json`
  （100），互斥已验证；与其他主干使用同一图集，跨主干直接可比
- 方法标签与输出：`llava16-vanilla-512`、`vhr-l1h14`、
  `scvhr-llava16-audit-a05-b01` → `llava16_audit_{dev,validation}_results/`
- 评测：`/home/nlp/anaconda3/envs/vhr/bin/python -m eval.quick_chair`（历史正式栈）
- 统计：相对 vanilla 的逐图 paired bootstrap（20,000 次，seed `20260716`，
  与 LLaVA-1.5 记录一致）

### 3.3 预注册解释规则（先于数据冻结）

对每方法相对 vanilla 的 ΔCHAIRs bootstrap：

- **正迁移**：95% CI 上界 ≤ 0 且点估计改善 ≥ 1.0 点；
- **负迁移**：95% CI 下界 ≥ 0 且点估计恶化 ≥ 1.0 点；
- **无显著/混合**：CI 跨 0。

同时强制报告 CHAIRi、Len、生成/幻觉 mention 数（覆盖代价）、exact cycle，以及
Qwen 式退化检查（长度比 >1.5 或 cycle 激增即标注"生成退化"）。

### 3.4 工程要求

1. 单次模型加载内顺序完成 dev→validation×全部方法，或按方法分进程；
2. 拒绝覆盖既有输出；ID 唯一性与两集合互斥运行前断言；
3. 确定性 spot check：任取 2 图每方法重复两次输出逐字一致；
4. 不修改任何历史结果文件与 `main.py/generation.py/vhr.py`；
5. 全部输出附 SHA256 manifest。

## 4. 局限声明（进论文 limitation）

离线环境无 InternVL2.5-8B / LLaVA-OneVision-7B 权重，现代（GQA/mRoPE）主干仅
Qwen2.5-VL 一支；LLaVA-1.6 作为 AnyRes 中间代补位。若网络恢复可在论文修订期
扩展第二现代主干。

## 5. 时间线

- 2026-09-11：本预注册冻结；实现 runner + 工程 spot check；
- 2026-09-12 起：三个方法 × dev+validation 运行与评测（预计单卡 <3 小时）；
- 之后：论文骨架、图表（nextplan 9.3 清单裁剪为审计版）、EMNLP 2027（2027-05）。

## 6. 实验日志

- 2026-09-11：方案 B 决策记录；LLaVA-1.6 对照预注册冻结。此时未编写任何
  llava16 代码，未生成任何输出。盘点确认 `models/llava-v1.6-vicuna-7b-hf`
  本地可用、`utils.py` 已支持 LlavaNext 加载、`main.py` 已支持 scvhr 方法分发。
- 2026-09-11（LLaVA-1.6 对照完成与判定）：
  - 工程：`scripts/run_llava16_audit.py`（配置逐字继承 LLaVA-1.5 记录）；
    三方法确定性 smoke 全过（每方法 2 图 × 2 次经生产路径逐字一致）；
    dev+validation 各 100 行、ID 唯一、与预注册文件一致；单方法单次加载内
    顺序完成两集合（vanilla 741s、VHR 720s、SC 1423s/200 图）。
  - 结果（quickchair，历史正式栈；bootstrap seed 20260716，20000 次）：

    | 集合 | 方法 | CHAIRs | CHAIRi | Len | 提及(幻觉) | ΔCHAIRs vs vanilla | 95% CI | P(<0) |
    |---|---|---:|---:|---:|---|---:|---|---:|
    | dev | vanilla | 35.0 | 10.68 | 157.4 | 618(66) | — | — | — |
    | dev | VHR | 22.0 | 6.59 | 155.3 | 622(41) | **−12.99** | [−23.0, −3.0] | 0.993 |
    | dev | SC-VHR | 29.0 | 7.84 | 149.5 | 612(48) | −6.03 | [−15.0, +3.0] | 0.883 |
    | validation | vanilla | 37.0 | 7.51 | 157.3 | 679(51) | — | — | — |
    | validation | VHR | 30.0 | 8.82 | 152.0 | 692(61) | −6.96 | [−17.0, +4.0] | 0.885 |
    | validation | SC-VHR | 34.0 | 9.47 | 153.1 | 697(66) | −2.98 | [−12.0, +6.0] | 0.707 |

  - 按第 3.3 节预注册解释规则：**VHR = 部分正迁移**（dev CI 上界 −3.0 ≤ 0
    且改善 ≥1 → 正；validation CI 跨 0 → 无显著；validation CHAIRi +1.30
    反向）；**SC-VHR = 无显著/混合**（两集合 CI 均跨 0，点估计均劣于同集 VHR）。
    退化检查：长度比 0.97–0.99、提及数保持、exact cycle 全部 0——无 Qwen 式
    生成退化。
  - 跨主干谱系定稿：InstructBLIP（SC 强正，官方五种子）→ LLaVA-1.5（SC 正向
    但覆盖坍塌）→ **LLaVA-1.6（VHR 部分正、SC 增量消失）** → Qwen2.5-VL（全
    家族负迁移 + 长度/cycle 退化）。主张一的叙事从"悬崖"细化为"梯度"：
    SC 对比增量最先消失（AnyRes 代），整个 head 干预家族在 GQA/mRoPE 代崩塌。
  - 输出 SHA256（`llava16_audit_{dev,validation}_results/llava-v1.6-vicuna-7b-hf/`）：
    dev vanilla `d1ff5f27…6fd03c`、dev VHR `90193314…bf0db2`、
    dev SC `ddc3e20c…335e9e0`、val vanilla `1776ed0e…b639e7`、
    val VHR `ef3c5aef…1def9732`、val SC `1c0d8e1a…e23ffdd`。
