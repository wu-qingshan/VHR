# Qwen2.5-VL 上的 VHR / SC-VHR 现代主干迁移

> 日期：2026-07-17
> 分支：`research/qwen-scvhr`
> 起点：`rrscvhr-efficiency-quality-20260717`
> 当前状态：E0/E1/E2 工程门槛全部通过；尚未创建效果划分、尚未运行效果数据

## 1. 目的与研究问题

当前最可靠的方法是 RR-SC-VHR：InstructBLIP 正式 2500 图上将
CHAIRs/CHAIRi 从 `37.84/9.72` 降至 `30.40/8.42`，同时通过响应形式路由
将 POPE 精确恢复到 VHR。LLaVA-1.5 三套互斥划分 pooled 300 图上也观察到
`38.67/12.66 -> 28.00/9.25`，但还缺少现代 MLLM 上的正式证据。

本阶段只回答以下问题：

1. VHR 的 head reweighting 能否正确迁移到具有 GQA、3D multimodal RoPE 和动态
   image token 数量的 Qwen2.5-VL？
2. SC-VHR 的完整高维 logit residual 在现代主干上是否仍能稳定减少对象幻觉？
3. 若信号存在，其代价是否仍表现为覆盖下降、二元任务保守偏置、重复和约 2 倍时延？

本阶段不继续扫描 InstructBLIP/LLaVA 参数，不修改历史结果，不把旧 validation/test
用于 Qwen 方法选择。

## 2. 隔离与可回滚规则

- Qwen 分支必须从稳定标签 `rrscvhr-efficiency-quality-20260717` 创建，不携带
  CIRD、SAVER、VCRP 等后续研究分支代码。
- 使用独立 `.venv-qwen/`；禁止升级历史环境中的 `transformers==4.45.x`。
- 模型权重放在已忽略的 `models/`；输出只写入 `qwen_*_results/`，保持 untracked。
- 只提交源码、固定配置、数据划分、manifest、分析脚本和本记录。
- validation/test 绝不调参；任何阶段未过预注册门槛立即停止该精确版本。

## 3. 固定软件与模型

- 模型：`Qwen/Qwen2.5-VL-7B-Instruct`
- Transformers：`4.51.3`
- qwen-vl-utils：`0.0.14`
- Torch：`2.5.1+cu121`
- dtype：模型原生 `bfloat16`（A6000 支持；避免 Qwen eager attention 对 fp16 inf 的
  额外修补路径）
- attention implementation：首轮使用 `eager` 完成数值审计；通过后才允许比较
  `sdpa`，不得因效果指标选择实现。
- generation：greedy，`do_sample=False`，`num_beams=1`，首轮 smoke
  `max_new_tokens=64`；保留模型原生 `repetition_penalty=1.05`，CHAIR 固定为项目
  现有详细描述上限。

权重下载后必须补记 Hugging Face revision/commit、文件 manifest 和总大小。

## 4. Qwen 特有工程风险

Qwen2.5-VL 不能直接复用当前 LLaMA attention patch：

- `num_key_value_heads < num_attention_heads`，必须在 GQA `repeat_kv` 后按 query
  head 进行 VHR 选择和缩放；
- 使用 multimodal 3D RoPE，不能自行重建普通 1D position id；
- image token 区间由 `image_grid_thw / spatial_merge_size` 动态决定，禁止硬编码
  InstructBLIP 的 32 token 或 LLaVA 的固定区间；
- generation cache 含 mRoPE 状态，SC 双分支不能共享被污染的 cache；
- 每个样本结束后必须清除 selected heads、prefill summary、route 和诊断状态。

优先实现独立 Qwen runner/generation loop，不修改历史 `generation.py` 的默认行为。

## 5. 阶段 E0：原生模型与接口审计

在真实 COCO 图像上完成以下 smoke，尚不启用 VHR：

1. 官方 chat template + `process_vision_info` 可生成非空、可终止文本；
2. 记录 `input_ids`、`attention_mask`、`pixel_values`、`image_grid_thw` 的 shape；
3. 由 token id 和 grid 独立计算 image token 区间，两者必须一致；
4. 同一输入连续两次 greedy 输出逐 token 相同；
5. 两张不同尺寸图像的 image token 数允许不同且均与 grid 公式一致。

任一 shape、cache、mRoPE 或终止条件错误时，只修工程问题，不运行 CHAIR。

## 6. 阶段 E1：VHR attention 等价性与状态审计

实现 Qwen 专用 attention 后必须满足：

1. patch 默认关闭时，原生与 patched 模型首步 logits 的最大绝对误差
   `<= 1e-4`，greedy 文本逐 token 一致；
2. `aug_ratio=1` 时与关闭 patch 的文本逐 token 一致；
3. text-only/clean prefill 各执行一次，所有选中 head id 均在合法范围；
4. decode 步只复用当前样本选头，不重新混入上一样本状态；
5. A->B 和 fresh B 的 logits/文本一致；
6. 所有 logits、attention output 和 cache 均 finite。

未全部通过，不创建效果划分。

## 7. 阶段 E2：SC 与 RR 工程审计

- SC 分别运行 VHR 分支和 vanilla/base 分支，使用冻结的
  `alpha=0.5, beta=0.1`；不扫描参数。
- RR 仅按通用响应形式路由：开放描述走 SC，明确二元问句走 VHR；不得读取
  benchmark 名称。
- 开放描述 RR 必须与 SC 逐 token 一致；二元 RR 必须与 VHR 逐 token 一致。
- 报告实际 forward 次数、秒/样本和峰值显存，不能只估算成本。

## 8. 新数据隔离方案

工程审计通过后才生成固定划分。候选池为 COCO val2014，先排除：

- 所有已提交历史 ID 文件；
- 所有现有 InstructBLIP/LLaVA dev、validation、official 原始输出中出现的 image id；
- 任何曾用于 SAVER、CARE、CIRD、VCRP 或其他机制选择的图像。

在剩余 ID 上用固定 seed `20260717` 打乱并依次分配：

- Qwen dev：300 图；
- locked validation：500 图；
- locked test：500 图。

脚本必须输出 exclusion manifest、各集合 SHA256，并断言三者互斥。validation/test
在 dev 配置冻结并提交 tag 后才能运行。

## 9. 阶段 D：dev 效果门槛

固定比较 vanilla、VHR、SC-VHR、RR-SC-VHR。VHR 使用原论文思想：视觉反事实与
真实图像 prefill 的 head output 差异选取 top 50%，倍率 2；具体适配层只允许根据
模型结构预先定义，不能依据 dev CHAIR 扫层。

SC 相对同图 VHR 必须同时满足：

1. CHAIRs 至少绝对下降 3.0 点；
2. CHAIRi 不恶化；
3. unique correct objects 下降不超过 2%；
4. 平均长度变化在 `+-5%`；
5. exact cycle hit rate 不增加超过 1 个百分点；
6. paired bootstrap 的 CHAIRs 差值方向为负；
7. POPE 固定小规模诊断中，RR 必须精确恢复 VHR，纯 SC 负结果单独披露。

任一门槛失败：停止该精确迁移，不运行 locked validation，不扫描 alpha、beta、层数、
head fraction 或 route 阈值。此时结论转为“SC 的收益未迁移到现代主干”，下一方法
改为对象证据监督的架构无关干预，而不是继续手工 gate。

## 10. 进入论文主表的最低条件

只有 dev 和 locked validation 均通过后才运行 locked test，并继续扩展：

- 幻觉：CHAIR、AMBER、POPE、MMHal-Bench 或 HallusionBench；
- 一般能力：MME、MMBench；
- 强基线：Greedy、VCD、OPERA、VHR、RR-SC，加至少两个近期 training-free 方法；
- 系统指标：时延、吞吐、峰值显存、对象覆盖、重复和 cycle；
- 统计：逐样本 paired bootstrap/McNemar，同时报告效果量和置信区间。

Qwen 上单一 CHAIR 正结果不足以声称达到论文目标。

## 11. 实验日志

- 2026-07-17：从 `rrscvhr-efficiency-quality-20260717` 创建
  `research/qwen-scvhr`；确认历史结果目录仍为 untracked，未携带 CIRD 分支代码。
- 2026-07-17：冻结独立环境版本、工程门槛、数据隔离方案和停止规则；尚未运行模型。
- 2026-07-17：创建 `.venv-qwen` 时发现当前实际 GPU runtime 为
  `torch 2.5.1+cu121 / torchvision 0.20.1+cu121`，并非仓库旧 requirements 中的
  2.1.2。模型尚未运行，因此将 Qwen 固定版本更正为 2.5.1；Qwen venv 只读复用
  `/home/nlp/.venv` 的 Torch/CUDA、NumPy/Pillow 等基础 runtime，本地优先安装
  Transformers 4.51.3、qwen-vl-utils 0.0.14 及其依赖，不修改原环境。
- 2026-07-17：源码与 config 审计确认模型为 28 层、28 query heads、4 KV heads，
  `spatial_merge_size=2`，原生 dtype 为 bfloat16。冻结 Qwen VHR 层为 layer 1 +
  last 14，与原 VHR 的 LLaVA 类主干策略一致；不在 dev 上扫描层数。
- 2026-07-17：首次 E1 中关闭 patch 的首步 logits 误差为 0，ratio=1、head finite、
  prefill 次数和 A->B 状态清理均通过，但独立 vanilla 循环从第 12 个 token 起与
  HF generate 分歧。诊断确认不是 cache/mRoPE，而是模型 generation config 固定的
  `repetition_penalty=1.05` 尚未在独立循环应用；仅修复这一工程遗漏后重跑 E1，
  不运行效果数据。
- 2026-07-17：完整下载固定 revision，共 16 个文件、`16,595,981,281` bytes；
  SHA256/大小见 `experiment/qwen25_vl_7b_model_manifest.json`。
- 2026-07-17：E0 原生 smoke 通过。两张 COCO 图的动态 image token 分别为
  391/234，均与 `image_grid_thw / spatial_merge_size^2` 一致；同图两次 greedy
  逐 token 相同；两图分别生成 23/29 token 后正常 EOS；峰值 allocated
  15.91/15.66 GiB。原始审计 SHA256 为
  `7a142d1d4d96ff3516add54c24dc884bcc4761efe0f3ab7b316fab8e7ff91851`，结果目录
  `qwen_native_smoke_results/` 保持 untracked。
- 2026-07-17：修复 repetition processor 后 E1/E2 全部通过：关闭 patch 首步
  logits 最大绝对误差 `0.0`；独立 vanilla 与 HF generate 16/16 token 一致；
  ratio=1 与 vanilla 一致；A->B 与 fresh B 一致；15 个目标层均恰好执行一次
  text/clean prefill，selected heads 合法且 head vary 全部 finite；SC 的 VHR/base
  forward 均为 16 次；开放问题 RR=SC、二元问题 RR=VHR。原始审计 SHA256 为
  `f3eae930d073ce1184726b8da3dedbbaedf259703e69360880bd2ab0de39229f`，结果目录
  `qwen_vhr_smoke_results/` 保持 untracked。

工程门槛完成后的固定 tag 为 `qwen-scvhr-engineering-ready-20260717`。下一步只生成
预注册的新互斥 dev/locked validation/test 划分，尚不允许运行 validation/test。
