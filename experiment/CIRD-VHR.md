# CIRD-VHR：因果干预残差蒸馏

> 开始日期：2026-07-17  
> 分支：`research/cid-vhr`  
> 起始检查点：`7e1434c`（tag: `rrscvhr-efficiency-quality-20260717`）  
> 状态：COCO-train 100图teacher-forcing gate已完成并失败；未安装训练依赖、未训练模型、未运行效果实验

## 1. 为什么从手工gate转向学习

已完成的负结果具有一致模式：

- DG、VEG、CARE等手工logit gate不能可靠区分修复与新增幻觉；
- SAVER的selected-head空间定位显著低于中心先验；
- VCRP的FFN冲突主信号AUC只有0.565，bootstrap CI跨0。

但SC-VHR本身在InstructBLIP正式2500图和LLaVA互斥300图上稳定显著改善，说明**完整高维干预方向有效，人工压缩成单个标量后信息丢失**。下一阶段不再继续构造阈值，而把VHR/base的因果logit残差蒸馏到一个单分支student中。

## 2. 为什么不直接做整段回答DPO

在PD dev 100图上，用每图幻觉mention和正确mention覆盖选择VHR/SC优势回答：

- 10图偏好SC，4图偏好VHR，72图幻觉数持平，14图存在覆盖冲突；
- 只有14%形成无明显覆盖损失的偏好对；
- 14对的词级SequenceMatcher相似度均值为0.451，中位数0.475，没有一对达到0.7。

这与switch诊断一致：SC通常先改变冠词、介词、形容词或动词，再改变后续实体轨迹。整段DPO会同时学习大量风格和句法差异，不能干净归因于幻觉修复。ADAPT/Groc-PO等近期工作也已覆盖常规response-level preference optimization，因此本项目不把简单VHR-vs-SC DPO作为主方法。

## 3. 方法候选

暂定名称：**Causal Intervention Residual Distillation（CIRD-VHR）**。

对同一图像、prompt和真实caption前缀 $y_{<t}$，冻结teacher计算：

$$
\ell_B(t),\qquad \ell_V(t),\qquad
\ell_{SC}(t)=\ell_V(t)+\alpha(\ell_V(t)-\ell_B(t)),
$$

并沿用SC的VHR plausibility mask。训练student LoRA，使其单分支分布逼近teacher，同时保留真实token监督：

$$
\mathcal L = \mathcal L_{CE}(y_t)
+\mathcal L_{KD}\bigl(p_{SC}(t),p_{student}(t)\bigr).
$$

binary存在性样本使用RR规则，以VHR而不是SC作为teacher，避免已知POPE recall回归。首版不引入response-level偏好模型，也不训练额外视觉检测器。

核心目标：把双分支SC的高维校正内化为LoRA参数，推理时只需一次普通student forward，从而同时解决SC约2倍时延与部分生成循环问题。

## 4. 与普通KD/SFT的区别

- teacher不是更大模型，而是同一模型在VHR干预与base反事实下得到的因果残差；
- 同一真实prefix消除了自由生成轨迹差异，监督直接作用于“下一token应该如何因图像而变化”；
- CE保证真实caption覆盖，KD只传递VHR/SC的视觉校正分布；
- 必须比较COCO-caption SFT、完整teacher KD、仅VHR KD和CIRD，证明收益不是多看了caption数据。

## 5. 训练前teacher-forcing gate

在实现LoRA前，只验证SC是否是合理的token分布teacher。固定使用COCO train2014的100张图，每图取annotation ID最小的caption；图像ID见 `experiment/cird_teacher_feasibility_train_ids.json`。这些图不属于COCO val上的任何历史dev/validation/test。

对caption token逐步强制teacher forcing，记录：

- base、VHR、SC对真实下一token的NLL；
- CHAIR object token子集的NLL；
- 非对象token NLL；
- VHR-base intervention幅度和SC plausibility保留情况；
- 每图长度与所有token的精确对齐。

预注册进入训练的门槛：

1. 100图全部完成、无token错位、所有logprob finite；
2. 至少200个真实对象token；
3. SC对象token平均NLL显著低于base，图像级paired bootstrap 95% CI上界小于0；
4. SC对象token平均NLL不高于VHR；
5. SC全token平均NLL相对base最多恶化0.02；
6. SC改善对象token的图像比例至少60%；
7. 不根据结果扫描alpha、beta、层数或head fraction。

任一门槛失败：停止CIRD，不安装PEFT/TRL、不生成大规模teacher数据、不运行训练。

## 6. gate通过后的训练路线

只有teacher-forcing gate通过，才执行：

1. 从COCO train构建至少5000张caption训练图和独立500张train-domain validation；
2. 预计算top-k teacher分布与other-mass，结果目录保持untracked；
3. 安装并锁定PEFT版本，LoRA只作用于language model attention/MLP线性层；
4. 训练CIRD、caption-SFT、VHR-KD和SC full-KD消融；
5. 在全新COCO val dev/locked validation上比较Greedy、VHR、SC/RR和student；
6. 通过后移植Qwen2.5-VL并扩展AMBER、POPE、MMHal/HallusionBench、MME/MMBench。

最终主张必须是：student以单分支成本达到或超过SC的幻觉—覆盖—稳定性Pareto，而不是只在COCO caption上记忆训练分布。

## 7. 隔离与回滚

- teacher feasibility只使用 `cird_teacher_*` flags和输出目录；
- 不修改或覆盖VHR/SC/RR历史结果；
- COCO train图与所有COCO val实验天然分离；
- 训练依赖在gate通过前不安装；
- 恢复起点：`rrscvhr-efficiency-quality-20260717`。

## 8. 执行日志与正式判定

- 2026-07-17：官方train2014整包下载速度预计超过80分钟，因此停止整包传输；改用可复现脚本从官方地址只下载固定100张图，共17MB。中止的部分zip已删除。
- 2026-07-17：实现独立forced-token teacher diagnostic。每一步在同一真实caption prefix下记录base/VHR/SC目标NLL、VHR-base target delta和SC plausibility；生成token被强制为固定caption，不产生自由生成效果结果。
- 2026-07-17：首图11个token中有8个被SC `beta=0.1` plausibility cutoff屏蔽，暴露SC自由生成分布可能不适合作为真实caption路径教师；未据此修改beta或取消cutoff。
- 2026-07-17：6 GPU完成固定100图，共1459 token、153个CHAIR object token，无caption/token对齐失败。

正式结果：

- SC不可用目标：507/1459（34.75%）；
- SC不可用对象目标：24/153（15.69%）；
- 对象token数153，低于预注册至少200；
- 仅在SC未屏蔽对象token的68张完整图上，61图SC对象NLL优于base，paired image mean CI为`[-0.234,-0.097]`。

后一个正结果存在明确选择偏差：SC cutoff已经删除大量低概率真实token，剩余token上的归一化NLL不能与完整base分布公平比较。因此不能用它证明SC是合格teacher。

| 门槛 | 结果 | 判定 |
|---|---:|---|
| 100图、无对齐失败 | 100、0失败 | 通过 |
| 所有SC目标logprob finite | 507个不可用 | **失败** |
| 对象token ≥200 | 153 | **失败** |
| SC对象NLL显著优于base | 条件子集CI<0 | 描述性通过但有选择偏差 |
| SC对象NLL不差于VHR | 条件子集通过 | 描述性通过 |
| 全tokenNLL最大回退0.02 | 因507个被屏蔽不可公平判定 | **失败** |
| 改善对象图比例 ≥60% | 条件子集89.7% | 描述性通过 |

**总判定：失败。** 按第5节停止规则，不安装PEFT/TRL，不生成5000图teacher数据，不训练LoRA，不扫描alpha/beta或移除plausibility mask。CIRD源码只保留为teacher分布诊断工具。

连续training-free gate和蒸馏gate的负结果说明，当前最有价值的下一步不是继续发明VHR变体，而是把已显著成立的RR-SC方法迁移到现代backbone并扩展标准benchmark，判断其核心贡献能否支撑跨架构论文。
