# VCRP-VHR：视觉一致残差投影

> 开始日期：2026-07-17  
> 分支：`research/vpb-vhr`  
> 起始检查点：`7e1434c`（tag: `rrscvhr-efficiency-quality-20260717`）  
> 状态：机制假设与诊断gate预注册；尚未实现干预或运行效果实验

## 1. 来自既有实验的新机制约束

当前证据排除了三类简单解释：

1. SC-VHR在InstructBLIP正式2500图和LLaVA互斥300图上显著降低幻觉，证明VHR干预相对base的logit方向包含有效信息；
2. PD-SC删除负向delta后完全失去收益，说明关键不是单纯增强视觉token，而是抑制与VHR干预相反的语言偏好；
3. SAVER在100图、8779步上的grounding诊断显示，VHR-selected head QK map对真实对象token的pointing仅10.0%，与all-head相同且显著低于中心先验33.85%。因此selected heads不能被当作空间检测器。

更合理的解释是：VHR-selected heads提供**全局、图像条件化的隐藏空间残差方向**；后续FFN写入的语言先验有时沿相反方向覆盖该视觉证据。下一方法不再解释attention peak，而直接约束attention残差与FFN残差的内部竞争。

## 2. 方法定义

暂定名称：**Visual-Consistent Residual Projection VHR（VCRP-VHR）**。

在VHR目标层 $l$、生成步骤 $t$，令：

- $a_{l,t}$：仅保留VHR-selected heads后，经该层 $W^O$ 投影到hidden space的VHR干预残差；当原倍率为2时，它等于VHR相对base attention输出额外写入的残差；
- $m_{l,t}$：同层MLP/FFN写入residual stream的输出。

FFN中与视觉残差反向的投影为：

$$
m^-_{l,t}=\frac{\min(\langle m_{l,t},a_{l,t}\rangle,0)}
{\lVert a_{l,t}\rVert_2^2+\epsilon}a_{l,t}.
$$

VCRP使用：

$$
m'_{l,t}=m_{l,t}-m^-_{l,t}.
$$

这不是任意steering公式，而是以下半空间投影问题的闭式解：

$$
m' = \arg\min_z \lVert z-m\rVert_2^2
\quad\text{s.t.}\quad \langle z,a\rangle\ge0.
$$

因此它具有三个固定性质：

1. 当FFN与视觉残差不冲突时完全不改动；
2. 冲突时只删除最小必要的反向分量，不缩放整个FFN；
3. 没有需要扫描的alpha、margin或layer threshold。

首版只在VHR目标层的autoregressive decode步骤应用，不改text/clean prefill，不改变VHR选头。binary任务在纯方法验证后再决定是否保留RR回退，不能预先隐藏负结果。

## 3. 与近期方法的边界

- **VHR**放大视觉敏感heads，但不建模FFN随后是否覆盖视觉残差；VCRP直接约束两条残差通路的冲突。
- **SC-VHR**需要VHR/base双分支并在词表logits上线性外推；VCRP是单分支hidden-space最小投影，目标是减少约2倍解码成本和logit外推引起的循环。
- **FADE**在选定层用任务相关强度衰减整个FFN；VCRP不衰减与视觉方向正交或一致的FFN成分，并且逐图、逐层、逐token自适应。
- **CAS/activation steering**依赖外部冲突样本得到固定方向；VCRP方向来自当前图像的VHR-selected residual，不需要校准样本或训练。
- **BRACS/ADAPT**分别使用attention barrier或空间attention anchor；VCRP不依赖attention map具备空间grounding能力。

## 4. 第一阶段只做机制诊断

在任何干预实现前，新增独立flag `vhr_vcrp_diagnostic`，只记录、不修改hidden states或logits。默认路径必须逐字不变。

每个生成token记录目标层聚合统计：

- `conflict_layer_fraction`：$\langle m,a\rangle<0$的层比例；
- `opposing_ffn_fraction`：$\sum_l\lVert m^-\rVert / \sum_l\lVert m\rVert$；
- `visual_ffn_norm_ratio`：$\sum_l\lVert a\rVert / \sum_l\lVert m\rVert$；
- `projection_hidden_fraction`：建议修正量相对layer hidden norm的比例；
- 生成token ID及step index。

诊断集固定为已使用的 `experiment/pdscvhr_dev_chair_ids.json` 100图。该集合只用于机制开发，不能支持最终效果主张；`pdscvhr_validation_chair_ids.json`保持封存，不运行。

使用CHAIR synonym/node-word规则将生成对象mention对齐到token步骤，并分为真实对象与幻觉对象。诊断不得计算或比较任何VCRP效果，因为此阶段不存在干预。

## 5. 机制gate与停止规则

预先固定主信号为 `opposing_ffn_fraction`，其他统计只作解释。进入效果实现必须同时满足：

1. 至少得到50个幻觉对象mention和100个真实对象mention的有效token对齐；
2. 幻觉对象的 `opposing_ffn_fraction` 中位数高于真实对象；
3. 分层bootstrap的中位数差95% CI下界大于0；
4. 以该信号区分幻觉/真实对象的ROC-AUC至少0.65；
5. 按生成位置四分位分组，至少3组保持同方向，排除纯长度混杂；
6. 开启诊断的VHR与同进程标准VHR在ID、text、confidence上100/100一致。

任何一项失败：记录负结果，停止VCRP效果实现，不扫描统计组合、层区间或连续阈值。

## 6. 若机制gate通过后的效果协议

只有上述gate全部通过，才允许：

1. 用单独flag实现闭式投影，公式固定为完整删除反向分量；
2. 从所有历史结果和ID文件中排除已用图像，创建全新的dev/locked-validation/reserved-test；
3. dev只比较VHR、SC/RR和VCRP，不扫描投影倍率；
4. 成功门槛同时覆盖CHAIRs/CHAIRi、unique correct、POPE、重复/cycle、时延和显存；
5. LLaVA通过后移植Qwen2.5-VL-7B，再进入AMBER、MMHal/HallusionBench及MME/MMBench。

VCRP的论文价值不只取决于最低CHAIR：它必须证明单分支、低成本、覆盖和稳定性Pareto优于SC/RR，并在现代backbone上成立。

## 7. 隔离与回滚

- 新代码仅使用 `vhr_vcrp_*` 命名；
- 机制输出目录固定为 `vcrp_diagnostic_results/`；
- 后续效果目录另建，不能覆盖任何VHR/SC/RR/CARE/SAVER结果；
- 所有结果目录保持untracked；
- 恢复起点：`rrscvhr-efficiency-quality-20260717`。

## 8. 执行日志

- 2026-07-17：建立独立分支并固定方法公式、机制gate、停止规则和PD dev只读诊断协议；尚未修改模型。
- 2026-07-17：新增独立 `vcrp.py`，实现FFN到视觉残差半空间的闭式投影、decoder-layer诊断forward和逐token统计聚合；`vhr.py`仅在 `vhr_vcrp_*` flag开启时额外计算VHR intervention residual。
- 2026-07-17：3项纯函数测试通过：对齐FFN不变、反向分量被最小删除、零视觉残差安全。默认flag关闭。
- 2026-07-17：PD dev首图同一模型加载依次运行标准VHR与无干预诊断VHR，question ID、text、confidence完全一致。诊断共83个token，每步15个目标层，全部统计finite、zero-visual layer fraction为0。
- 2026-07-17：首图分析器完成token decode、Treebank word span、CHAIR synonym/node-word和diagnostic step对齐，无对齐失败。该图只有1个CHAIR对象mention，不能用于gate判定；完整100图尚未运行。
