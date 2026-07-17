# VERA-VHR：视觉证据路由的局部修订

> 日期：2026-07-17
> 分支：`research/qwen-vera`
> 起点：`qwen-sear-dev-negative-20260717`
> 当前状态：实现与旧开放数据工程smoke通过；新300图dev尚未运行

## 1. 研究动机

现有Qwen证据形成三个互补结论：

1. 原生vanilla为23.67/6.46，而VHR/SC为29.67/8.26与27.00/7.93，说明现代主干上
   全局head amplification会破坏已较强的base轨迹；
2. 固定对象verifier在617 queries上达到ROC-AUC 0.926，说明具体对象存在性具有可靠
   的独立视觉信号；
3. SEAR把CHAIRs显著降到17.00，但整句删除只保留91.53% unique correct objects。
   311个删除句移除28次幻觉mention的同时移除243次真实mention，证明失败来自修订粒度，
   而不是对象证据完全无效。

VERA-VHR（Verified Evidence-Routed Amendment with VHR）不沿VHR轨迹生成整段caption，
而把VHR降为**只在证据触发后提供局部修订候选**。vanilla始终是主轨迹和安全锚点。

## 2. 固定方法

### 2.1 源caption与触发

1. 使用Qwen2.5-VL-7B-Instruct原生vanilla生成完整caption；
2. 使用SEAR已冻结的`spaCy 3.8.11 + en_core_web_sm 3.8.0`、determiner规则和stoplist
   提取noun chunks；
3. 在进入visual query前应用固定输入合法性：root lemma与phrase必须都至少包含一个ASCII
   英文字母；该规则只排除Markdown `*`等非词汇parser artifact，不使用CHAIR或图像标签；
4. 使用完全冻结的Yes/No verifier prompt、token ids和score；
5. 一个句子只在任一候选`score<=0.1`时触发局部修订；同句所有低分phrase一次性列入，
   不逐phrase多轮修改。

### 2.2 固定修订prompt

对触发句使用原图和如下固定模板：

```text
Original sentence: {sentence}
The following phrases are not clearly supported by the image: {phrases}.
Rewrite only this sentence using details clearly visible in the image. Preserve every other
supported object and fact, do not add new objects, and do not mention the unsupported phrases.
Answer with exactly one revised sentence and no explanation.
```

每个触发句生成两个greedy候选：

- `base-revision`：原生Qwen，不安装VHR干预；
- `vhr-revision`：原VHR固定Qwen适配，layer 1 + last 14、top 50%、ratio=2、原filter。

两者均保留模型原生`repetition_penalty=1.05`，`max_new_tokens=128`。不使用SC，不扫描
prompt、层数、ratio、温度或候选数。

### 2.3 修订候选验证与接受

对两个候选再次运行同一通用noun-chunk提取和冻结verifier。原句中`score>=0.5`的root
lemma定义为protected roots。候选必须同时满足：

1. parser得到且只得到一个非空句子；
2. 不含`score<=0.1`的候选对象；
3. 保留全部protected roots，按root lemma精确匹配；
4. 不再包含任一触发phrase的root lemma；
5. word length为原句的`[0.5,1.25]`；
6. 不含period≤16、连续≥3的exact whitespace cycle。

通过候选按固定词典序选择：

1. protected roots保留数（理论上全部通过者相同）；
2. 候选最低object score更高；
3. 与原句word-level Levenshtein edit distance更小；
4. 完全并列时优先base-revision。

若两个候选均失败，逐字保留原句。每个源句最多修订一次，不递归重写、不做整句删除、
不使用CHAIR词表或标注参与触发、生成、接受或排序。

## 3. 数据隔离

从完整COCO val2014重新创建300/500/500 dev/locked validation/locked test，必须排除：

- 所有`experiment/**/*ids.json`，包括现有Qwen dev/validation/test；
- 所有历史与当前`*_results/**/*.json`中出现的COCO ID；
- Qwen smoke IDs 42与136；
- 三个新集合彼此重叠。

固定seed为`2026071701`。输出命名为`qwen_vera_{dev,validation,test}_chair_ids.json`及
`qwen_vera_split_manifest.json`。locked validation/test在方法代码、工程smoke和dev gate
提交tag之前禁止生成caption、query或verifier结果。

## 4. 工程门槛

1. VERA关闭修订时与封存vanilla输出逐字段一致；
2. base-revision使用原生分支，VHR关闭/ratio=1时首步logits与base误差为0；
3. 同一修订query重复两次token序列和score完全一致；
4. base→VHR、VHR→base顺序不改变各自fresh结果；
5. 原caption未触发的句子逐字保留；
6. 接受候选满足全部六项规则，fallback逐字等于原句；
7. 输出非空、确定性，所有query/候选/接受决策有可追溯manifest；
8. 不修改历史`main.py/generation.py/vhr.py`和任何历史结果文件。

任一工程门槛失败时只修复实现问题，不运行效果数据。

## 5. 新dev效果门槛

固定比较vanilla、原VHR、SC-VHR、SEAR-style sentence deletion诊断、base-only revision、
VHR-only revision与完整VERA-VHR。SEAR只作为失败粒度消融，不参与参数选择。

完整VERA-VHR必须同时满足：

1. CHAIRs相对vanilla至少下降3.0点；
2. CHAIRi不高于vanilla；
3. unique correct objects至少保留vanilla的98%；
4. Len为vanilla的`[0.90,1.05]`；
5. exact cycle和512 source hit不高于vanilla；
6. paired bootstrap CHAIRs差值方向为负，95% CI上界不大于0；
7. 总pipeline平均时延低于同图SC-VHR；
8. 触发句中至少20%接受一个修订候选，避免结果等同vanilla；
9. 报告base/VHR候选生成数、各自通过/被选择率、fallback率、protected-root保持、
   新增/移除正确与幻觉对象、修订前后grammar与失败案例。

任一门槛失败：停止，不运行locked validation，不扫描0.1/0.5、prompt、长度、候选排序或
VHR配置。全部通过后冻结实现并提交tag，才允许一次性运行locked validation 500图。

## 6. 论文贡献假设

若成功，VERA-VHR支持三项比单一CHAIR增益更完整的贡献：

1. 系统证明视觉head amplification从旧Q-Former/projector架构迁移到现代GQA/mRoPE
   主干时可能产生负迁移；
2. 将VHR从全局解码干预重构为证据触发的局部proposal，避免破坏base主轨迹；
3. 用显式protected-root与二次视觉验证形成可审计的hallucination/coverage Pareto。

即使Qwen dev通过，论文目标仍需InstructBLIP、LLaVA和至少一个独立现代架构迁移，以及
AMBER/MMHal/HallusionBench、POPE、MME/MMBench和近期强基线。

## 7. 实验日志

- 2026-07-17：从`qwen-sear-dev-negative-20260717`创建独立分支；冻结上述方法、数据
  排除、工程门槛和dev停止规则。此时尚未创建新split、实现或运行VERA。
- 2026-07-17：划分脚本扫描7个历史ID文件和321个`*_results` JSON，排除4446个已使用
  或已保留COCO ID，剩余36058。固定seed `2026071701`创建300/500/500集合，三者互斥，
  且与原Qwen dev/validation/test完全无重叠。尚未生成任何新集合caption。

新split manifest内的canonical JSON digest：

- dev：`fe4df8aaf564aae0dd79e7a4b62885063d5720f971aca5c4440b0d3c6859c595`
- validation：`b8f8b27026e4553b464bde4080823bfa7edb70f045d0a32b4d0f3c209728ef85`
- test：`99bf437b501dbfa3370aa98a388bd95200e60614914b68a929ca59658632a230`

实际split文件SHA256分别为：

- dev：`fa4234988eccb3588ee0b3e27e9005f6ce4507a6e2c16dea55fc380696a63ddf`
- validation：`d8e892cf1036ebfea84c1efb6f264599b3016e1341de018c949ccbd4aa053636`
- test：`bc13b91dde4757d82d7aa63627fef941cebc86eedaa3d49f382ec0c3157d410a`
- 2026-07-17：首次旧dev工程smoke发现spaCy把Markdown标题`**Bed**`中的`*`解析为
  noun root，导致无意义低分触发；此时新VERA dev尚未运行。旧开放数据316个初始任务中
  12个含非ASCII词汇root。加入上述通用input-validity规则并形成v2预注册；不依据CHAIR
  错误、效果指标或locked数据修改，原v1 tag保留审计。
- 2026-07-17：v2首个有效任务上base-revision连续生成128个`The`。同输入HF generate
  正常输出，逐步诊断定位到SRR提交`28e8198`的回归：为了rollback后停止VHR cache，
  `update_vhr`条件遗漏了vanilla mode，令当前独立vanilla循环重复使用prefill logits。
  封存Qwen vanilla dev结果生成于该回归之前，SRR走独立base cache，因此历史效果结论
  不受影响。恢复vanilla primary cache更新后必须重新通过HF逐token等价门槛。
- 2026-07-17：恢复`qwen_scvhr.py`的vanilla primary cache逐token更新。同一修订输入上
  HF generate与独立base循环均生成12个完全相同token；ratio=1 VHR也与HF完全一致。
  base/vhr各自重复运行以及base→VHR、VHR→base顺序均逐token一致。
- 2026-07-17：在旧开放Qwen dev构造v2任务，仅用于工程验证。合法性过滤后为306个触发
  句/172图；固定前10个任务生成20个base/VHR候选、84个二次verifier query。完整VERA
  接受3/10：base 2、VHR 1，其余7个逐字fallback；base-only接受2，VHR-only接受1。
  接受和fallback路径、protected-root、trigger-root、length、cycle和score审计均物化。
  新VERA dev/validation/test仍未运行。
- 2026-07-17：新增冻结dev分析器，固定计算CHAIR、unique coverage、object增删、cycle、
  source max-hit、总pipeline时延、候选routing、grammar和paired bootstrap，并逐项执行
  第5节门槛。分析器提交前没有运行新dev CHAIR；随后才启动六GPU baseline生成。

工程结果SHA256：

- HF/ratio/order smoke：`1cf4e2c7ed285574bf150fa2b16f54250179288c83fc6f43c931c5c9baf59916`
- 10-task revisions：`93af08fd7993e0e6087f778fa9a14da944aab217ee6e5b091514570f73b9e8f8`
- candidate verifier：`8d876a0c8c4d2708629de8f5ea91c7e7e6388db0c8aefbe050a33b1dfe7e3eea`
- VERA smoke outputs：`4699246c45cab3a78b6f71f3047097318b47e13103e0650d43ba799b5e279615`
