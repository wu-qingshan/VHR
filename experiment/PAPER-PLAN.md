# PAPER-PLAN：审计论文写作计划与分工

> 日期：2026-09-12
> 分支：`research/qwen-mgr`
> 素材基线：tag `audit-llava16-complete-20260911` 及 AUDIT-PLAN.md 第 2 节全部冻结索引
> 目标：EMNLP 2027 主会长文（8 页正文 + 无限附录）；备选 Findings。**投稿定位由本人最终决策。**

## 0. 论文身份（题目候选，本人定夺）

1. *When Do Visual-Head Interventions Survive Backbone Modernization? A Preregistered Audit of Decoding-Time and Post-hoc Hallucination Mitigation*
2. *Self-Consistent Hallucination: An Audit of Why Head-Level and Mention-Level Mitigation Both Fade on Modern LVLMs*
3. *The Gradient Collapse of Vision-Head Interventions: A Four-Backbone Preregistered Audit*

叙事双主张（AUDIT-PLAN 第 1 节三主张合并表述）：
- **A（解码期）**：VHR/SC-VHR 家族的收益沿主干现代化呈梯度消失——InstructBLIP 强正
  → LLaVA-1.5 正但覆盖坍塌 → LLaVA-1.6 VHR 部分正/SC 增量消失 → Qwen2.5-VL 全家族
  负迁移 + 生成退化；
- **B（后处理）**：mention 级修订三层解剖——任务天花板存在（oracle −17.0 CHAIRs/
  100% 覆盖）、触发错位可修（人工 precision 0.158 → 实体提取 0.714）、但同模型证据
  自洽幻觉（幻觉 mention 中位分 0.818，AUC ≤0.78）是结构性上限；
- **C（方法论）**：预注册 + locked 集 + 冻结分析器的负结果审计协议。

## 1. 写作期纪律

1. **写作期不产生新实验数字**；正文所有数字必须来自已冻结文档/物化 JSON，每张表
   附来源（tag 或文件路径）。
2. 若审稿或写作暴露缺口需要新数字 → 新小额预注册（见第 4 节待定清单），先冻结
   再运行。
3. 数字一致性由脚本自审（我负责）：论文表格 ↔ 物化 JSON 自动对账。

## 2. 分工表（核心）

### 我（实验代理）负责

| 项 | 说明 | 交付物 |
|---|---|---|
| LaTeX 骨架与全部章节英文初稿 | `paper/main.tex`（article 类占位，投稿时换 ACL 模板，本机无编译器，源码供 Overleaf） | M1–M2 |
| 表格数字提取与对账 | 每表标注来源；对账脚本 `scripts/audit_paper_numbers.py` | M1、持续 |
| 图 v1 | matplotlib 脚本 + PDF：①主干梯度图（四主干 ΔCHAIRs + CI）②幻觉/正确 mention 分数分布 ③幻觉-覆盖-成本 Pareto ④触发精度阶梯（VERA→E0.2→MGR） | M3 |
| BibTeX 初稿 | VHR(ACL'25)、CHAIR、POPE、VCD、OPERA、DoLa、HALC、Woodpecker、LLaVA-1.5/NeXT、InstructBLIP、Qwen2.5-VL、AMBER/MMHal（如提及） | M2 |
| 复现附录 | tag/SHA256/门槛文档索引表自动生成 | M3 |
| 语言初稿润色一轮 | 语法与一致性，不改变主张 | M4 |

### 你（作者本人）负责——不可委托

| 项 | 说明 | 时机 |
|---|---|---|
| 科学主张终审 | Intro 的 claim 强度、Discussion 的因果解释边界（如 GQA/mRoPE 解释只能写"consistent with"而非"proved"） | M2 后逐节 |
| Related work 核对 | 我给的每条引用你读原文摘要确认表述无误 | M2 |
| 图的最终判定 | 可读性、信息密度、"论文品味" | M3 |
| 语言定稿与 voice | 全文润色；作者立场句（"we argue…"）由你定 | M4–M5 |
| 投稿决策 | 题目、venue（main/Findings）、作者列表、匿名化边界 | M5 |
| 补实验决策 | 第 4 节待定清单的取舍 | 随时 |

## 3. 章节映射与页数预算（8 页正文）

| § | 内容 | 页 | 数字来源 |
|---|---|---|---|
| 1 | Introduction（双主张 + 贡献列表） | 1 | — |
| 2 | Related Work（解码期/后处理/审计与预注册） | 0.75 | — |
| 3 | Audit I：head 干预跨主干（四主干表 + 退化分析 + 机制讨论） | 1.5 | SC-VHR.md §4、SC-VHR-LLAVA15.md §6、AUDIT-PLAN §6、QWEN-SC-VHR.md |
| 4 | Audit II：mention 级修订三层解剖（E0.1 表、E0.2 表、阶段 1 表 + 分数分布） | 1.75 | MGR-CEILING.md、QWEN-MGR-VERIFIER.md §7 |
| 5 | 审计协议（预注册/locked/冻结分析器） | 0.5 | 各预注册文档 |
| 6 | Discussion（为何梯度消失；self-consistency 的含义；跨模型证据是开放方向） | 0.75 | — |
| 7 | Limitations | 0.25 | AUDIT-PLAN §4 |
| 8 | Conclusion | 0.25 | — |
| 附录 | 全部门槛判定表、逐 seed 数字、复现索引、E0.2 协议 | 不限 | 物化 JSON |

## 4. 待定小额补测清单（需要才做，各自先小额预注册）

1. **InstructBLIP 与 LLaVA-1.5 的 vanilla 基线**（各 100 图 dev，方法=vanilla-512）：
   跨主干主表需要"干预 vs 自身 vanilla"口径；历史运行以 VHR 为基线，vanilla 未物化。
   审计型（report-only），预计单卡 <1 小时。
2. （可选）Qwen vanilla 在 llava 同款 100 图 dev 上的数字，统一图集口径。
3. （可选，若网络恢复）第二现代主干（InternVL2.5）替换 limitation 声明。

## 5. 里程碑

- **M1（今天起）**：`paper/` 骨架 + §3/§4 初稿（数字嵌入、TODO-NUM 标注待提取项）
- **M2**：§1/§2/§5/§6/§7/§8 初稿 + bib；交你逐节审
- **M3**：图 v1 + 附录 + 对账脚本全绿
- **M4**：你审改回传，我改稿 + 一轮润色
- **M5**：Overleaf 编译、交叉引用、匿名化、投稿包

## 6. 日志

- 2026-09-12：本计划冻结；`paper/` 骨架与 §3/§4 初稿开始。
