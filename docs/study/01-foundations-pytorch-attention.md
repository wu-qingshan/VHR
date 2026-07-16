# 第 1–2 周：神经网络与 Transformer 基底（对齐本仓库）— 零基础加长版

> 本文档写给 **编程会一点、但没系统学过深度学习 / Transformer** 的同学。你已能读 Python；若线性代数也完全忘了，先完成下面「零基础垫脚石」再打勾进入主线。

---

## 文档怎么用（建议你照做）

每日约 **2–3 小时** 时推荐拆成三块（与总计划一致）：

| 时段 | 做什么 | 判据 |
|------|--------|------|
| 45–60 分钟 | 读 **一个** 新概念 + 看视频/教程的一小节 | 能关掉资料后 **用自己的话复述 30 秒** |
| 60–90 分钟 | 对照本讲义读 [`vhr.py`](../../vhr.py) 对应行号 | 能对每一行说「在张量上做了什么」 |
| 15 分钟 | 写 **3 条笔记**：今天定义 / 和 VHR 的关系 / 明天要问的问题 | 不必漂亮，能复查即可 |

**不要做的事**：同时开 3 门课并行；一周换一套教材。**固定一条主线**（下面「推荐学习顺序」）+ **官方文档查漏补缺** 就足够。

---

## 学习目标（读完后你应该达到什么）

读完第 1–2 周后，你应该能 **闭卷口述**（可对照代码指行号）下面这些，并它们在 [`vhr.py`](../../vhr.py) 里的落点：

- 解码器里一次 forward：`hidden_states` → Q/K/V → 多头 reshape → SDPA → 合并 head → `o_proj`。
- **RoPE**：位置信息加在 **Q、K** 上；与本仓库 `rotary_emb` / `apply_rotary_pos_emb` 的关系。
- **KV Cache（`past_key_value`）**：生成时为何不每步都算整句的 K/V。
- **Prefill vs decode**：`q_len > 1` 与 `q_len == 1` 分别在干什么；与 VHR 里 `attn_output.shape[-2]` 分支的关系。
- **SDPA**：`scaled_dot_product_attention`；`is_causal` 与 `attn_mask` 二选一逻辑。
- **GQA**：`num_key_value_heads`、`repeat_kv` 在做什么。

---

## 一、前置知识分层：你需要什么、缺什么补什么

### 1.1 必须掌握（否则读不动 `vhr.py`）

| 领域 | 具体内容 | 掌握到什么程度算够 |
|------|----------|----------------------|
| **Python** | 函数、类、`**kwargs`、浅拷贝/深拷贝（读 `generation.py` 时会遇到） | 能跟调用栈，不怵 `partial` |
| **NumPy 级直觉** | 多维数组、`.shape`、axis 含义 | 能口头说 `(2,3,4)` 哪一维是 batch |
| **极简线性代数** | 向量、矩阵乘法、转置 \(A^\top\) | 知道 \(QK^\top\) 的维度要能对齐（见下文练习） |
| **概率入门** | softmax 把一组实数变成「和为 1 的权重」 | 知道 attention 里 softmax 在「对哪些维度」做 |

### 1.2 强烈建议（第一周末前补齐）

| 领域 | 具体内容 | 掌握到什么程度算够 |
|------|----------|----------------------|
| **PyTorch 张量** | `tensor.shape`、`view`/`reshape`、`transpose`、`contiguous`、`device`、`dtype` | 见第二节「合格线」 |
| **自动求导（概念即可）** | `requires_grad`、推理时一般 `torch.no_grad()` | 本仓库推理为主，只需知道「训练才要梯度」 |
| **神经网络一口饭** | 一层线性变换 + 非线性（GELU/SiLU）堆起来 | 能指认 `q_proj` 就是 `nn.Linear` |

### 1.3 可选（锦上添花，不阻塞读 VHR）

- 信息论、变分推断、强化学习 — **读 VHR 不需要**。
- CUDA 编程 — **不需要**；知道「GPU 上张量在 `.cuda` 或 `.to(device)`」即可。

### 1.4 若线性代数几乎为零：最小垫脚石（约 2～4 小时）

1. 复习：**向量点积**、**矩阵 × 向量** 的几何直觉（投影、加权组合）。  
2. 会做一道手算题：\(A\) 形状 \((m,n)\)，\(B\) 形状 \((n,p)\)，乘积形状是？  
3. 知道 **Softmax**：输入 \((x_1,\ldots,x_n)\)，输出 \(\frac{e^{x_i}}{\sum_j e^{x_j}}\)，用于把「打分」变「概率式权重」。

**资源**：可搜「3Blue1Brown 线性代数」前 4 集（中文可找带字幕搬运）或国内「麻省理工线性代数」导读；只看与「矩阵乘法、维度」相关的部分即可。

---

## 二、PyTorch：要学到什么深度、怎么学、去哪学

### 2.1 你需要会什么（对照 `vhr.py`）

| 技能点 | 在代码里的例子 | 合格线（自测） |
|--------|----------------|----------------|
| 读 shape | `bsz, q_len, _ = hidden_states.size()`（L57） | 立刻反应：最后一维是 `hidden_size` |
| 线性层在干什么 | `self.q_proj(hidden_states)`（L59） | 说出：最后一维从 `hidden` 变 `num_heads * head_dim`（由模块内部决定） |
| `view` + `transpose` | L63-65 | 能解释：为何要先变成 `(B, T, H, D)` 再转成 `(B, H, T, D)` —— **为了 SDPA 把 head 放在维 1** |
| `contiguous` | L94-97 | 知道：`transpose` 可能不连续，有些算子要求连续内存 |
| `dtype` / 半精度 | 仓库里 `float16`（见 [`utils.py`](../../utils.py)） | 知道：推理省显存，数值范围比 float32 窄 |

### 2.2 怎么学

1. **先 30 分钟** 过一遍官方「Tensors」教程，在 Python 里自己建两个随机张量做 `matmul`。  
2. **再 1 小时** 只练「改 shape」：同一个 `(2, 8, 4096)` 张量，用 `view` 变成 `(2, 8, 32, 128)` 再 `transpose(1,2)` 得到 `(2, 32, 8, 128)`，打印 `shape` 与 `stride()`（可选）。  
3. **最后** 打开 [`vhr.py`](../../vhr.py)，从 L57 读到 L110，**每行写中文注释**（给自己看即可）。

### 2.3 去哪学（免费、稳定）

| 资源 | 适合做什么 |
|------|------------|
| [PyTorch 官方：Tensors](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html) | 张量基础 |
| [PyTorch 官方：Autograd 简介](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) | 知道梯度即可，不用深 |
| 中文：搜「PyTorch 官方教程 张量」多数为搬运翻译 | 适合喜欢看中文的同学 |

---

## 三、深度学习最小概念包（只为读懂推理代码）

### 3.1 监督学习一句话

有一堆输入 \(x\) 和标签 \(y\)，模型带参数 \(\theta\)，训练时改 \(\theta\) 让预测接近 \(y\)。**本仓库几乎只做推理**：`model.eval()` + `torch.inference_mode()`，参数固定。

### 3.2 语言模型在干什么

把 token 序列 \(x_1,\ldots,x_T\) 喂进网络，在每一步输出 **下一个 token 的得分分布**（logits），再 softmax 成概率。Decoder-only（如 Llama）是 **只看左侧上下文** 的因果模型。

**掌握程度**：能画「输入 embedding → 若干层 Transformer block → 输出 logits」的方框图即可。

### 3.3 去哪学

- Andrej Karpathy 的 *Neural Networks: Zero to Hero*（YouTube / GitHub，英文）— 从头写 micrograd、MLP，**只需看前几集** 建立直觉。  
- 若时间紧：读 [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)（英文图解）或中文转载版，**重点看 Encoder-Decoder 里 Decoder 部分**（本仓库语言模型侧是 decoder-only，结构类似 decoder stack）。

---

## 四、Transformer 解码块里有什么（精读级）

一小块通常包含：

1. **Self-Attention**（本仓库改的正是这里）：每个位置从「允许看到的所有位置」取信息。  
2. **FFN（前馈子层）**：对每个位置单独做两遍线性+激活。  
3. **残差 + 归一化**（Pre-Norm 或 Post-Norm，Llama 系常见 Pre-Norm）：缓解梯度问题，具体顺序可查模型 config，**读 VHR 不必死记**。

**你要掌握的程度**：能说清「attention 的输出和 `hidden_states` 是同形状序列，再接 FFN」，并知道 **本仓库只管 attention 里的 `vhr_forward`**，没有把 FFN 改掉。

---

## 五、缩放点积注意力（详细版）

### 5.1 直觉

对每个 **query 位置**（可以理解为「当前词在问：我该从哪些词取信息？」），用它和所有 **key** 的相似度得到权重，再对 **value** 加权求和。相似度常用点积；除以 \(\sqrt{d_h}\) 防止点积数值太大导致 softmax 极陡。

### 5.2 单个 head 的公式

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right) V
\]

- \(Q,K,V\) 的最后一维都是 \(d_h\)（head 维度）。  
- \(QK^\top\)：query 位置 × key 位置 的得分矩阵，形状涉及 `(q_len, k_len)`。

### 5.3 多头（Multi-Head）

同一层有 **多个 head**，每个 head 有一套小的 \(Q_h,K_h,V_h\)，最后再 **拼接** 所有 head，经 **`o_proj`** 投影回原隐藏维。好处：不同 head 可以关注不同模式（语法、局部、长远依赖等）。

**在 `vhr.py` 里**：SDPA 一次算完所有 head；VHR 在 **head 维** 上乘以不同倍数（原版离散选 head，A-VHR 连续 ratio）。所以你必须熟悉 **`attn_output` 哪个维度是 head** —— SDPA 输出约定为 `(batch, num_heads, q_len, head_dim)`。

### 5.4 对齐代码

- [`vhr.py`](../../vhr.py) L103-L110：`torch.nn.functional.scaled_dot_product_attention(...)`  
- **`is_causal`**（L101）：无自定义 mask 且 `q_len>1` 时 `True`，用内核做因果遮蔽；若有 `attn_mask`（如 padding），则 `False`，由 mask 定义可见性。  
- **`contiguous`**（L94-L97）：CUDA + 自定义 mask 时的兼容性处理。

### 5.5 掌握程度自检

- [ ] 给定 `q_len`、`k_len`，说出注意力权重矩阵（未 softmax 前）的两维尺寸。  
- [ ] 解释：为什么解码时经常出现 `q_len=1`、而 `k_len` 随 cache 增大。

### 5.6 去哪学

- Lilian Weng：《Attention Mechanisms》博客（可选，英文）。  
- Stanford CS224N 讲义中 Attention 小节（可免费下载 PDF）。

---

## 六、分组查询注意力 GQA / MQA（Llama 常见）

有时 **KV 的 head 数** 少于 **Q 的 head 数**：少数组 key/value 被多组 query **共享**。实现上先做 `repeat_kv`，把 K/V **在 head 维上复制扩展**，再送进 SDPA，这样接口仍是「每组 Q 对每个 head 的 K,V」。

**本仓库**：L85-86 `repeat_kv`。  
**掌握程度**：能口头解释「为何 K/V head 可以再复制给多个 Q head」，不必会推导论文。

---

## 七、RoPE（旋转位置编码）

### 7.1 和「加在 embedding 上的位置向量」有什么不同

传统做法是位置向量加到词向量上。**RoPE** 把相对位置编码进 **Q、K 的旋转**：只依赖 token 之间的距离，对部分相对位置 extrapolation 更友好（细节不必一次吃完）。

### 7.2 在本仓库的流程

[`vhr.py`](../../vhr.py)：`rotary_emb` 产出 cos/sin → `apply_rotary_pos_emb(query_states, key_states, cos, sin)`。若 Transformers 传入 `position_embeddings`，则直接用外部的 cos/sin。

### 7.3 和 KV Cache 为何绑在一起

解码步只新来 **一个** query 位置，但 key/value 序列变长；RoPE 要给 **新位置** 正确的旋转，cache 里常存 **已旋转过的 K** 或配合 `cache_position` 更新。你**第一遍**只需建立印象：「cache 更新带上了 sin/cos/cache_position」（见 L80-L83）。

### 7.4 去哪学

- Transformer 文档里 **Rotary Embedding** 章节。  
- 论文 *RoFormer* / *LLaMA* 位置编码描述（选读摘要即可）。

### 7.5 掌握程度

- [ ] 能说出「RoPE 作用在 Q、K，不是 V」  
- [ ] 能指出本仓库 L67-L78 在做什么（不必会推旋转矩阵）

---

## 八、KV Cache 与 Prefill / Decode

### 8.1 无 cache 时生成有多浪费

每生成一个新 token，若从头算整句，计算量随长度平方增长。**KV cache** 存下已经算过的各层 K、V，新步只对 **新 token** 算 Q，并与 **历史 K、V** 做 attention。

### 8.2 Prefill vs decode（口语）

- **Prefill**：第一次把 **整段 prompt**（可能 `q_len` 很大）并行算完，并把 K/V 写入 cache。  
- **Decode**：之后每步 `q_len` 通常为 1，只更新 cache、只算当前步 logits。

### 8.3 与 VHR 的关系（对接后几周）

[`vhr.py`](../../vhr.py) 用 `attn_output.shape[-2]`（即 **query 序列长度**）区分：

- **`> 1`**：视为 **prefill 阶段**（含 VHR 的两次对比 prefill，详见 [03-lvlm-vhr-avhr.md](03-lvlm-vhr-avhr.md)）。  
- **`== 1`**：**decode 步**，复用已算好的 `aug_heads` 或 `aug_ratio_per_head`。

### 8.4 掌握程度

- [ ] 能向非程序员解释「为什么聊天越长首字慢、后面每字时间相对稳定」（简化版：首包 prefill 长）。  
- [ ] 能在 [`vhr.py`](../../vhr.py) L115-L146 标出「哪段对应 prefill、哪段对应 decode」。

### 8.5 去哪学

- HuggingFace 文档：*Cache*、*generate* 概念页。  
- 搜「LLM KV cache 图解」中文文章，看 1～2 篇即可，以 **一致的几何直觉** 为准，不必背实现细节。

---

## 九、SDPA 与 `output_attentions`（本仓库特殊分支）

[`vhr.py`](../../vhr.py) L40-L55：若 `output_attentions=True`，走 `super().forward(...)`，因为 **SDPA 默认不返回注意力权重矩阵**。  
**掌握程度**：知道「要权重 → 常要 eager 或手动实现」即可；本仓库 VHR 路径用 SDPA，一般 **不开** `output_attentions`。

---

## 十、推荐学习顺序（2 周示例，可放慢到 3 周）

| 天数块 | 主题 | 建议资料 | 代码任务 |
|--------|------|----------|----------|
| 第 1–3 天 | PyTorch 张量 + 矩阵乘维度 | PyTorch Tensor 教程 | 自建 `(B,T,H,D)` 做 `transpose` |
| 第 4–5 天 | Softmax / 单层 MLP 直觉 | Karpathy 前 2 课 或任意入门 | 无 |
| 第 6–8 天 | Transformer Decoder、Self-Attention | Illustrated Transformer（Decoder） | 手画 QKV 流向图 |
| 第 9–10 天 | RoPE + KV Cache 概念 | HF Llama + 中文博客 1 篇 | 读 `vhr.py` L57-83 |
| 第 11–12 天 | GQA、SDPA、is_causal | PyTorch SDPA 文档 | 读 `vhr.py` L85-110 |
| 第 13–14 天 | 综合 + VHR 插入点 | 重读全文 | 完成下方「进阶实践」 |

若每天只有 2 小时：把「天数块」×1.5 倍拉长即可。

---

## 十一、实践作业（由易到难）

### 入门级（≤30 分钟）

**A.** 给定 `bsz=1, q_len=4, hidden=4096, num_heads=32, head_dim=128`，写完 `q_proj` 后理想 shape 是什么？经过 `view(bsz,q_len,num_heads,head_dim).transpose(1,2)` 之后？

**B.** 在 HF 文档打开 **LlamaSdpaAttention**，对照 [`vhr_forward`](../../vhr.py)：列步骤 1～N，并用一句话标出「VHR 加在 SDPA **之后**、o_proj **之前**」（见 L112 之后）。

### 进阶（1～2 小时，强烈建议）

写一个 **极小** 脚本（CPU 即可）：随机 `Q,K,V`，形状 `(1, num_heads, q_len, head_dim)`，手动实现一次 softmax\(Q K^\top / \sqrt{d}\) \(V\)（用 `torch.einsum` 或 `@`），与 `scaled_dot_product_attention` 比对 max 误差。你会彻底理解 **`q_len`** 出现在哪两维。

---

## 十二、本周结束自检清单（打勾再走）

- [ ] 能解释 `now_attn_output = attn_output[0, :, -1, :]`：为何取 batch 0、`:` 保留所有 head、`-1` 是最后一个 query 位置。  
- [ ] 能说明 `repeat_kv` 之后 Q 与 K 在 head 维对齐。  
- [ ] 能解释 `output_attentions=True` 时为何要 `super().forward`（第五节 + 第九节）。  
- [ ] **闭卷**：用 1 分钟说清 prefill 与 decode 的 `q_len` 差异，以及 VHR 为何关心这个差异。

---

## 十三、常见挫折与对策

| 现象 | 对策 |
|------|------|
| 一看公式就懵 | 先只看 **维度追踪**（哪一维是多少），公式当「注解」 |
| `transpose` / `view` 晕 | 只画 **三维方块图**（batch 固定为 1） |
| 官方 Llama 代码太长 | 只读 **`LlamaSdpaAttention.forward`** + 本仓库 [`vhr.py`](../../vhr.py) |
| 想一次学完所有优化器/损失函数 | **停**：读 VHR 不需要 |

---

下一周继续读：[02-decoding-and-codegen-integration.md](02-decoding-and-codegen-integration.md)。
