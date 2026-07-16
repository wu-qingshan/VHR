# VHR 项目学习讲义（按计划拆周）

本目录把「读懂 VHR 仓库」的路线落实为可分周阅读的讲义与自检清单。**对应仓库根目录学习计划中的四周主题**（不改变该计划文件）。

| 周次主题 | 文档 | 学完应能做的事 |
|---------|------|----------------|
| 第 1–2 周：PyTorch + Attention / Decoder / RoPE / Cache | [01-foundations-pytorch-attention.md](01-foundations-pytorch-attention.md) | 对照 [vhr.py](../../vhr.py) 说出 QKV 形状、`scaled_dot_product_attention`、KV cache、`q_len==1` 的含义 |
| 第 3–4 周：解码与代码集成 | [02-decoding-and-codegen-integration.md](02-decoding-and-codegen-integration.md) | 能从 `generate` 走到 `_vhr_decoding`；解释 `Greedy_search`/`Beam`/`VHR_GENERATION` 的触发条件 |
| 第 5–6 周：LVLM + VHR / A-VHR | [03-lvlm-vhr-avhr.md](03-lvlm-vhr-avhr.md) | 画出 text-contrast prefill ↔ 全文 prefill ↔ decode 的流程；复述 `head_vary` 与 A-VHR 公式 |
| 第 7–8 周：论文参数、复现、评测 | [04-paper-repro-eval.md](04-paper-repro-eval.md) | CLI、`GenerationConfig`、评测脚本与 JSON 行列字段一一对应 |

## 建议使用方式（每日约 2–3 小时）

1. **概念块（45–60 分钟）**：只读讲义中当日章节 + 任选一门主线视频或 Transformers/Llama 官方文档节选。
2. **代码块（60–90 分钟）**：用 IDE 跳到讲义里给的符号；在文件内用搜索定位次要分支。
3. **复述（约 15 分钟）**：用三句话写给未来的自己：**今天搞懂的定义 / 与本仓库的连接 / 还剩什么疑问**。

## 源代码核心地图（先看关系再看细节）

```
main.py                     # CLI、数据集循环、generate_kwargs、各 method 分支
    └─► generation.replace_generation()  → 补丁 HuggingFace generate / GenerationConfig

generation.py               # VHR 解码环 _vhr_decoding；get_generation_mode 选模式
    └─► vhr.replace_attn     # 对目标层绑定 vhr_forward

vhr.py                      # LlamaSdpaAttention 风格：SDPA + VHR/A-VHR 头加权

utils.py                    # get_model / get_layers / InstructBLIP patch

eah.py                      # 另一路注意力改写（method 含 eah 时用 eager）
```

从 [README.md](../../README.md) 开始对照数据与模型路径；论文标题见仓库说明（ACL 2025）。
