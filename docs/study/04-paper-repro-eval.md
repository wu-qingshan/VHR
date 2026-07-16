# 第 7–8 周：论文—代码—实验对齐、复现与评测

## 学习目标

- 建立 **超参双表**：论文/经验设定 ↔ [`main.py` CLI](../../main.py) ↔ [`GenerationConfig`](../../generation.py) 字段。
- 能描述 **从推理 JSON 到指标** 的数据流，并能在本机按 README 检查数据与模型是否就绪。

---

## 1. 超参映射（VHR / A-VHR）

| 概念（论文/文档） | `main.py` 参数 | `GenerationConfig`（`generation_config_init`） | 代码消费处 |
|-------------------|----------------|--------------------------------------------------|------------|
| 后若干层启用 VHR | `--vhr_layers`（默认 14） | 经 `generate_kwargs` → `vhr_layers` | `replace_attn` 的 `target_layers`；[`main.py`](../../main.py) 用最后 N 层索引列表 |
| 额外包含第 1 层 | `--vhr_layer1` | 同上（列表含 `1`） | [`vhr.py::replace_attn`](../../vhr.py) |
| 头放大倍数（原版） | `--vhr_aug_ratio`（默认 2） | `vhr_aug_ratio` | `vhr_forward` 的 `aug_ratio` |
| 离群头 filter | `--vhr_filter` | `vhr_filter` | `vhr_forward` L122-L124 |
| A-VHR 温度 | `--vhr_tau` | `vhr_tau` → `avhr_tau` | `vhr_forward` |
| 层深衰减强度 | `--vhr_alpha` | `vhr_alpha` → `avhr_gamma_l` 计算 | `_compute_gamma_l` |
| 单头最大倍率 | `--vhr_rho_max` | `vhr_rho_max` | `vhr_forward` |
| 启用 A-VHR | `method` 字符串含 `avhr` | `vhr_use_avhr=True` | `replace_attn` 的 `use_avhr` |
| 图像 token 边界 | （推理时计算） | `vhr_image_start`, `vhr_image_end` | `_vhr_decoding` 裁切对比输入 |

**注意**：`method` 是约定字符串（如 `vhr`、`avhr_vhr` 等组合），[`main.py`](../../main.py) 用子串 `"vhr" in self.method` / `"avhr" in self.method` 打开分支；具体命名以你使用的 `scripts/*.sh` 为准。

---

## 2. 数据与模型（README 核对清单）

摘自 [`README.md`](../../README.md)，复现前逐项打勾：

- [ ] **COCO**：`data/coco/val2014` + `annotations/captions_val2014.json`、`instances_val2014.json`
- [ ] **POPE**：`data/pope/coco/*.json`
- [ ] **LLaVA-Bench-W**：`data/llava-bench-in-the-wild/images` + `questions.jsonl`
- [ ] **权重**：LLaVA-1.5-7b、LLaVA-NeXT-7b、InstructBLIP-7b；脚本中修改 `model_path`
- [ ] **InstructBLIP**：`config.json` 内 `text_config.eos_token_id` 改为 `2`（README 说明）

---

## 3. 推理输出 JSONL 形态

[`main.py::infer_dataset`](../../main.py) 每行一个 JSON 对象：

- `question_id`, `text`（模型回答）, `gt`, `ans_conf`, `question`

合并分片：`complete_output_file_path` 命名规则见 L67-L68；评测脚本通常读**完整**或按行累积文件。

---

## 4. 评测入口

| 任务 | 脚本/函数 | 备注 |
|------|-----------|------|
| CHAIR / 综合 | [`eval/eval.py`](../../eval/eval.py) | `eval_chair(answer_file)`；内部 COCO + CHAIR 指标 |
| POPE | 同文件 `eval_pope` | 需预测与标签对齐 |
| LLaVA-Wild | [`eval/llava_wild.py`](../../eval/llava_wild.py) | `--response1` / `--response2` + `--api-key`；比较两条 JSON 回答 |

README 示例：

```bash
python eval/llava_wild.py --response1 <baseline.json> --response2 <compared.json> --api-key <api-key>
```

（具体参数以脚本 `argparse` 为准。）

---

## 5. 阅读论文时的建议笔记结构

为每一节建四列：**主张 / 与本仓库对应文件 / 超参 / 可做的消融**。重点对照：

- 视觉敏感 head 的 **定义** ↔ 代码中的 `head_vary`。
- **作用层范围** ↔ `vhr_layers` 与可选 layer1。
- **对比输入构造** ↔ `_vhr_decoding` 去图或截 embed。

---

## 6. `generation.py` 体量控制（复习用检索式）

只读这些符号即可覆盖 VHR 主路径：

- `replace_generation`
- `get_generation_mode` 中 `VHR_GENERATION`
- `GenerationMode.VHR_GENERATION` 分支
- `_vhr_decoding`
- `generation_config_init` 中 `vhr_` 字段块

---

## 7. 验收（与总计划一致）

1. **口头**：两次 prefill 的触发条件 + 加权作用在 **head 维** 的方式。  
2. **指路**：给定 `method`（greedy / vcd / vhr / eah 等），说出主要改动发生在 **哪个模块**。  
3. **数据**：从 `main.py` 写出的一行 JSON，到 `eval_*.py` 所需字段，无断档。

---

## 8. 本周自检清单

- [ ] 有一张自己的超参表（含默认值）。
- [ ] 能独立完成「小样本单次 `generate` + 打印一行 JSON」的心智推演（不一定要真跑）。
- [ ] 知道 A-VHR 的单一事实来源除代码外还有 [`experiment/A-VHR.md`](../../experiment/A-VHR.md)。

学完后可随时回到索引 [docs/study/README.md](README.md) 重刷薄弱周。
