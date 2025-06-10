# Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence

Official implementation of our paper "Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence" in ACL 2025.

## Install
```
conda create -n vhr python=3.10
conda activate vhr
pip install -r requiresments.txt
```

## Prepare datasets and models
Download [MSCOCO](https://cocodataset.org/#download) and [LLaVA-Bench (In-the-Wild)](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/tree/main) and organize the data as follows in `./data` .
```
├── coco
│     ├── val2014
│     └── annotations
│           ├── captions_val2014.json
│           └── instances_val2014.json
├── pope
│     └── coco
│           ├── coco_pope_popular.json
│           ├── coco_pope_random.json
│           └── coco_pope_adversarial.json
└── llava-bench-in-the-wild
            ├── images
            └── questions.jsonl
```
Download [LLaVA-1.5-7b](https://huggingface.co/llava-hf/llava-1.5-7b-hf/tree/16952161b5e90aea6e332e36a6fe99024096dd0a), [LLaVA-NeXT-7b](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf/tree/30f8c4fe610ab808d4d3612c01456be5f725184f), and [instructBLIP-7b](https://huggingface.co/Salesforce/instructblip-vicuna-7b/tree/52ba0cb2c44d96b2fcceed4e84141dc40d2b6a92), and modify the `model_path` in inference scripts. 

For instructBLIP, modify the `text_config.eos_token_id` in `config.json` to `2`.

## Infer and Evaluate

Each inference script accepts a `method` parameter:

```bash
# Syntax
bash infer_script.sh <method>

# Available methods: greedy, beam, dola, vcd, opera, code, eah, vhr

# Examples
bash scripts/infer_llava1_5.sh vhr
bash scripts/infer_llava1_6.sh eah
bash scripts/infer_instructblip.sh greedy

# GPT4-V evaluation on LLaVABench-W
python eval/llava_wild.py --response1 <baseline.json> --response2 <compared.json> --api-key <api-key>
```
