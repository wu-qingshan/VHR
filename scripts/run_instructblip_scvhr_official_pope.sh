#!/bin/bash
set -e

MODEL_PATH="${MODEL_PATH:-models/instructblip-vicuna-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-scvhr_official_results}"
METHOD="${METHOD:-scvhr-h18-a05-b01-official}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
DEVICES=(0 1 2 3)
NUM_CHUNKS=${#DEVICES[@]}

mkdir -p logs
pids=()
for chunk_idx in "${!DEVICES[@]}"; do
  device="${DEVICES[$chunk_idx]}"
  python main.py --model_path "$MODEL_PATH" --method "$METHOD" \
    --vhr_layers 18 --vhr_filter --vhr_score_mode magnitude \
    --vhr_head_fraction 0.5 --vhr_contrast_mode remove \
    --vhr_logit_alpha 0.5 --vhr_logit_beta 0.1 \
    --datasets pope_popular,pope_adversarial,pope_random \
    --max_new_tokens "$MAX_NEW_TOKENS" --output_dir "$OUTPUT_DIR" \
    --device "cuda:$device" --num_chunks "$NUM_CHUNKS" --chunk_idx "$chunk_idx" \
    > "logs/official_scvhr_pope_chunk${chunk_idx}_gpu${device}.log" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

bash scripts/cat_files.sh "$OUTPUT_DIR/instructblip-vicuna-7b"
/home/nlp/anaconda3/envs/vhr/bin/python eval/eval.py \
  "$OUTPUT_DIR/instructblip-vicuna-7b/pope_*_0-$METHOD-$MAX_NEW_TOKENS.json" --eval_pope
