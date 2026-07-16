#!/bin/bash
set -e

MODEL_PATH="${MODEL_PATH:-models/instructblip-vicuna-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-official_linear_results}"
METHOD="${METHOD:-dlvhr-linear50-h18-official}"
DEVICES=(3 4 5 6 7)
NUM_CHUNKS=${#DEVICES[@]}

mkdir -p logs
pids=()
for chunk_idx in "${!DEVICES[@]}"; do
  device="${DEVICES[$chunk_idx]}"
  python main.py --model_path "$MODEL_PATH" --method "$METHOD" \
    --vhr_layers 18 --vhr_filter --vhr_score_mode directional_linear \
    --vhr_head_fraction 0.5 --vhr_positive_only \
    --datasets pope_popular,pope_adversarial,pope_random \
    --max_new_tokens 512 --output_dir "$OUTPUT_DIR" \
    --device "cuda:$device" --num_chunks "$NUM_CHUNKS" --chunk_idx "$chunk_idx" \
    > "logs/official_linear_pope_chunk${chunk_idx}_gpu${device}.log" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -ne 0 ]]; then exit "$status"; fi

bash scripts/cat_files.sh "$OUTPUT_DIR/instructblip-vicuna-7b"
conda run -n vhr python eval/eval.py \
  "$OUTPUT_DIR/instructblip-vicuna-7b/pope_*_0-$METHOD-512.json" --eval_pope
