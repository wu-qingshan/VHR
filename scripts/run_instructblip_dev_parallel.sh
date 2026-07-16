#!/bin/bash
set -e

MODEL_PATH="${MODEL_PATH:-models/instructblip-vicuna-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-dev_results}"
IDS_FILE="${IDS_FILE:-experiment/dev_chair_ids.json}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

mkdir -p logs

python main.py --model_path "$MODEL_PATH" --method vhr-dev-h18 \
  --vhr_layers 18 --vhr_filter --vhr_head_fraction 0.5 \
  --datasets chair --chair_seeds 0 --chair_ids_file "$IDS_FILE" \
  --max_new_tokens "$MAX_NEW_TOKENS" --output_dir "$OUTPUT_DIR" \
  --device cuda:0 --num_chunks 1 --chunk_idx 0 \
  > logs/dev_d0_vhr.log 2>&1 &
pid_d0=$!

python main.py --model_path "$MODEL_PATH" --method csvhr-dir50-h18 \
  --vhr_layers 18 --vhr_filter --vhr_head_fraction 0.5 \
  --datasets chair --chair_seeds 0 --chair_ids_file "$IDS_FILE" \
  --max_new_tokens "$MAX_NEW_TOKENS" --output_dir "$OUTPUT_DIR" \
  --device cuda:6 --num_chunks 1 --chunk_idx 0 \
  > logs/dev_d1_dir50.log 2>&1 &
pid_d1=$!

python main.py --model_path "$MODEL_PATH" --method csvhr-dir25-h18 \
  --vhr_layers 18 --vhr_filter --vhr_head_fraction 0.25 \
  --datasets chair --chair_seeds 0 --chair_ids_file "$IDS_FILE" \
  --max_new_tokens "$MAX_NEW_TOKENS" --output_dir "$OUTPUT_DIR" \
  --device cuda:7 --num_chunks 1 --chunk_idx 0 \
  > logs/dev_d2_dir25.log 2>&1 &
pid_d2=$!

status=0
wait "$pid_d0" || status=1
wait "$pid_d1" || status=1
wait "$pid_d2" || status=1
exit "$status"

