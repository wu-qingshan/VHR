#!/bin/bash
set -e

MODEL_PATH="${MODEL_PATH:-models/instructblip-vicuna-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-dev2_results}"
IDS_FILE="${IDS_FILE:-experiment/dev_chair_ids.json}"

mkdir -p logs
common=(--model_path "$MODEL_PATH" --vhr_layers 18 --vhr_filter --vhr_head_fraction 0.5 --datasets chair --chair_seeds 0 --chair_ids_file "$IDS_FILE" --max_new_tokens 512 --output_dir "$OUTPUT_DIR" --num_chunks 1 --chunk_idx 0)

python main.py "${common[@]}" --method vhr-nr3-h18 --no_repeat_ngram_size 3 --device cuda:4 > logs/dev2_vhr_nr3.log 2>&1 &
p0=$!
python main.py "${common[@]}" --method csvhr-nr3-h18 --no_repeat_ngram_size 3 --device cuda:5 > logs/dev2_csvhr_nr3.log 2>&1 &
p1=$!
python main.py "${common[@]}" --method vhr-rp11-h18 --repetition_penalty 1.1 --device cuda:6 > logs/dev2_vhr_rp11.log 2>&1 &
p2=$!
python main.py "${common[@]}" --method csvhr-rp11-h18 --repetition_penalty 1.1 --device cuda:7 > logs/dev2_csvhr_rp11.log 2>&1 &
p3=$!

status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
  wait "$pid" || status=1
done
exit "$status"

