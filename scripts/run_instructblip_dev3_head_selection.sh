#!/bin/bash
set -e
common=(--model_path models/instructblip-vicuna-7b --vhr_layers 18 --vhr_filter --datasets chair --chair_seeds 0 --chair_ids_file experiment/dev_chair_ids.json --max_new_tokens 512 --output_dir dev3_results --num_chunks 1 --chunk_idx 0)
mkdir -p logs
python main.py "${common[@]}" --method cvhr-calibrated-h18 --vhr_score_mode calibrated --vhr_head_fraction 0.5 --device cuda:4 > logs/dev3_calibrated.log 2>&1 & p0=$!
python main.py "${common[@]}" --method dlvhr-linear50-h18 --vhr_score_mode directional_linear --vhr_head_fraction 0.5 --vhr_positive_only --device cuda:5 > logs/dev3_linear50.log 2>&1 & p1=$!
python main.py "${common[@]}" --method dsvhr-dir375-h18 --vhr_score_mode directional --vhr_head_fraction 0.375 --vhr_positive_only --device cuda:6 > logs/dev3_dir375.log 2>&1 & p2=$!
python main.py "${common[@]}" --method dsvhr-dir625-h18 --vhr_score_mode directional --vhr_head_fraction 0.625 --vhr_positive_only --device cuda:7 > logs/dev3_dir625.log 2>&1 & p3=$!
status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid" || status=1; done
exit "$status"

