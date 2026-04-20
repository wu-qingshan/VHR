#!/bin/bash
# A-VHR (Adaptive-ratio VHR) inference for InstructBLIP.
# 用法:
#   bash scripts/infer_instructblip_avhr.sh avhr            # 默认 tau=1.0 alpha=0.5 rho_max=2.0
#   AVHR_TAU=0.5 AVHR_ALPHA=1.0 AVHR_RHO_MAX=2.5 bash scripts/infer_instructblip_avhr.sh avhr
# 也可以单卡前台跑：
#   CUDA_DEVICE=cuda:2 NUM_CHUNKS=1 bash scripts/infer_instructblip_avhr.sh avhr
# 后台不挂断：
#   nohup bash scripts/infer_instructblip_avhr.sh avhr > logs/avhr_$(date +%m%d_%H%M).log 2>&1 &

set -e

model_path="${MODEL_PATH:-models/instructblip-vicuna-7b}"

max_new_tokens="${MAX_NEW_TOKENS:-512}"
vhr_layers="${VHR_LAYERS:-14}"

# A-VHR 超参（可通过环境变量覆盖）
avhr_tau="${AVHR_TAU:-1.0}"
avhr_alpha="${AVHR_ALPHA:-0.5}"
avhr_rho_max="${AVHR_RHO_MAX:-2.0}"

method="${1:-avhr}"

# 给输出文件名加个后缀，避免和原版 vhr 结果混在一起
if [[ $method == avhr ]]; then
  method="avhr-h${vhr_layers}-t${avhr_tau}-a${avhr_alpha}-r${avhr_rho_max}"
fi

# 解析 model_name
_base=$(basename "$(realpath "$model_path")")
if [[ $_base == *"--"* ]]; then
  model_name="${_base##*--}"
else
  model_name="$_base"
fi

num_chunks="${NUM_CHUNKS:-4}"

mkdir -p logs

if [[ $num_chunks -eq 1 ]]; then
  device="${CUDA_DEVICE:-cuda:0}"
  echo "[A-VHR] single-card run on $device, method=$method"
  python main.py --model_path "$model_path" \
                 --method "$method" \
                 --vhr_layers "$vhr_layers" \
                 --vhr_filter \
                 --vhr_tau "$avhr_tau" \
                 --vhr_alpha "$avhr_alpha" \
                 --vhr_rho_max "$avhr_rho_max" \
                 --max_new_tokens "$max_new_tokens" \
                 --device "$device" \
                 --num_chunks 1 \
                 --chunk_idx 0
else
  echo "[A-VHR] parallel on $num_chunks GPUs (cuda:0..cuda:$((num_chunks-1))), method=$method"
  pids=()
  for i in $(seq 0 $((num_chunks-1))); do
    python main.py --model_path "$model_path" \
                   --method "$method" \
                   --vhr_layers "$vhr_layers" \
                   --vhr_filter \
                   --vhr_tau "$avhr_tau" \
                   --vhr_alpha "$avhr_alpha" \
                   --vhr_rho_max "$avhr_rho_max" \
                   --max_new_tokens "$max_new_tokens" \
                   --device "cuda:$i" \
                   --num_chunks "$num_chunks" \
                   --chunk_idx "$i" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || { echo "child $pid failed"; exit 1; }
  done
  bash scripts/cat_files.sh "results/$model_name"
fi

# 评估（CHAIR 与 POPE）。用 *xxx 通配既能匹配原始单卡的 0-... 也能匹配 cat 后的合并文件。
python eval/eval.py "results/$model_name/*chair_*-${method}-${max_new_tokens}.json" --eval_chair
python eval/eval.py "results/$model_name/*pope_*_0-${method}-${max_new_tokens}.json" --eval_pope
