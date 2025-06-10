model_dir=models--llava-hf--llava-1.5-7b-hf
model_path="/path/to/model_cache/$model_dir"

max_new_tokens=512
vhr_layers=14

method=$1

if [[ $method == vhr ]]; then
method=vhr-l1h$vhr_layers
fi

arr=(${model_dir//--/ })
model_name=${arr[2]}

num_chunks=8
for i in $(seq 0 $((num_chunks-1))); do
  python main.py --model_path $model_path \
                 --method $method \
                 --vhr_layers $vhr_layers \
                 --vhr_filter \
                 --vhr_layer1 \
                 --max_new_tokens $max_new_tokens \
                 --device cuda:$i \
                 --num_chunks $num_chunks \
                 --chunk_idx $i &
done

wait

bash scripts/cat_files.sh results/$model_name

python eval/eval.py results/$model_name/chair_\*-$method-$max_new_tokens.json --eval_chair
python eval/eval.py results/$model_name/pope_\*_0-$method-$max_new_tokens.json --eval_pope
