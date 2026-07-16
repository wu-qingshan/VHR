#!/bin/bash

# 合并分片推理结果：把 {0..7}-<base> 拼成 <base>。
# 健壮性修复：
#   - 只合并实际存在的分片（支持任意 num_chunks，不再要求 0-7 全部存在）；
#   - 仅当 cat 成功写出合并文件后，才删除分片（避免「删了又没合并」导致结果丢失 → eval 出 nan）。

cd "$1" || exit 1

zero_files=$(find . -type f -name "0-*")

for zero_file in $zero_files; do
    base_filename=$(echo "$zero_file" | sed 's|/0-|/|')

    files_to_concat=()
    for i in {0..7}; do
        current_file=$(echo "$zero_file" | sed "s|/0-|/${i}-|")
        if [[ -s "$current_file" ]]; then
            files_to_concat+=("$current_file")
        fi
    done

    if [[ ${#files_to_concat[@]} -gt 0 ]]; then
        if cat "${files_to_concat[@]}" > "$base_filename"; then
            echo "Created $base_filename by concatenating ${#files_to_concat[@]} chunk(s)"
            for file in "${files_to_concat[@]}"; do
                rm -f "$file"
                echo "Removed $file"
            done
        else
            echo "WARNING: failed to create $base_filename, chunks kept"
        fi
    fi
done

cd ..
