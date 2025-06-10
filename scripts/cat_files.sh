#!/bin/bash

# Change to the directory containing the files
cd $1

# Find all files starting with "0-"
zero_files=$(find . -type f -name "0-*")

for zero_file in $zero_files; do
    base_filename=$(echo $zero_file | sed 's/0-//')
    all_files_exist=true
    files_to_concat=()

    # Check for files with prefixes 0-7
    for i in {0..7}; do
        current_file=$(echo "$zero_file" | sed "s/0-/${i}-/")
        if [[ ! -s "$current_file" ]]; then
            all_files_exist=false
        else
        files_to_concat+=("$current_file")
        fi
    done
    # If all files exist, concatenate them
    if $all_files_exist; then
        cat "${files_to_concat[@]}" > "$base_filename"
        echo "Created $base_filename by concatenating files"
    fi
    # Remove existing files in the set
    for file in "${files_to_concat[@]}"; do
        rm -f "$file"
        echo "Removed $file"
    done
done

# for i in {1..7}; do
#     other_files=$(find . -type f -name "${i}-*")
#     for file in $other_files; do
#         rm -f "$file"
#         echo "Removed $file"
#     done
# done

cd ..