#!/bin/bash

find "${OpenLLM_OUTPUT}/datasets" -type d -name "stats" | while read data_path; do
    output=$data_path/merged_stats.json
    if [ ! -f "$output" ]; then
        echo -e "\n---------------\nProcessing $data_path\n---------------"
        python ~/datatrove/src/datatrove/tools/merge_stats.py "$data_path" --output "$output" 
    fi
done
