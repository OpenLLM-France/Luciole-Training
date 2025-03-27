#!/bin/bash

find "${OpenLLM_OUTPUT}/datasets" -type d -name "stats" | while read data_path; do
    relative_path="${data_path#"$DATA"/}"  # Remove $DATA prefix
    echo -e "\n---------------\nPath \$OpenLLM_OUTPUT/datasets/$relative_path\n---------------"
    python print_stats.py --stats_dir "$data_path" 
done