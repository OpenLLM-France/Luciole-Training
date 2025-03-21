#!/bin/bash

find "$DATA" -type d -name "stats" | while read data_path; do
    relative_path="${data_path#"$DATA"/}"  # Remove $DATA prefix
    echo -e "\n---------------\nPath \$DATA/$relative_path\n---------------"
    python print_stats.py --stats_dir "$data_path" 
done