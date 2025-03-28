#!/bin/bash

YAML_FILE="datasets_to_tokenize.yaml"
MAIN_PATH="$OpenLLM_OUTPUT/data"

while IFS= read -r line; do
    # Extract name
    if [[ $line =~ name:\ (.*) ]]; then
        name="${BASH_REMATCH[1]}"
    fi

    # Extract path
    if [[ $line =~ path:\ (.*) ]]; then
        datapath="${BASH_REMATCH[1]}"
        
        # Check if the raw dataset exists
        raw_dataset_path="$MAIN_PATH/raw_datasets_debug"
        if [[ -d "$raw_dataset_path/$datapath" ]]; then
            # Check if the log file for this dataset exists
            if [[ ! -f "$LOG_FOLDER/$name.log" ]]; then
                echo "--------------------------------------"
                echo "🚀 Processing dataset: $name"
                echo "📂 Path: $raw_dataset_path/$datapath"
                echo "--------------------------------------"

                # Run the sbatch command with the paths
                sbatch --job-name=tok_$name tokenize_one_dataset.slurm "$raw_dataset_path/$datapath" "$MAIN_PATH/tokens_debug/$name"
            else
                echo "--------------------------------------"
                echo "⏩ Skipping $name, already processed."
                echo "--------------------------------------"
            fi
        else
            echo "--------------------------------------"
            echo "❌ Raw dataset not found for $name at $raw_dataset_path/$datapath."
            echo "--------------------------------------"
        fi
    fi 
done < "$YAML_FILE"