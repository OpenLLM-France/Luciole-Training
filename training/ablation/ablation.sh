#!/bin/bash
# USAGE : bash ablation.sh folder_containing_datamixes evaluation_task
# folder_containing_datamixes is the path of a folder in $OpenLLM_OUTPUT/ablations/regmix
# See training/evaluation/tasks for the available tasks

ENVIRONMENT="dev"

if [[ "$ENVIRONMENT" == "dev" ]]; then
    QOS="qos_gpu_h100-dev"
    MAX_JOBS=8
else
    QOS="qos_gpu_h100-t3"
    MAX_JOBS=100
fi

DATAMIXES_DIR_PATH="$OpenLLM_OUTPUT/ablations/regmix/$1"
count=0

for file in "$DATAMIXES_DIR_PATH"/*.json; do
    if [[ -f "$file" ]]; then
        echo "Launching slurm_pipeline.py for $file"
        output=$(python3 ../train/slurm_pipeline.py --config "$file" \
                                                    --arch ablation_llama90m \
                                                    --qos $QOS \
                                                    --mode 1b \
                                                    --output_dir regmix/$1 \
                                                    --tasks "${@:2}")
        if [[ $output == *"Job submitted"* ]]; then
            ((count++))
            if [[ $count -ge $MAX_JOBS ]]; then
                echo "Reached the maximum of $MAX_JOBS jobs. Stopping."
                break
            fi
        fi

    fi
done
