#!/bin/bash
#SBATCH --job-name=ablation_launcher
#SBATCH --output=slurm_logs/%x_%j.out 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00 
#SBATCH --hint=nomultithread 
#SBATCH --account=qgz@cpu
#SBATCH --partition=cpu_p1
#CSBATCH --dependency=afterany:1843180:1843177:1843178:1843179:1843176:1843175:1843174:1843194:1843192:1843190:1843188:1843186:1843184:1843182

EXAMPLE_DIR_PATH="$OpenLLM_OUTPUT/ablations/regmix/$1"
MAX_JOBS=8  # TO DO CHANGE SYSTEM, LOOK IF SUBMITTED
count=0

for file in "$EXAMPLE_DIR_PATH"/*.json; do
    if [[ -f "$file" ]]; then
        echo "Launching slurm_pipeline.py for $file"
        python3 ../train/slurm_pipeline.py --config "$file" --arch ablation_llama90m --mode 1b --output_dir regmix/$1 --tasks "${@:2}"
        ((count++))
        if [[ $count -ge $MAX_JOBS ]]; then
            echo "Reached the maximum of $MAX_JOBS jobs. Stopping."
            break
        fi
    fi
done
