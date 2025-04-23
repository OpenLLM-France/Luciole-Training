#!/bin/bash
#SBATCH --job-name=eval 
#SBATCH --output=%x.out 
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64  
#SBATCH --time=02:00:00 
#SBATCH --hint=nomultithread 
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

module purge
module load arch/h100
module load anaconda-py3/2024.06
conda activate eval-env

export OpenLLM_OUTPUT=$ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$ALL_CCFRSCRATCH/.cache/huggingface
export HF_HUB_OFFLINE=1

expe_name="datamix_.5_fra_Latn_.5_eng_Latn_4n_20b"
PARENT_DIR="$OpenLLM_OUTPUT/ablations/train/language_ablations/$expe_name/huggingface_checkpoints"

for path in "$PARENT_DIR"/*/; do
    echo "Processing folder: $path"

    # Get base directory name (e.g., checkpoint-1000)
    dir_name=$(basename "$path")
    
    # Escape equal signs if any (just in case)
    escaped_name="${dir_name//=/\_}"
    
    # Only rename if the name changed
    if [[ "$dir_name" != "$escaped_name" ]]; then
        mv "$path" "$PARENT_DIR/$escaped_name"
        path="$PARENT_DIR/$escaped_name"
    fi

    accelerate launch --multi_gpu --num_processes=4 -m \
        lighteval accelerate \
        --model_args "pretrained=$path,dtype="bfloat16"" \
        --tasks "task_to_evaluate.txt" \
        --output-dir "$OpenLLM_OUTPUT/ablations/evaluation/language_ablations"
done
