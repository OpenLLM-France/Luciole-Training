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

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
export HF_HUB_OFFLINE=1

path_to_experiment=$1 
task_to_evaluate="$(pwd)/$2"

if [ "$3" == "multilingual" ]; then
  extra_arg="--custom-tasks lighteval.tasks.multilingual.tasks"
else
  extra_arg=""
fi

PARENT_DIR="$path_to_experiment/huggingface_checkpoints"
cd $PARENT_DIR

output_dir="$path_to_experiment/evaluation"
mkdir -p "$output_dir"
echo "Output directory: $output_dir"

log_dir="$output_dir/slurm"
mkdir -p "$log_dir"
echo "Log directory: $log_dir"

for checkpoint_dir in *; do    
    # Escape equal signs if any (just in case)
    escaped_checkpoint_dir="${checkpoint_dir//=/\_}"
    
    # Only rename if the name changed
    if [[ "$checkpoint_dir" != "$escaped_checkpoint_dir" ]]; then
        mv "$checkpoint_dir" "$escaped_checkpoint_dir"
        checkpoint_dir=$escaped_checkpoint_dir
    fi

    echo "Processing checkpoint: $checkpoint_dir"

    accelerate launch --multi_gpu --num_processes=4 -m \
        lighteval accelerate \
        "pretrained=$checkpoint_dir,dtype="bfloat16"" \
        "$task_to_evaluate" \
        --output-dir $output_dir \
        $extra_arg \
        > "$log_dir/$checkpoint_dir.out" 2>&1
done
