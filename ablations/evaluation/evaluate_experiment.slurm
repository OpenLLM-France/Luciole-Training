#!/bin/bash
#SBATCH --job-name=eval 
#SBATCH --output=%x.out 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  
#SBATCH --time=02:00:00 
#SBATCH --hint=nomultithread 
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

module purge
module load anaconda-py3/2024.06
conda activate eval-env

export OpenLLM_OUTPUT=$ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$ALL_CCFRSCRATCH/.cache/huggingface
export HF_HUB_OFFLINE=1

path="$OpenLLM_OUTPUT/ablations/train/language_ablations/datamix_.5_fra_Latn_.5_eng_Latn_4n_20b/huggingface_checkpoints/datamix_.5_fra_Latn_.5_eng_Latn_4n_20b--step=19071.0000-epoch=0-consumed_samples=9764864.0"

# Remove = signs in model name
escaped_path="${path//=/\_}"
mv $path $escaped_path

lighteval accelerate "pretrained=$escaped_path" "task_to_evaluate.txt" --output-dir "$OpenLLM_OUTPUT/ablations/evaluation/language_ablations"