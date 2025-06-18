#!/bin/bash
#SBATCH --job-name=generate 
#SBATCH --output=slurm/log_%x-%j.out 
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64  
#SBATCH --time=15:00:00 
#SBATCH --hint=nomultithread 
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

module load arch/h100 
module load anaconda-py3/2024.06
module load cuda/12.4.1
conda activate distilabel-env

language=$1
model_name=$DSDIR/HuggingFace_Models/Qwen/Qwen3-32B
output_dir=$OpenLLM_OUTPUT/data/synthetic_data/fineweb_edu_annotation

VLLM_USE_V1=0 python generate.py --model_name $model_name --data_language $language --output_dir $output_dir --prompt multi_v2 --disable_thinking --vllm --gpus 4 --nsamples 400000 --temperature 0.6 --use_cache