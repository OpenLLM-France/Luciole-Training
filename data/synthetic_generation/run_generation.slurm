#!/bin/bash
#SBATCH --job-name=generate 
#SBATCH --output=log_%x-%j.out 
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64  
#SBATCH --time=20:00:00 
#SBATCH --hint=nomultithread 
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

module load arch/h100 
module load anaconda-py3/2024.06
module load cuda/12.4.1
conda activate distilabel-test

export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface

model_name=$DSDIR/HuggingFace_Models/Qwen/Qwen3-32B

VLLM_USE_V1=0 python generate.py --model_name $model_name --data_language fra_Latn --prompt multi_task --disable_thinking --vllm --gpus 4 --nsamples 400000 