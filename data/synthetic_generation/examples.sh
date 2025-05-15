#!/bin/bash

# - /lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3-32B
# - Qwen/Qwen3-8B
# - Qwen/Qwen3-1.7B
# - Qwen/Qwen3-0.6B

module load arch/h100 
module load anaconda-py3/2024.06
module load cuda/12.4.1
conda activate distilabel-env

export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
# huggingface-cli download Qwen/Qwen3-0.6B --repo-type model
# huggingface-cli download Qwen/Qwen3-1.7B --repo-type model
# huggingface-cli download Qwen/Qwen3-8B --repo-type model

model_name=Qwen/Qwen3-0.6B
model_name=Qwen/Qwen3-1.7B
model_name=Qwen/Qwen3-8B

python generate.py --model_name $model_name --prompt en --disable_thinking
python generate.py --model_name $model_name --prompt fr --disable_thinking