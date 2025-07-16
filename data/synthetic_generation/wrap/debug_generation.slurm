#!/bin/bash
#SBATCH --job-name=generate 
#SBATCH --output=slurm/log_%x-%j.out 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64  
#SBATCH --time=00:10:00 
#SBATCH --hint=nomultithread 
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

module load arch/h100 
module load anaconda-py3/2024.06
module load cuda/12.4.1
conda activate distilabel-env

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

data_path=$OpenLLM_OUTPUT/data/raw_data/full_datasets/fineweb2/fra_Latn/split_by_edu_score/data/edu_4
expe_name=TEST_edu4
output_dir=$OpenLLM_OUTPUT/data/synthetic_data/wrap/test

# model_name=/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3-8B
model_name=Qwen/Qwen3-1.7B


# QWEN 1.7B
# VLLM_USE_V1=0 python generate.py $data_path $expe_name --output_dir $output_dir --prompt_path prompt/qa_fr.txt --gpus 4 --nsamples 100 --temperature 0.5 --use_cache --disable_thinking
# VLLM_USE_V1=0 python generate.py $data_path $expe_name --output_dir $output_dir --prompt_path prompt/easy_fr.txt --gpus 4 --nsamples 100 --temperature 0.5 --use_cache --disable_thinking
# VLLM_USE_V1=0 python generate.py $data_path $expe_name --output_dir $output_dir --prompt_path prompt/medium_fr.txt --gpus 4 --nsamples 100 --temperature 0.5 --use_cache --disable_thinking
# VLLM_USE_V1=0 python generate.py $data_path $expe_name --output_dir $output_dir --prompt_path prompt/hard_fr.txt --gpus 4 --nsamples 100 --temperature 0.5 --use_cache --disable_thinking

# Fineweb-2
# VLLM_USE_V1=0 python generate.py $data_path $expe_name --output_dir $output_dir --prompt_path prompt/nemotron_cc/diverse_qa_pairs_fr2.txt --gpus 1 --nsamples 500 --temperature 0.5 --use_cache --disable_thinking # --seed 0
# VLLM_USE_V1=0 python generate.py $data_path $expe_name --output_dir $output_dir --prompt_path prompt/nemotron_cc/distill_fr.txt --max 2048 --gpus 1 --nsamples 500 --temperature 0.5 --use_cache --disable_thinking # --seed 0
# VLLM_USE_V1=0 python generate.py $data_path $expe_name --output_dir $output_dir --model_name $model_name --prompt_path prompt/nemotron_cc/extract_knowledge_fr.txt --max 2048 --gpus 1 --nsamples 500 --temperature 0.5 --use_cache --disable_thinking # --seed 0

# VLLM_USE_V1=0 python generate.py $data_path --expe_name $expe_name --output_dir $output_dir --model_name $model_name --prompt_path prompt/nemotron_cc/extract_knowledge_fr3.txt --max 2048 --gpus 1 --nsamples 50 --temperature 0.5 --use_cache --disable_thinking # --seed 0

VLLM_USE_V1=0 python generate.py $data_path --expe_name $expe_name --output_dir $output_dir --model_name $model_name --prompt_path prompt/wrap/medium_fr.txt --max 2048 --gpus 1 --nsamples 50 --temperature 0.5 --use_cache --disable_thinking # --seed 0

# QWEN 8B
# model_name=/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3-8B
# VLLM_USE_V1=0 python generate.py $data_path $expe_name --model_name $model_name --output_dir $output_dir --prompt_path prompt/qa_fr.txt --gpus 4 --nsamples 100 --temperature 0.5 --use_cache --disable_thinking
# VLLM_USE_V1=0 python generate.py $data_path $expe_name --model_name $model_name --output_dir $output_dir --prompt_path prompt/easy_fr.txt --gpus 4 --nsamples 100 --temperature 0.5 --use_cache --disable_thinking
# VLLM_USE_V1=0 python generate.py $data_path $expe_name --model_name $model_name --output_dir $output_dir --prompt_path prompt/medium_fr.txt --gpus 4 --nsamples 100 --temperature 0.5 --use_cache --disable_thinking
# VLLM_USE_V1=0 python generate.py $data_path $expe_name --model_name $model_name --output_dir $output_dir --prompt_path prompt/hard_fr.txt --gpus 4 --nsamples 100 --temperature 0.5 --use_cache --disable_thinking

