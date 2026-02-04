#!/bin/bash
#SBATCH --job-name=convert
#SBATCH --output=log_%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
export HF_HUB_OFFLINE=1

module purge
module load arch/h100 nemo/2.4.0

hf_model=$OpenLLM_OUTPUT/pretrain/luciole_serie/luciole_nemotron23b_phase2_noised/huggingface_checkpoints/luciole_nemotron23b_phase2_noised-step_0358926
output_path=$OpenLLM_OUTPUT/pretrain/luciole_serie/luciole_nemotron23b_phase2_noised/nemo_conversion/luciole_nemotron23b_phase2_noised-step_0358926

torchrun --nproc_per_node=1  hf_to_nemo.py --hf_model hf://$hf_model --output_path $output_path --arch nemotron