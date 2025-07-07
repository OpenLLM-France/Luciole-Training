#!/bin/bash
#SBATCH --job-name=batch_examples
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

module purge
module load arch/h100 nemo/2.3.1
export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface

folder_path=$1
seq_length=$2

python dataloader.py $folder_path --number_of_data 100 --seq_length $seq_length