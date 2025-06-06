#!/bin/bash
#SBATCH --output=slurm/%x.out 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00 
#SBATCH --hint=nomultithread 
#SBATCH --qos=qos_cpu-dev
#SBATCH --account=qgz@cpu
#SBATCH --partition=cpu_p1

# Requires nemo installation - on Jeanzay for now
module purge
module load arch/a100 nemo/2.2.1
export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface

data_path=$1
extra_args=$2

python generate_token_stats.py --data_path $data_path $extra_args
python plot_doc_length_distribution.py --data_path $data_path $extra_args 