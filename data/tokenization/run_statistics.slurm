#!/bin/bash
#SBATCH --output=slurm/%x.out 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00 
#SBATCH --hint=nomultithread 
#SBATCH --qos=qos_cpu-dev
#SBATCH --account=qgz@cpu
#SBATCH --partition=prepost

# Requires nemo installation - on Jeanzay for now
module purge
module load arch/h100 nemo/2.1.0

data_path=$OpenLLM_OUTPUT/data/tokenized_data/$1

python generate_token_stats.py --data_path $data_path $2
python plot_doc_length_distribution.py --data_path $data_path $2 