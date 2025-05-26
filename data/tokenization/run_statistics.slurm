#!/bin/bash
#SBATCH --output=slurm/%x.out 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00 
#SBATCH --hint=nomultithread 
#SBATCH --qos=qos_cpu-dev
#SBATCH --account=qgz@cpu

module purge
module nemo/2.2.1

data_path=$OpenLLM_OUTPUT/data/$1

python generate_token_stats.py --data_path $data_path
python plot_doc_length_distribution.py --data_path $data_path