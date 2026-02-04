#!/bin/bash
#SBATCH --output=slurm_logs/%x_%j.out 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00 
#SBATCH --hint=nomultithread 
#SBATCH --account=qgz@cpu
#SBATCH --partition=cpu_p1

module purge
module load arch/h100 nemo/2.3.1
export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface

python $1 $2 $3 $4