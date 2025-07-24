#!/bin/bash
#SBATCH --output=slurm/%x.out 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00 
#SBATCH --hint=nomultithread 
#SBATCH --qos=qos_cpu-t3
#SBATCH --account=qgz@cpu
#SBATCH --partition=cpu_p1

module purge
module load arch/a100 nemo/2.2.1
export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface

script=$1
shift
python $script "$@"