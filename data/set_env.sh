#!/bin/bash

export PYTHONPATH=$(pwd)/processing:$PYTHONPATH

# Get machine hostname
HOSTNAME=$(hostname)

# Set DATA path based on machine

if [[ "$HOSTNAME" == "koios" ]]; then
    export OpenLLM_OUTPUT="/media/storage0/$USER/OpenLLM-BPI-output"
else
    export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
    export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
    module purge
    module load anaconda-py3/2024.06
fi

export DATA=$OpenLLM_OUTPUT/datasets

# SLURM accounts (override these for your project)
export SLURM_ACCOUNT_GPU=${SLURM_ACCOUNT_GPU:-"wuh@h100"}
export SLURM_ACCOUNT_CPU=${SLURM_ACCOUNT_CPU:-"qgz@cpu"}

conda activate datatrove-env