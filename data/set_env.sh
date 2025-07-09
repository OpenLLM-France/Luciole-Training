#!/bin/bash

export PYTHONPATH=$(pwd)/processing:$PYTHONPATH

# Get machine hostname
HOSTNAME=$(hostname)

# Set DATA path based on machine

if [[ "$HOSTNAME" == "koios" ]]; then
    export OpenLLM_OUTPUT="/media/storage0/ogouvert/OpenLLM-BPI-output"
else
    export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
    export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
    module purge
    module load anaconda-py3/2024.06
fi

export DATA=$OpenLLM_OUTPUT/datasets

conda activate datatrove-env