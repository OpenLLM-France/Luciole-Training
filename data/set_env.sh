#!/bin/bash

# Get machine hostname
HOSTNAME=$(hostname)

# Set DATA path based on machine
if [[ "$HOSTNAME" == "jean-zay"* ]]; then
    export DATA="$ALL_CCFRSCRATCH/datasets/training"
    export HF_HOME=$ALL_CCFRSCRATCH/.cache/huggingface

    module purge
    module load anaconda-py3/2023.09 
elif [[ "$HOSTNAME" == "koios" ]]; then
    export DATA="/media/storage0/ogouvert/datasets/training"
else
    echo "Unknown machine: $HOSTNAME"
fi

conda activate datatrove-env