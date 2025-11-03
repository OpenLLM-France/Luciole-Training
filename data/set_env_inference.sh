#!/bin/bash

export PYTHONPATH=$(pwd)/processing:$PYTHONPATH

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
module purge
module load arch/h100 
module load anaconda-py3/2024.06
module load cuda/12.4.1
conda activate datatrove-env

export HF_HUB_OFFLINE=1