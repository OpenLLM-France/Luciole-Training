#!/bin/bash

set -e

module purge
module load arch/h100
module load anaconda-py3/2024.06
conda activate eval-env

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface

for i in {1..20}; do
    step=$((i*100000))
    tokens=$(python3 -c "import math; print(math.ceil($i * 209.73))")
    revision="stage1-step${step}-tokens${tokens}B"
    echo -e "\n******\nLoading $revision\n"
    hf download allenai/OLMo-2-0425-1B --revision "$revision"
done