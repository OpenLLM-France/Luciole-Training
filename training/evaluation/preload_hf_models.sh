#!/bin/bash

set -e

module purge
module load arch/h100
module load anaconda-py3/2024.06
conda activate eval-env

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface

# OLMO2
for i in {1..20}; do
    step=$((i*100000))
    tokens=$(python3 -c "import math; print(math.ceil($i * 209.73))")
    revision="stage1-step${step}-tokens${tokens}B"
    echo -e "\n******\nLoading $revision\n"
    hf download allenai/OLMo-2-0425-1B --revision "$revision"
done

for i in {1..20}; do
    step=$((i*50000))
    tokens=$(python3 -c "import math; print(math.ceil($i * 209.767))")
    revision="stage1-step${step}-tokens${tokens}B"
    echo -e "\n******\nLoading $revision\n"
    hf download allenai/OLMo-2-1124-7B --revision "$revision"
done

# Lucie
for i in {1..15}; do
    step=$(python3 -c "print(f'{$i*50000:07d}')")
    echo -e "\n******\nLoading $step\n"
    hf download OpenLLM-France/Lucie-7B --revision "step$step"
done

# EUROLLM
hf download utter-project/EuroLLM-1.7B
hf download HuggingFaceTB/SmolLM2-1.7B
hf download HuggingFaceTB/SmolLM3-3B

# Count parameters
python count_parameters.py allenai/OLMo-2-0425-1B utter-project/EuroLLM-1.7B HuggingFaceTB/SmolLM2-1.7B HuggingFaceTB/SmolLM3-3B