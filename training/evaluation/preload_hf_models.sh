#!/bin/bash

set -e

module purge
module load arch/h100
module load anaconda-py3/2024.06
conda activate eval-env

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface

# OLMO2

for i in {1..19}; do
    step=$((i*25000))
    tokens=$(python3 -c "import math; print(math.ceil($i * 209.72))")
    revision="stage1-step${step}-tokens${tokens}B"
    echo -e "\n******\nLoading $revision\n"
    hf download allenai/OLMo-2-1124-13B --revision "$revision"
done


exit 0

for i in {1..19}; do
    step=$((i*100000))
    tokens=$(python3 -c "import math; print(math.ceil($i * 209.72))")
    revision="stage1-step${step}-tokens${tokens}B"
    echo -e "\n******\nLoading $revision\n"
    hf download allenai/OLMo-2-0425-1B --revision "$revision"
done

for i in {1..18}; do
    step=$((i*50000))
    if [ $i -eq 2 ]; then
        continue # One step is missing
    fi
    tokens=$(python3 -c "import math; print(math.ceil($i * 209.72))")
    revision="stage1-step${step}-tokens${tokens}B"
    echo -e "\n******\nLoading $revision\n"
    hf download allenai/OLMo-2-1124-7B --revision "$revision"
done

for i in {1..18}; do
    step=$((i*25000))
    if [ $i -eq 14 ]; then
        continue # One step is missing
    fi
    tokens=$(python3 -c "import math; print(math.ceil($i * 209.72))")
    revision="stage1-step${step}-tokens${tokens}B"
    echo -e "\n******\nLoading $revision\n"
    hf download allenai/OLMo-2-0325-32B --revision "$revision"
done

# Apertus
for i in {1..20}; do
    step=$((i*50000))
    tokens=$(python3 -c "import math; print(math.ceil($i * 210))")
    revision="step${step}-tokens${tokens}B"
    echo -e "\n******\nLoading $revision\n"
    hf download swiss-ai/Apertus-8B-2509 --revision "$revision"
done

# Lucie
for i in {1..15}; do
    step=$(python3 -c "print(f'{$i*50000:07d}')")
    echo -e "\n******\nLoading $step\n"
    hf download OpenLLM-France/Lucie-7B --revision "step$step"
done

# SmolLM2
for i in {1..15}; do
    step=$((i*250000))
    echo -e "\n******\nLoading $step\n"
    hf download HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints --revision "step-$step"
done

# EUROLLM
hf download utter-project/EuroLLM-1.7B
hf download utter-project/EuroLLM-9B
hf download HuggingFaceTB/SmolLM2-1.7B
hf download HuggingFaceTB/SmolLM3-3B
hf download croissantllm/CroissantLLMBase
hf download BSC-LT/salamandra-7b
hf download openGPT-X/Teuken-7B-base-v0.6
hf download almanach/Gaperon-1125-1B
hf download almanach/Gaperon-1125-8B
hf download almanach/Gaperon-1125-24B

# Count parameters
python count_parameters.py allenai/OLMo-2-0425-1B utter-project/EuroLLM-1.7B HuggingFaceTB/SmolLM2-1.7B HuggingFaceTB/SmolLM3-3B