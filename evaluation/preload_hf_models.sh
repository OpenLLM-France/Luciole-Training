#!/bin/bash

set -e

module purge
module load arch/h100
module load anaconda-py3/2024.06
conda activate eval-env

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface

# instruct models
hf download mistralai/Ministral-3-3B-Reasoning-2512
hf download mistralai/Ministral-3-8B-Reasoning-2512
hf download allenai/Olmo-3-7B-Instruct-DPO
hf download allenai/Olmo-3-7B-Think-DPO
hf download almanach/Gaperon-1125-1B-SFT
hf download almanach/Gaperon-1125-8B-SFT
hf download almanach/Gaperon-1125-24B-SFT
hf download BSC-LT/ALIA-40b
hf download BSC-LT/salamandra-2b-instruct
hf download BSC-LT/salamandra-7b-instruct
hf download croissantllm/CroissantLLMChat-v0.1
hf download meta-llama/Llama-3.1-8B-Instruct
hf download meta-llama/Llama-3.2-1B-Instruct
# hf download HuggingFaceTB/SmolLM2-1.7B
hf download HuggingFaceTB/SmolLM3-3B
hf download mistralai/Ministral-3-8B-Instruct-2512
hf download mistralai/Ministral-3-3B-Instruct-2512-BF16
hf download mistralai/Ministral-3-8B-Instruct-2512-BF16
hf download mistralai/Mistral-Small-24B-Instruct-2501
hf download mistralai/Mistral-Small-3.2-24B-Instruct-2506
hf download OpenLLM-France/Lucie-7B-Instruct-v1.1
hf download Qwen/Qwen3-1.7B
hf download Qwen/Qwen3-8B
hf download swiss-ai/Apertus-8B-Instruct-2509
hf download utter-project/EuroLLM-1.7B-Instruct
hf download utter-project/EuroLLM-22B-Instruct-2512
hf download utter-project/EuroLLM-9B-Instruct


# pretrained models
hf download almanach/Gaperon-1125-1B
hf download almanach/Gaperon-1125-8B
hf download almanach/Gaperon-1125-24B
hf download BSC-LT/salamandra-2b
hf download BSC-LT/salamandra-7b
hf download croissantllm/CroissantLLMBase
hf download google/gemma-2-9b
hf download google/gemma-3-1b-pt
hf download google/gemma-7b
hf download HuggingFaceTB/SmolLM3-3B-Base
hf download meta-llama/Llama-2-7b-hf
hf download meta-llama/Llama-3.1-70B
hf download meta-llama/Llama-3.1-8B
hf download meta-llama/Llama-3.2-1B
hf download mistralai/Ministral-3-3B-Base-2512
hf download mistralai/Ministral-3-8B-Base-2512
hf download mistralai/Mistral-7B-v0.1
hf download mistralai/Mistral-7B-v0.3
hf download mistralai/Mistral-Small-24B-Base-2501
hf download mistralai/Mistral-Small-3.1-24B-Base-2503
hf download openGPT-X/Teuken-7B-base-v0.6
hf download OpenLLM-France/Lucie-7B
hf download OpenLLM-France/Luciole-1B-Base
hf download OpenLLM-France/Luciole-23B-Base
hf download OpenLLM-France/Luciole-8B-Base
hf download Qwen/Qwen-14B
hf download Qwen/Qwen2-1.5B
hf download Qwen/Qwen2.5-72B
hf download Qwen/Qwen2.5-7B
hf download Qwen/Qwen2-7B
hf download Qwen/Qwen3-14B-Base
hf download Qwen/Qwen3-1.7B
hf download Qwen/Qwen3-1.7B-Base
hf download Qwen/Qwen3-8B-Base
hf download swiss-ai/Apertus-70B-2509
hf download swiss-ai/Apertus-8B-2509
hf download utter-project/EuroLLM-1.7B
hf download utter-project/EuroLLM-9B
hf download utter-project/EuroLLM-22B-2512

# Apertus

for i in {1..20}; do
    step=$((i*50000))
    tokens=$(python3 -c "import math; print(math.ceil($i * 210))")
    revision="step${step}-tokens${tokens}B"
    echo -e "\n******\nLoading $revision\n"
    hf download swiss-ai/Apertus-8B-2509 --revision "$revision"
done
for i in {0..2}; do
    step=$((i*238000 + 1194000))
    tokens=$((i*1000 + 5014))
    revision="step${step}-tokens${tokens}B"
    echo -e "\n******\nLoading $revision\n"
    hf download swiss-ai/Apertus-8B-2509 --revision "$revision"
done
for i in {0..8}; do
    step=$((i*100000 + 1800000))
    tokens=$((i*840 + 8072))
    revision="step${step}-tokens${tokens}B"
    echo -e "\n******\nLoading $revision\n"
    hf download swiss-ai/Apertus-8B-2509 --revision "$revision"
done

# OLMO2

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

for i in {1..19}; do
    step=$((i*25000))
    tokens=$(python3 -c "import math; print(math.ceil($i * 209.72))")
    revision="stage1-step${step}-tokens${tokens}B"
    echo -e "\n******\nLoading $revision\n"
    hf download allenai/OLMo-2-1124-13B --revision "$revision"
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


# Count parameters
python count_parameters.py allenai/OLMo-2-0425-1B utter-project/EuroLLM-1.7B HuggingFaceTB/SmolLM2-1.7B HuggingFaceTB/SmolLM3-3B