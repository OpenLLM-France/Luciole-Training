#!/bin/bash

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# export OpenLLM_OUTPUT=$ALL_CCFRSCRATCH/OpenLLM-BPI-output
# export OpenLLM_DATA=$ALL_CCFRSCRATCH/preprocessed_data/Lucie/lucie_tokens_65k_grouped

module purge
module load arch/h100 nemo/2.1.0