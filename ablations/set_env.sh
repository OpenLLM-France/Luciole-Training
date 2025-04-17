#!/bin/bash

export OpenLLM_OUTPUT=$ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$ALL_CCFRSCRATCH/.cache/huggingface

export NVIDIA_PYTORCH_VERSION=24.02

module purge
module load arch/h100 nemo/2.1.0