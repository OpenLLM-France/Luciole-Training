#!/bin/bash

export OpenLLM_OUTPUT=$ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$ALL_CCFRSCRATCH/.cache/huggingface

module purge
module load arch/h100 nemo/2.1.0