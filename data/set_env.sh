#!/bin/bash

module purge
module load anaconda-py3/2023.09 
conda activate datatrove-env

export HF_HOME=$ALL_CCFRSCRATCH/.cache/huggingface