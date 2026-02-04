#!/bin/bash

module purge
module load arch/h100
module load anaconda-py3/2024.06
conda activate eval-env

export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface

hf download Qwen/Qwen3-0.6B

for TASK in en gsm8k mixeval ifbench ifeval ifeval_fr gsm_plus aime live_code_bench gpqa gpqa-fr; do
    lighteval accelerate "model_name=Qwen/Qwen3-0.6B" tasks/${TASK}.txt
done

for TASK in fr multilingual reasoning ; do
    lighteval accelerate "model_name=Qwen/Qwen3-0.6B" tasks/${TASK}.txt --custom-tasks lighteval.tasks.multilingual.tasks
done

for TASK in mmlu_pro ; do
    lighteval accelerate "model_name=Qwen/Qwen3-0.6B" tasks/${TASK}.txt --custom-tasks smollm3
done

for TASK in ruler_4096 ruler_8192 ruler_16384 ruler_32768 ruler_65536 ruler_131072 ; do
    lighteval accelerate "model_name=Qwen/Qwen3-0.6B" tasks/${TASK}.txt --custom-tasks ruler
done