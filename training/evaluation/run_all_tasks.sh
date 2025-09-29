#!/bin/bash
module load anaconda-py3/2024.06
conda activate eval-env

python evaluate_experiment.py "$@" tasks/en.txt 
python evaluate_experiment.py "$@" tasks/fr.txt --command vllm --custom_tasks multilingual 
python evaluate_experiment.py "$@" tasks/multilingual.txt --command vllm --custom_tasks multilingual