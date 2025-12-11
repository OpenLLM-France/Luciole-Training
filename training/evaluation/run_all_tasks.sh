#!/bin/bash

python evaluate_experiment.py "$@" tasks/gsm8k.txt --command vllm
python evaluate_experiment.py "$@" tasks/en.txt --command vllm
python evaluate_experiment.py "$@" tasks/fr.txt --command vllm --custom_tasks multilingual --max_samples 1000
python evaluate_experiment.py "$@" tasks/multilingual.txt --command vllm --custom_tasks multilingual --max_samples 1000

# python evaluate_experiment.py "$@" tasks/math.txt --command vllm
