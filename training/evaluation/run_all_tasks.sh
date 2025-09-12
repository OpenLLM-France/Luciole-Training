#!/bin/bash
module load anaconda-py3/2024.06
conda activate eval-env

shift
python evaluate_experiment.py "$@" tasks/en.txt
python evaluate_experiment.py "$@" tasks/math.txt
python evaluate_experiment.py "$@" tasks/math_5fewshot.txt
python evaluate_experiment.py "$@" tasks/fr.txt --custom_tasks multilingual
python evaluate_experiment.py "$@" tasks/multilingual.txt --custom_tasks multilingual
