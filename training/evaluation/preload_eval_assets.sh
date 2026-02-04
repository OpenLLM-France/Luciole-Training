#!/bin/bash

set -e

module purge
module load arch/h100
module load anaconda-py3/2024.06
conda activate eval-env

export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface

python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger_eng')" # ifbench
python -c "from spacy.cli import download; download('en_core_web_sm')"  # ifbench

hf download flowaicom/Flow-Judge-v0.1

# lighteval/src/lighteval/tasks/extended/tiny_benchmarks/tinyBenchmarks.pkl
lighteval accelerate "pretrained=gpt2" "leaderboard|gsm8k|5" --max-samples 1
