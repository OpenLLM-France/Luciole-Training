#!/bin/bash

# Set this to the base directory containing the experiments
dir=$1

# Loop over each subdirectory (each experiment path)
for expe_path in "$dir"/*; do
    if [ -d "$expe_path" ] && [ -d "$expe_path/huggingface_checkpoints" ]; then
        echo ">>> Evaluating $expe_path"
        python evaluate_experiment.py "$expe_path" tasks/the_pile.txt --command accelerate --max_samples 1000
    fi
done