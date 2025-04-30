#!/bin/bash

main_path=$OpenLLM_OUTPUT/ablations/train/lucie2_ablations

python postprocess_results.py \
    --group fr \
    --experiment_path $main_path/multi_base_4n_20b $main_path/multi_gallica_4n_20b $main_path/olmo2_4n_20b \
    --output_path $OpenLLM_OUTPUT/ablations/evaluation/test