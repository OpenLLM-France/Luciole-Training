#!/bin/bash

### Language ablation example
main_path=$OpenLLM_OUTPUT/ablations/train/language_ablations

python postprocess_results.py \
    --group fr en \
    --experiment_path $main_path/datamix_1._eng_Latn_4n_20b \
    $main_path/datamix_1._fra_Latn_4n_20b \
    $main_path/datamix_.25_fra_Latn_.75_eng_Latn_4n_20b \
    $main_path/datamix_.5_fra_Latn_.5_eng_Latn_4n_20b \
    $main_path/datamix_.75_fra_Latn_.25_eng_Latn_4n_20b \
    $main_path/datamix_.95_fra_Latn_.05_eng_Latn_4n_20b \
    --output_path $OpenLLM_OUTPUT/ablations/evaluation/language_ablations

### Lucie2 ablation
main_path=$OpenLLM_OUTPUT/ablations/train/lucie2_ablations

python postprocess_results.py \
    --group fr en \
    --experiment_path $main_path/multi_base_4n_20b $main_path/multi_gallica_4n_20b $main_path/olmo2_4n_20b \
    --output_path $OpenLLM_OUTPUT/ablations/evaluation/lucie2_ablations