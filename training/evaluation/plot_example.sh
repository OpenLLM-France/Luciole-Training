#!/bin/bash

### Language ablation example
main_path=$OpenLLM_OUTPUT/ablations_v0/train/language_ablations

python plot_results.py \
    --group fr en agg \
    --experiment_path $main_path/datamix_1._eng_Latn_4n_20b \
    $main_path/datamix_.25_fra_Latn_.75_eng_Latn_4n_20b \
    $main_path/datamix_.5_fra_Latn_.5_eng_Latn_4n_20b \
    $main_path/datamix_.75_fra_Latn_.25_eng_Latn_4n_20b \
    $main_path/datamix_.95_fra_Latn_.05_eng_Latn_4n_20b \
    $main_path/datamix_1._fra_Latn_4n_20b \
    --output_path $OpenLLM_OUTPUT/ablations_v0/evaluation/language_ablations --xlog 

### Rehydratation
main_path=$OpenLLM_OUTPUT/ablations_v0/train/language_ablations
other_path=$OpenLLM_OUTPUT/ablations_v0/train/fineweb2_filtering

python plot_results.py \
    --group fr en agg \
    --experiment_path \
    $main_path/datamix_1._fra_Latn_4n_20b \
    $main_path/datamix_1._fra_Latn_norehydratation_4n_20b \
    $other_path/llama1b_datamix_fr_upsampled_by_edu_score_8n_20b \
    $other_path/llama1b_datamix_fr_upsampled_by_edu_score_8n_35b \
    $other_path/llama1b_datamix_fineweb2_fra_Latn_rehydratation_v2_8n_20b \
    --output_path $OpenLLM_OUTPUT/ablations_v0/evaluation/rehydratation --xlog 

### Lucie2 ablation
main_path=$OpenLLM_OUTPUT/ablations_v0/train/lucie2_ablations

python plot_results.py \
    --group fr en agg \
    --experiment_path $main_path/multi_base_4n_20b $main_path/multi_gallica_4n_20b $main_path/datamix_multi_gallica_plus_4n_20b $main_path/llama_datamix_multi_gallica_full_4n_20b\
    --output_path $OpenLLM_OUTPUT/ablations_v0/evaluation/multi_gallica_ablations --xlog --fit