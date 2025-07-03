#!/bin/bash
#SBATCH --job-name=fasttext 
#SBATCH --output=slurm/log_%x-%j.out 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40  
#SBATCH --time=01:00:00 
#SBATCH --hint=nomultithread 
#SBATCH --qos=qos_cpu-dev
#SBATCH --account=qgz@cpu

set -euo pipefail

module purge
module load anaconda-py3/2024.06
conda activate distilabel-env

# export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface


train_all_labels() {
  local input_path=$1
  local output_path=$2

  for label in educational_score is_toxic is_ad topic; do
    python train_fasttext.py \
      --input_path "$input_path" \
      --output_path "$output_path" \
      --label "$label"
  done
}

train_all_labels_from_parquet() {
  local input_path=$1
  local output_path=$2

  for label in educational_score is_toxic is_ad topic; do
    python train_fasttext.py \
      --input_path "$input_path" \
      --output_path "$output_path" \
      --label "$label" \
      --from_parquet
  done
}

# Fra
input_path=$OpenLLM_OUTPUT/data/synthetic_data/fineweb_edu_annotation/Qwen3-32B_content_edu_fra_Latn_400k_2025-05-21T10-04-59.003164/default
output_path=$OpenLLM_OUTPUT/fasttext_classifiers/fineweb_edu_annotation/Qwen3-32B_content_edu_fra_Latn
train_all_labels $input_path $output_path

# Deu
input_path=$OpenLLM_OUTPUT/data/synthetic_data/fineweb_edu_annotation/Qwen3-32B_content_edu_deu_Latn_400k_2025-05-21T10-04-59.003467/default
output_path=$OpenLLM_OUTPUT/fasttext_classifiers/fineweb_edu_annotation/Qwen3-32B_content_edu_deu_Latn
train_all_labels $input_path $output_path

# Ita
input_path=$OpenLLM_OUTPUT/data/synthetic_data/fineweb_edu_annotation/Qwen3-32B_content_edu_ita_Latn_400k_2025-05-21T10-05-32.610020/default
output_path=$OpenLLM_OUTPUT/fasttext_classifiers/fineweb_edu_annotation/Qwen3-32B_content_edu_ita_Latn
train_all_labels $input_path $output_path

# Spa
input_path=$OpenLLM_OUTPUT/data/synthetic_data/fineweb_edu_annotation/Qwen3-32B_content_edu_spa_Latn_400k_2025-05-21T10-04-52.498525/default
output_path=$OpenLLM_OUTPUT/fasttext_classifiers/fineweb_edu_annotation/Qwen3-32B_content_edu_spa_Latn
train_all_labels $input_path $output_path

# Arab
input_path=$OpenLLM_OUTPUT/data/synthetic_data/fineweb_edu_annotation/Qwen3-32B_content_edu_arb_Arab_t0.5_n400k/53bedcbb53ca1db3fed4f15c1fb88e4d8a6089c8/executions/9225776c9aa457fa8ede8b3581ce5944615b6a72/data/steps_outputs/text_generation_0
output_path=$OpenLLM_OUTPUT/fasttext_classifiers/fineweb_edu_annotation/Qwen3-32B_content_edu_ara_Arab
train_all_labels_from_parquet $input_path $output_path

# Por
input_path=$OpenLLM_OUTPUT/data/synthetic_data/fineweb_edu_annotation/Qwen3-32B_content_edu_por_Latn_t0.5_n400k/53bedcbb53ca1db3fed4f15c1fb88e4d8a6089c8/executions/9225776c9aa457fa8ede8b3581ce5944615b6a72/data/steps_outputs/text_generation_0
output_path=$OpenLLM_OUTPUT/fasttext_classifiers/fineweb_edu_annotation/Qwen3-32B_content_edu_por_Latn
train_all_labels_from_parquet $input_path $output_path

# Nld
input_path=$OpenLLM_OUTPUT/data/synthetic_data/fineweb_edu_annotation/Qwen3-32B_content_edu_nld_Latn_t0.5_n400k/53bedcbb53ca1db3fed4f15c1fb88e4d8a6089c8/executions/9225776c9aa457fa8ede8b3581ce5944615b6a72/data/steps_outputs/text_generation_0
output_path=$OpenLLM_OUTPUT/fasttext_classifiers/fineweb_edu_annotation/Qwen3-32B_content_edu_nld_Latn
train_all_labels_from_parquet $input_path $output_path
