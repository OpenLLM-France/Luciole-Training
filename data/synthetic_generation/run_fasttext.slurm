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

module purge
module load anaconda-py3/2024.06
conda activate distilabel-env

export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface


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

# Ita
input_path=$OpenLLM_OUTPUT/data/synthetic_data/fineweb_edu_annotation/Qwen3-32B_content_edu_spa_Latn_400k_2025-05-21T10-04-52.498525/default
output_path=$OpenLLM_OUTPUT/fasttext_classifiers/fineweb_edu_annotation/Qwen3-32B_content_edu_spa_Latn
train_all_labels $input_path $output_path