#!/bin/bash

python dump_subsampled_parquets.py \
--output_path $OpenLLM_OUTPUT/data/raw_data/data_for_tokenization \
--fasttext_path $OpenLLM_OUTPUT/fasttext_classifiers/fineweb_edu_annotation/Qwen3-32B_content_edu_fra_Latn/model/educational_score_ngram2_epoch5_lr0.1.bin \
--slurm \
--target 1e9 \
--language fra_Latn

python dump_subsampled_parquets.py \
--output_path $OpenLLM_OUTPUT/data/raw_data/data_for_tokenization \
--fasttext_path $OpenLLM_OUTPUT/fasttext_classifiers/fineweb_edu_annotation/Qwen3-32B_content_edu_ara_Arab/model/educational_score_ngram2_epoch5_lr0.1.bin \
--slurm \
--target 1e9 \
--language arb_Arab

python dump_subsampled_parquets.py \
--output_path $OpenLLM_OUTPUT/data/raw_data/data_for_tokenization \
--fasttext_path $OpenLLM_OUTPUT/fasttext_classifiers/fineweb_edu_annotation/Qwen3-32B_content_edu_deu_Latn/model/educational_score_ngram2_epoch5_lr0.1.bin \
--slurm \
--target 2e8 \
--language deu_Latn

python dump_subsampled_parquets.py \
--output_path $OpenLLM_OUTPUT/data/raw_data/data_for_tokenization \
--fasttext_path $OpenLLM_OUTPUT/fasttext_classifiers/fineweb_edu_annotation/Qwen3-32B_content_edu_ita_Latn/model/educational_score_ngram2_epoch5_lr0.1.bin \
--slurm \
--target 2e8 \
--language ita_Latn

python dump_subsampled_parquets.py \
--output_path $OpenLLM_OUTPUT/data/raw_data/data_for_tokenization \
--fasttext_path $OpenLLM_OUTPUT/fasttext_classifiers/fineweb_edu_annotation/Qwen3-32B_content_edu_nld_Latn/model/educational_score_ngram2_epoch5_lr0.1.bin \
--slurm \
--target 2e8 \
--language nld_Latn

python dump_subsampled_parquets.py \
--output_path $OpenLLM_OUTPUT/data/raw_data/data_for_tokenization \
--fasttext_path $OpenLLM_OUTPUT/fasttext_classifiers/fineweb_edu_annotation/Qwen3-32B_content_edu_por_Latn/model/educational_score_ngram2_epoch5_lr0.1.bin \
--slurm \
--target 2e8 \
--language por_Latn

python dump_subsampled_parquets.py \
--output_path $OpenLLM_OUTPUT/data/raw_data/data_for_tokenization \
--fasttext_path $OpenLLM_OUTPUT/fasttext_classifiers/fineweb_edu_annotation/Qwen3-32B_content_edu_spa_Latn/model/educational_score_ngram2_epoch5_lr0.1.bin \
--slurm \
--target 2e8 \
--language spa_Latn