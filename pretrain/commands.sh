#!/bin/bash

## Baselines

cd evaluation/
bash run_all_tasks.sh $OpenLLM_OUTPUT/pretrain/EuroLLM-1.7B --hf_model utter-project/EuroLLM-1.7B
bash run_all_tasks.sh $OpenLLM_OUTPUT/pretrain/SmolLM2-1.7B --hf_model HuggingFaceTB/SmolLM2-1.7B
bash run_all_tasks.sh $OpenLLM_OUTPUT/pretrain/SmolLM3-3B --hf_model HuggingFaceTB/SmolLM3-3B
bash run_all_tasks.sh $OpenLLM_OUTPUT/pretrain/OLMo-2-0425-1B --hf_model allenai/OLMo-2-0425-1B
bash run_all_tasks.sh $OpenLLM_OUTPUT/pretrain/OLMo-2-1124-7B --hf_model allenai/OLMo-2-1124-7B
bash run_all_tasks.sh $OpenLLM_OUTPUT/pretrain/Lucie-7B --hf_model OpenLLM-France/Lucie-7B

## Ablation

cd train/
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix ablation01_luciole --mode phase1 --num_nodes 16 --arch llama1b --email ogouvert@linagora.com --config ../../data/tokenization/chronicles/ablation_1/phase1/datamix.json 
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix ablation01-2_luciole --mode phase1 --num_nodes 16 --arch llama1b --email ogouvert@linagora.com --config ../../data/tokenization/chronicles/ablation_1.2/datamix.json 
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix ablation02_luciole --mode phase1 --num_nodes 16 --arch llama1b --email ogouvert@linagora.com --config ../../data/tokenization/chronicles/ablation_2/datamix.json 

cd conversion/
sbatch convert.slurm $OpenLLM_OUTPUT/pretrain/ablation01_luciole_llama1b --no_completion
sbatch convert.slurm $OpenLLM_OUTPUT/pretrain/ablation01-2_luciole_llama1b --no_completion
sbatch convert.slurm $OpenLLM_OUTPUT/pretrain/ablation02_luciole_llama1b --no_completion

cd evaluation/
bash run_all_tasks.sh $OpenLLM_OUTPUT/pretrain/ablation01_luciole_llama1b
bash run_all_tasks.sh $OpenLLM_OUTPUT/pretrain/ablation01-2_luciole_llama1b
bash run_all_tasks.sh $OpenLLM_OUTPUT/pretrain/ablation02_luciole_llama1b

cd evaluation/
models="
$OpenLLM_OUTPUT/pretrain/luciole_llama1b 
$OpenLLM_OUTPUT/pretrain/ablation01_luciole_llama1b 
$OpenLLM_OUTPUT/pretrain/ablation02_luciole_llama1b
"
groups="fr en multilingual math agg" 
python plot_results.py $models --group $groups --output_path $OpenLLM_OUTPUT/pretrain/figs --seq_length 4096 --xlog 

## Phase 1

cd train/
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix luciole --mode phase1 --num_nodes 128 --arch llama1b --email ogouvert@linagora.com

cd conversion/
sbatch convert.slurm $OpenLLM_OUTPUT/pretrain/luciole_llama1b --no_completion

cd evaluation/
bash run_all_tasks.sh $OpenLLM_OUTPUT/pretrain/luciole_llama1b

## Plot results
cd evaluation/
models="
$OpenLLM_OUTPUT/pretrain/luciole_llama1b
$OpenLLM_OUTPUT/pretrain/OLMo-2-0425-1B
$OpenLLM_OUTPUT/pretrain/EuroLLM-1.7B
$OpenLLM_OUTPUT/pretrain/SmolLM2-1.7B
$OpenLLM_OUTPUT/pretrain/SmolLM3-3B
$OpenLLM_OUTPUT/pretrain/Lucie-7B
"
groups="math math_5fewshot multilingual fr en agg"

python plot_results.py $models --group $groups --output_path $OpenLLM_OUTPUT/pretrain/figs --seq_length 4096 --xlog --flops
