# 1B model

## Results

### Aggregated benchmarks
![Aggregated benchmarks](figs/agg_xlog_flops.png)

### English benchmarks
![English benchmarks](figs/en_xlog_flops.png)

### French benchmarks
![French benchmarks](figs/fr_xlog_flops.png)

### Multilingual benchmarks
![French benchmarks](figs/multilingual_xlog_flops.png)

## Baselines

Evaluate
```bash
cd evaluation/

module load anaconda-py3/2024.06
conda activate eval-env

# EuroLLM
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/EuroLLM-1.7B --hf_model utter-project/EuroLLM-1.7B tasks/en.txt 
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/EuroLLM-1.7B --hf_model utter-project/EuroLLM-1.7B tasks/fr.txt --custom_tasks multilingual
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/EuroLLM-1.7B --hf_model utter-project/EuroLLM-1.7B tasks/multilingual.txt --custom_tasks multilingual

# SmolLM2
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/SmolLM2-1.7B --hf_model HuggingFaceTB/SmolLM2-1.7B tasks/en.txt 
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/SmolLM2-1.7B --hf_model HuggingFaceTB/SmolLM2-1.7B tasks/fr.txt --custom_tasks multilingual
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/SmolLM2-1.7B --hf_model HuggingFaceTB/SmolLM2-1.7B tasks/multilingual.txt --custom_tasks multilingual

# SmolLM2
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/SmolLM3-3B --hf_model HuggingFaceTB/SmolLM3-3B tasks/en.txt 
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/SmolLM3-3B --hf_model HuggingFaceTB/SmolLM3-3B tasks/fr.txt --custom_tasks multilingual
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/SmolLM3-3B --hf_model HuggingFaceTB/SmolLM3-3B tasks/multilingual.txt --custom_tasks multilingual

# OLMO2 - 1B
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/OLMo-2-0425-1B --hf_model allenai/OLMo-2-0425-1B tasks/en.txt 
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/OLMo-2-0425-1B --hf_model allenai/OLMo-2-0425-1B tasks/multilingual.txt --custom_tasks multilingual
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/OLMo-2-0425-1B --hf_model allenai/OLMo-2-0425-1B tasks/fr.txt --custom_tasks multilingual

# OLMO2 - 7B
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/OLMo-2-1124-7B --hf_model allenai/OLMo-2-1124-7B tasks/en.txt 
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/OLMo-2-1124-7B --hf_model allenai/OLMo-2-1124-7B tasks/multilingual.txt --custom_tasks multilingual
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/OLMo-2-1124-7B --hf_model allenai/OLMo-2-1124-7B tasks/fr.txt --custom_tasks multilingual

# LUCIE
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/Lucie-7B --hf_model OpenLLM-France/Lucie-7B tasks/en.txt 
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/Lucie-7B --hf_model OpenLLM-France/Lucie-7B tasks/multilingual.txt --custom_tasks multilingual
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/Lucie-7B --hf_model OpenLLM-France/Lucie-7B tasks/fr.txt --custom_tasks multilingual
```

## Ablation

Train
```bash
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix ablation01_luciole --mode phase1 --num_nodes 16 --arch llama1b --email ogouvert@linagora.com --config ../../data/tokenization/run/chronicles/ablation_1/phase1/datamix.json 

python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix ablation02_luciole --mode phase1 --num_nodes 16 --arch llama1b --email ogouvert@linagora.com --config ../../data/tokenization/run/chronicles/ablation_2/datamix.json 
```

Convert
```bash
cd conversion/
sbatch convert.slurm $OpenLLM_OUTPUT/pretrain/ablation01_luciole_llama1b --no_completion
sbatch convert.slurm $OpenLLM_OUTPUT/pretrain/ablation02_luciole_llama1b --no_completion
```

Evaluate
```bash
cd evaluation/

module load anaconda-py3/2024.06
conda activate eval-env

python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/ablation01_luciole_llama1b tasks/en.txt
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/ablation01_luciole_llama1b tasks/fr.txt --custom_tasks multilingual

python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/ablation02_luciole_llama1b tasks/en.txt
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/ablation02_luciole_llama1b tasks/fr.txt --custom_tasks multilingual
```

Plot results
```bash
cd evaluation/

module load anaconda-py3/2024.06
conda activate eval-env

models="$OpenLLM_OUTPUT/pretrain/ablation01_luciole_llama1b $OpenLLM_OUTPUT/pretrain/ablation02_luciole_llama1b"

python plot_results.py $models --group fr --output_path $OpenLLM_OUTPUT/pretrain/figs --seq_length 4096 --xlog 
python plot_results.py $models --group en --output_path $OpenLLM_OUTPUT/pretrain/figs --seq_length 4096 --xlog 

```

## Phase 1

[Repeeat](../../../data/tokenization/run/chronicles/phase_1/repeats.csv)
[Datamix](../../../data/tokenization/run/chronicles/phase_1/datamix.json)

```bash
cd train/
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix luciole --mode phase1 --num_nodes 128 --arch llama1b --email ogouvert@linagora.com
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix luciole --mode phase1 --time 16:00:00 --num_nodes 128 --arch llama1b --email ogouvert@linagora.com
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix luciole --mode phase1 --time 12:00:00 --num_nodes 128 --arch llama1b --email ogouvert@linagora.com
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix luciole --mode phase1 --time 48:00:00 --num_nodes 64 --arch llama1b --email ogouvert@linagora.com
```

Convert
```bash
cd conversion/

sbatch convert.slurm $OpenLLM_OUTPUT/pretrain/luciole_llama1b --no_completion
python rope_scaling_correction.py 
```

Evaluate
```bash
cd evaluation/

module load anaconda-py3/2024.06
conda activate eval-env

python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/luciole_llama1b tasks/en.txt
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/luciole_llama1b tasks/recommended_set.txt 
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/luciole_llama1b tasks/multilingual.txt --custom-tasks lighteval.tasks.multilingual.tasks
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/luciole_llama1b tasks/fr.txt --custom_tasks multilingual
```

Plot results
```bash
cd evaluation/

module load anaconda-py3/2024.06
conda activate eval-env

models="$OpenLLM_OUTPUT/pretrain/luciole_llama1b $OpenLLM_OUTPUT/pretrain/OLMo-2-0425-1B $OpenLLM_OUTPUT/pretrain/EuroLLM-1.7B $OpenLLM_OUTPUT/pretrain/SmolLM2-1.7B $OpenLLM_OUTPUT/pretrain/SmolLM3-3B $OpenLLM_OUTPUT/pretrain/Lucie-7B"

python plot_results.py $models --group multilingual --output_path $OpenLLM_OUTPUT/pretrain/figs --seq_length 4096 --xlog --flops
python plot_results.py $models --group fr --output_path $OpenLLM_OUTPUT/pretrain/figs --seq_length 4096 --xlog --flops
python plot_results.py $models --group en --output_path $OpenLLM_OUTPUT/pretrain/figs --seq_length 4096 --xlog --flops
python plot_results.py $models --group agg --output_path $OpenLLM_OUTPUT/pretrain/figs --seq_length 4096 --xlog --flops
```

## Phase 2

```bash
cd train/
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix luciole --mode phase2 --num_nodes 128 --arch llama1b --email ogouvert@linagora.com
```

## Annealing

```bash
cd train/
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix luciole --mode annealing --num_nodes 128 --arch llama1b --config datamix.json --email ogouvert@linagora.com
```

# 7B model

# 20 B model