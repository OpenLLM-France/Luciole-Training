
# Evaluate

##  Install

You should create a new environment for evaluation and clone our fork of lighteval.
```bash
module purge
module load anaconda-py3/2024.06
conda create -n eval-env python=3.10
conda activate eval-env

git clone git@github.com:OpenLLM-France/lighteval.git
cd lighteval/
pip install -e .[multilingual,vllm]
pip install seaborn
pip install python-slugify
```

## Run Evaluations

### Define the tasks you want to run: 
- you can create a new .txt file in `tasks/` folder
- or use one of the predefined (`tasks/en.txt`, `tasks/fr.txt`). 

### Load benchmarks in the cache:

Evaluations are run on h100 partition, so you have to prepare the cache first.
You can run this on prepost partition for example:

```bash
module purge
module load arch/h100
module load anaconda-py3/2024.06
conda activate eval-env

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface

lighteval accelerate "model_name=Qwen/Qwen3-0.6B" "tasks/en.txt"
lighteval accelerate "model_name=Qwen/Qwen3-0.6B" "tasks/gsm8k.txt"
lighteval accelerate "model_name=Qwen/Qwen3-0.6B" "tasks/fr.txt" --custom-tasks lighteval.tasks.multilingual.tasks
lighteval accelerate "model_name=Qwen/Qwen3-0.6B" "tasks/multilingual.txt" --custom-tasks lighteval.tasks.multilingual.tasks
```

Don't forget to load qwen model in the cache: `hf download Qwen/Qwen3-0.6B`

### Evaluate all the checkpoints of your experiment:

```bash
python evaluate_experiment.py $experiment_path $task_to_evaluate --custom_tasks multilingual ...
```

where:
- `$experiment_path` is the path to your experiments. It should have a `"huggingface_checkpoints"` folder in it. 
- `$task_to_evaluate` is the name of your .txt file (with the extension)
- add `--custom_tasks multilingual` if you need to evaluate multilingual tasks. It will activate lighteval args: `--custom-tasks lighteval.tasks.multilingual.tasks`

You can also use the bash script `run_all_tasks.sh` to run all the benchmarks of an experiment.

For example if you want to evaluate the nemotron 23b checkpoints:
```bash 
bash run_all_tasks.sh /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/pretrain/luciole_serie/luciolr_nemotron23b_phase1/ --multiple_of 5000 --gpus 2
```

### Evaluation of HF models

First you can download models with the bash script `preload_hf_model.sh` (or complete it to add new models).
It will load the model and some of their checkpoints (if available).

Then evaluate them with the same python script `evaluate_experiment.py` by using the argument `--hf_model` to specify the name of the hf repo. The evaluations will be saved in `$experiment_path`.

## Plotting the results...

You can use the script `plot_results.py` to plot your results.

For example, if you want to plot the evaluation of:

- nemotron 1b:

```bash
models="
$OpenLLM_OUTPUT/pretrain/luciole_serie/luciole_nemotron1b
$OpenLLM_OUTPUT/pretrain/luciole_serie/luciole_nemotron1b_phase2
$OpenLLM_OUTPUT/pretrain/compared_models/Lucie-7B
$OpenLLM_OUTPUT/pretrain/compared_models/OLMo-2-0425-1B
$OpenLLM_OUTPUT/pretrain/compared_models/CroissantLLMBase
$OpenLLM_OUTPUT/pretrain/compared_models/EuroLLM-1.7B
"

python plot_results.py $models --group all --output_path $OpenLLM_OUTPUT/pretrain/luciole_serie/luciole_nemotron1b_phase2/figs --flops 
python plot_results.py $models --group all --output_path $OpenLLM_OUTPUT/pretrain/luciole_serie/luciole_nemotron1b_phase2/figs --flops --max_samples 1000
```

- nemotron-h 8b:

```bash
models="
$OpenLLM_OUTPUT/pretrain/luciole_serie/luciole_nemotronh8b_phase1
$OpenLLM_OUTPUT/pretrain/luciole_serie/luciole_nemotron1b
$OpenLLM_OUTPUT/pretrain/luciole_serie/luciole_nemotron1b_phase2
$OpenLLM_OUTPUT/pretrain/compared_models/Lucie-7B
$OpenLLM_OUTPUT/pretrain/compared_models/CroissantLLMBase
"

python plot_results.py $models --group all --output_path $OpenLLM_OUTPUT/pretrain/luciole_serie/luciole_nemotronh8b_phase1/figs --flops 
python plot_results.py $models --group all --output_path $OpenLLM_OUTPUT/pretrain/luciole_serie/luciole_nemotronh8b_phase1/figs --flops --max_samples 1000
```

- nemotron 23b: