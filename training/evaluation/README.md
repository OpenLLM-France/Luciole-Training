
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
python evaluate_experiment.py $experiment_path $task_to_evaluate --custom_tasks multilingual
```

where:
- `$experiment_path` is the path to your experiments. It should have a `"huggingface_checkpoints"` folder in it. 
- `$task_to_evaluate` is the name of your .txt file (with the extension)
- add `--custom_tasks multilingual` if you need to evaluate multilingual tasks. It will activate lighteval args: `--custom-tasks lighteval.tasks.multilingual.tasks`

### Plotting the results...

You can use the script `plot_results.py` to plot your results.
