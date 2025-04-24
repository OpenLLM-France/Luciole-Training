# Evaluation

All about evaluation

## Installation

```
module purge
module load anaconda-py3/2024.06
conda create -n eval-env python=3.10
conda activate eval-env
pip install lighteval[extended_tasks,math,multilingual]
pip install hf-xet
```

## Run evaluation

You can create a new .txt file or use one of the predefined (en.txt, fr.txt). 

```
sbatch evaluate_experiment.slurm $expe_name $task_to_evaluate multilingual
```
where:
- `$expe_name` is your expe name in language ablation. It will evaluate every checkpoints in `"$OpenLLM_OUTPUT/ablations/train/language_ablations/$expe_name/huggingface_checkpoints"`. (we will make it more general)
- `$task_to_evaluate` is the name of your .txt file (without the extension)
- add `multilingual` only if you need to evaluate multilingual tasks. It will activate lighteval args: `--custom-tasks lighteval.tasks.multilingual.tasks`