# Ablations

Add in your `.bashprofile`: 
```
export OpenLLM_OUTPUT=/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output
```

## Create a datamix

In `tools/`, you can use `create_datamix.py` to create a new datamix in `datamix/`. It will create a folder with a summary of your datamix and a `datamix_xxx.json` file that you can use for training.

For example:
```
python create_datamix.py --data_path path/to/your/tokenized/datasets --code .5 --es 0. --de 0. --it 0.
```

### Important note
You should have tokenize your dataset and run statistics before! Refer to the README in `../data/tokenization`.

## Train a small 1B model

### Install

Nemo is already installed on JZ, you just need to install `zarr` in your `.local`.

Note: If you have some errors, it might be due to your `.local` directory. Try some clean up and it should run.

```
module load arch/h100 nemo/2.1.0
pip install --user --no-cache-dir zarr
```

### Train

Example of command training:

In debug mode:
```
python slurm_launcher.py --config mock.json --output_dir test --mode debug
```

Otherwise, if you want to train on 20B tokens:
```
python slurm_launcher.py --config xxx.json --output_dir xxx --mode 20b
```

### Estimate training time for a 1b model

`Number of tokens per step: seq_length (2048) * global_batch_size (512) = 1 048 576`

`Number of steps in order to see 35B: 33 378 steps`

`Time to see 35b tokens on 1 node: (33 378 * 5.3)/3600 = 49h `

| Number of nodes | 1 step | 20B tokens | 35B tokens  | 
|-----------------|--------|------------|-------------|
| 1               | 5.3s   |            | 49h         |
| 2               | 2.7s   |            | 25h         |
| 4               | 1.43s  | 7h34       | 13h15       |

## Convert checkpoints to HF

Run `convert.slurm` to convert all the checkpoints of your experiment.

For example:
```
sbatch convert.slurm $OpenLLM_OUTPUT/ablations/train/languages_ablations/datamix_dclm_dolmino_4n_20b
```

## Evaluate

###  Install

You should create a new environment for evaluation.

```
module purge
module load anaconda-py3/2024.06
conda create -n eval-env python=3.10
conda activate eval-env
pip install lighteval[extended_tasks,math,multilingual]
pip install hf-xet
```

### Run Evaluations

#### Define the tasks you want to run: 
- you can create a new .txt file 
- or use one of the predefined (en.txt, fr.txt). 

#### Evaluate all the checkpoints of your experiment:
```
sbatch evaluate_experiment.slurm $expe_name $task_to_evaluate multilingual
```
where:
- `$expe_name` is your expe name in language ablation. It will evaluate all the checkpoints in `"$OpenLLM_OUTPUT/ablations/train/language_ablations/$expe_name/huggingface_checkpoints"`. (we will make it more general)
- `$task_to_evaluate` is the name of your .txt file (without the extension)
- add `multilingual` only if you need to evaluate multilingual tasks. It will activate lighteval args: `--custom-tasks lighteval.tasks.multilingual.tasks`

