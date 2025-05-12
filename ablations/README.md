# Ablations

Add in your `.bashprofile`: 
```bash
export OpenLLM_OUTPUT=/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output
```

## Create a datamix

First, you should have tokenize your dataset and run statistics before! Refer to the [README in `../data`](../data/README.md).

In `data/tools/`, you can use `create_datamix.py` to create a new datamix in `datamix/`. It will create a folder with a summary of your datamix and a `datamix_xxx.json` file that you can use for training.

For example:
```bash
cd data/tools/
python create_datamix.py --data_path path/to/your/tokenized/datasets --code .5 --es 0. --de 0. --it 0.
```

## Train a small 1B model

### Install

Nemo is already installed on JZ, you just need to install `zarr` in your `.local`.

Note: If you have some errors, it might be due to your `.local` directory. Try some clean up and it should run.

<!-- module load arch/h100 nemo/2.1.0 -->
```bash
pip install --user --no-cache-dir zarr
```

If you want to train a Mamba model:
```bash
pip install --user --no-cache-dir --no-build-isolation mamba-ssm[causal-conv1d]
```

### Train

Here is an example of training command,
to train on 20B tokens:
```bash
cd train/
python slurm_launcher.py --config xxx.json --output_dir xxx --mode 20b [--email xxx@xxx.com] [--nodes 4]
```
Use `--mode debug` to try your script before running it.

### Estimate training time for a 1b model

* Number of tokens per step: seq_length (2048) * global_batch_size (512) = 1 048 576
* Number of steps in order to see 35B: 33 378 steps
* Time to see 35b tokens on 1 node: (33 378 * 5.3)/3600 = 49h

| Number of nodes | 1 step | 20B tokens | 35B tokens  | 
|-----------------|--------|------------|-------------|
| 1               | 5.3s   |            | 49h         |
| 2               | 2.7s   |            | 25h         |
| 4               | 1.43s  | 7h34       | 13h15       |

## Convert checkpoints to HF

Run `convert.slurm` to convert all the checkpoints of your experiment, giving the parent output folder.

For example:
```bash
cd conversion/
sbatch convert.slurm $OpenLLM_OUTPUT/ablations/train/languages_ablations/datamix_dclm_dolmino_4n_20b
```

## Evaluate

###  Install

You should create a new environment for evaluation.
```bash
module purge
module load anaconda-py3/2024.06
conda create -n eval-env python=3.10
conda activate eval-env
pip install lighteval[extended_tasks,math,multilingual]==0.8.1
pip install hf-xet
pip install matplotlib
```

Warning: in more recent versions of `lighteval`,
"`pretrained=`" option has been renamed to "`model_name=`".

### Run Evaluations

#### Define the tasks you want to run: 
- you can create a new .txt file 
- or use one of the predefined (en.txt, fr.txt). 

#### Evaluate all the checkpoints of your experiment:
```bash
cd evaluation/
sbatch evaluate_experiment.slurm $experiment_path $task_to_evaluate multilingual
```
where:
- `$experiment_path` is the path to your experiments. It should have a `"huggingface_checkpoints"` folder in it. 
- `$task_to_evaluate` is the name of your .txt file (with the extension)
- add `multilingual` only if you need to evaluate multilingual tasks. It will activate lighteval args: `--custom-tasks lighteval.tasks.multilingual.tasks`

#### Plotting the results...
You can use the script `plot_results.py` to plot your results.
