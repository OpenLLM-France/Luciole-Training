# Training

Add in your `.bashprofile`: 
```bash
export OpenLLM_OUTPUT=/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output
```

## Create a datamix

First, you should have tokenize your dataset and run statistics before! Refer to the [README in `../data`](../data/README.md).

In `datamix/`, you can use `create_datamix.py` to create a new datamix. It will create a folder with a summary of your datamix and a `datamix_xxx.json` file that you can use for training.

For example:
```bash
cd datamix/
python create_datamix.py --data_path path/to/your/tokenized/datasets --starcoder 1.
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

### Pipeline

Here is an example of pipeline command, which will train, convert and evaluate your model:
```bash
cd train/
python slurm_pipeline.py --config xxx.json --output_dir xxx --mode 35b [--email xxx@xxx.com] [--num_nodes 4] --tasks task1.txt task2.txt
```
If no tasks are given it will just train and convert.
Don't forget to use `--mode debug` to try your script before running it.

The script will print the path of your experiment so you can easily ``cd``. 
In the experiment folder, once it has finished, you should have :
- a folder with the same name as the experiment where you can find the checkpoints
- a `datamix` folder containing all used datamixes
- a `failed.out` where errors raised during training are written
- a `completed.txt` showing that training completed sucessfully 
- a `huggingface_checkpoints` folder where you will find the converted checkpoints
- a folder for each job `job_xxx` where you will find the log, config, `command.sh` (command used to launch the job), the `launch.slurm` and datamix of that job
- a `conversion` folder where you will find the log, `conversion.slurm` and a `completed.txt`
- a `evaluation` folder where you will find a slurm for each checkpoints, a `results` folder with a subfolder per checkpoint and a json per ablation run in it. There is also a `slurm_logs` with logs of the evaluation for each task and checkpoint.

### Train

Here is an example of training command:
```bash
cd train/
python slurm_launcher.py --config xxx.json --output_dir xxx --mode 35b [--email xxx@xxx.com] [--num_nodes 4]
```
Don't forget to use `--mode debug` to try your script before running it.

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