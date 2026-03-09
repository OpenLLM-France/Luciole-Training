
# Train

```bash
cd train
```

```bash
python slurm_launcher.py --output_dir $OpenLLM_OUTPUT/pretrain/ablations_language \
    --name_prefix fr100 --mode phase1 --scheduler cosine --num_nodes 16 --arch llama1b \
    --datamix language_datamixes/fr100.json \
    --time 20:00:00 \
    --qos qos_gpu_h100-t3 \
    --account wuh@h100 \
    --email xxx
```

```bash
python slurm_launcher.py --output_dir $OpenLLM_OUTPUT/pretrain/ablations_language \
    --name_prefix fr5_en95 --mode phase1 --scheduler cosine --num_nodes 16 --arch llama1b \
    --datamix language_datamixes/fr5_en95.json \
    --time 20:00:00 \
    --qos qos_gpu_h100-t3 \
    --account wuh@h100 \
    --email xxx
```

```bash
python slurm_launcher.py --output_dir $OpenLLM_OUTPUT/pretrain/ablations_language \
    --name_prefix fr50_en50 --mode phase1 --scheduler cosine --num_nodes 16 --arch llama1b \
    --datamix language_datamixes/fr50_en50.json \
    --time 20:00:00 \
    --qos qos_gpu_h100-t3 \
    --account wuh@h100 \
    --email xxx
```

```bash
python slurm_launcher.py --output_dir $OpenLLM_OUTPUT/pretrain/ablations_language \
    --name_prefix en100 --mode phase1 --scheduler cosine --num_nodes 16 --arch llama1b \
    --datamix language_datamixes/en100.json \
    --time 20:00:00 \
    --qos qos_gpu_h100-t3 \
    --account wuh@h100 \
    --email xxx
```

```bash
python slurm_launcher.py --output_dir $OpenLLM_OUTPUT/pretrain/ablations_language \
    --name_prefix fr1_en99 --mode phase1 --scheduler cosine --num_nodes 16 --arch llama1b \
    --datamix language_datamixes/fr1_en99.json \
    --time 20:00:00 \
    --qos qos_gpu_h100-t3 \
    --account wuh@h100 \
    --email xxx
```

```bash
python slurm_launcher.py --output_dir $OpenLLM_OUTPUT/pretrain/ablations_language \
    --name_prefix fr33_en66 --mode phase1 --scheduler cosine --num_nodes 16 --arch llama1b \
    --datamix language_datamixes/fr33_en66.json \
    --time 20:00:00 \
    --qos qos_gpu_h100-t3 \
    --account wuh@h100 \
    --email xxx
```

# Convert

```bash
cd conversion
```

```bash
experiment_path=...
sbatch convert.slurm $experiment_path --arch llama 
```

# Evaluate

```bash
cd evaluation
```

```bash
experiment_path=...
python evaluate_experiment.py $experiment_path tasks/idiomatic_expressions.txt --command vllm --custom_tasks idiomatic_expressions --lighteval_kwargs "--save-details" 
``` 

Add `--hf_model ORGA/MODEL_NAME` to evaluate a HF model

# Plot

```bash
cd evaluation

export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
export HF_HUB_OFFLINE=1

module purge
module load anaconda-py3/2024.06
conda activate eval-env
```

```bash
main_dir=/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/pretrain/ablations_language # can be a list of models
experiment_path=$main_dir/en100_llama1b_phase1 \
    other_experiment_path_to_add
python plot_results.py $experiment_path --group xxx --output_path $main_dir/figs
``` 

add `--save_csv` if you want a csv format.