# Evaluation

Benchmark evaluation of pretrained and fine-tuned models using [OpenLLM-France's lighteval fork](https://github.com/OpenLLM-France/lighteval).

## Scripts

| Script | Description |
|--------|-------------|
| `evaluate_experiment.py` | Evaluate all checkpoints of an experiment (submits SLURM array job) |
| `auto_eval.py` | Full pipeline: convert → evaluate → plot (with SLURM job chaining) |
| `plot_results.py` | Plot and compare evaluation results across models |
| `agg_score.py` | Aggregate evaluation scores |
| `run_all_tasks.sh` | Run all benchmark suites for an experiment |

## Prerequisites

Set environment variables:
```bash
export SLURM_ACCOUNT_GPU="your_account@h100"
export SLURM_ACCOUNT_CPU="your_account@cpu"
```

## Install

Create a dedicated environment and install [our lighteval fork](https://github.com/OpenLLM-France/lighteval):
```bash
module purge
module load anaconda-py3/2024.06
module load git
conda create -n eval-env python=3.10
conda activate eval-env

git clone git@github.com:OpenLLM-France/lighteval.git
cd lighteval/
pip install -e .[multilingual,vllm]
pip install language_data langdetect syllapy seaborn python-slugify
pip install unbabel-comet>=2.2.0
pip install jieba fuzzywuzzy rouge # For longbench eval

module load cuda/12.8.0 # or CUDA version used to compile torch (see `python -c "import torch; print(torch.version.cuda)"`)
pip install --no-cache-dir --no-build-isolation mamba-ssm[causal-conv1d]
```

> **Note:** The `mamba-ssm` compilation can fail on front-end nodes due to memory. Use a compute node instead:
> ```bash
> srun -p compil_h100 -c 24 --hint=nomultithread --pty -A $SLURM_ACCOUNT_CPU bash
> ```

If you have issues with installation, you can try `module load gcc/14.2.0`
<!-- module load nccl/2.27.3-1-cuda -->

### Preload assets and datasets

Evaluations run on H100 partition (no internet). Pre-cache everything first:
```bash
bash preload_eval_assets.sh     # LM judges, nltk assets
bash preload_eval_datasets.sh   # Benchmark datasets
bash preload_hf_models.sh       # Baseline models for comparison
```

## Run Evaluations

### Task files

Task files in `tasks/` define which benchmarks to run:

| File | Content |
|------|---------|
| `tasks/en.txt` | English benchmarks |
| `tasks/fr.txt` | French benchmarks |
| `tasks/multilingual.txt` | Multilingual benchmarks |
| `tasks/mmlu.txt` | MMLU |
| `tasks/gsm8k.txt` | GSM8K math |
| `tasks/ruler_*.txt` | RULER long-context benchmarks |

You can create custom task files — one lighteval task per line.

### Evaluate all checkpoints of an experiment

```bash
python evaluate_experiment.py $experiment_path tasks/en.txt

# With multilingual tasks
python evaluate_experiment.py $experiment_path tasks/fr.txt --custom_tasks multilingual

# Limit to specific checkpoint intervals
python evaluate_experiment.py $experiment_path tasks/en.txt --multiple_of 5000

# Limit number of samples per task
python evaluate_experiment.py $experiment_path tasks/fr.txt --custom_tasks multilingual --max_samples 1000
```

The experiment path must contain a `huggingface_checkpoints/` folder.

### Run all benchmarks at once

```bash
bash run_all_tasks.sh $experiment_path --multiple_of 5000 --gpus 2
```

### Evaluate HuggingFace models

Download baseline models first:
```bash
bash preload_hf_models.sh
```

Then evaluate:
```bash
python evaluate_experiment.py $experiment_path tasks/en.txt --hf_model OpenLLM-BPI/Luciole-7B
```

Results are saved in `$experiment_path/evaluation/`.

### Full auto-evaluation pipeline

`auto_eval.py` chains conversion → evaluation → plotting in a single command:

```bash
python auto_eval.py $experiment_path --arch nemotronh --eval_type pretrain --email user@example.com
```

Evaluation types: `pretrain`, `finetune`, `ruler`, `context_extension`.

## Plotting Results

Compare models on evaluation benchmarks:

```bash
# Basic comparison
python plot_results.py $model1 $model2 $model3 --group all --output_path ./figs

# With FLOPS on x-axis
python plot_results.py $model1 $model2 --group all --output_path ./figs --flops

# Limit samples
python plot_results.py $model1 $model2 --group all --output_path ./figs --flops --max_samples 1000
```

Plot groups: `all`, `en`, `fr`, `multilingual`, `ruler`, `finetune`.

### Example: Compare Luciole 1B against baselines

```bash
models="\
$OpenLLM_OUTPUT/pretrain/luciole_serie/luciole_nemotron1b \
$OpenLLM_OUTPUT/pretrain/luciole_serie/luciole_nemotron1b_phase2 \
$OpenLLM_OUTPUT/pretrain/compared_models/OLMo-2-0425-1B \
$OpenLLM_OUTPUT/pretrain/compared_models/EuroLLM-1.7B \
"

python plot_results.py $models --group all \
    --output_path $OpenLLM_OUTPUT/pretrain/luciole_serie/luciole_nemotron1b_phase2/figs --flops
```

## RULER (Long-Context Evaluation)

See [ruler/README.md](ruler/README.md) for RULER benchmark setup and usage.

## Tool Evaluation

Follow [BFCL README](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/README.md)

### Installation

```bash
module purge
module load arch/h100
module load anaconda-py3/2024.06
module load cuda/12.4.1
module load gcc/14.2.0

# Create a new Conda environment with Python 3.10
conda create -n BFCL python=3.10
conda activate BFCL

# Clone the Gorilla repository
git clone https://github.com/ShishirPatil/gorilla.git

# Change directory to the `berkeley-function-call-leaderboard`
cd gorilla/berkeley-function-call-leaderboard

# # Install the package in editable mode
# pip install -e .

# Vllm for local model
pip install -e .[oss_eval_vllm]
pip install -U "vllm>=0.10" transformers
pip install soundfile
```

### Evaluate

```bash
module purge
module load arch/a100 anaconda-py3/2024.06 cuda/12.4.1 gcc/14.2.0 # on a100
module load arch/h100 anaconda-py3/2024.06 cuda/12.4.1 gcc/14.2.0 # on h100
conda activate BFCL
export VLLM_USE_FLASHINFER_SAMPLER=0

# JZ
export MODEL_PATH=...

NAME=${MODEL_PATH##*/}
CATS=non_live,live 
RESULT_DIR="$SCRATCH/result/${NAME}"
SCORE_DIR="$SCRATCH/score/${NAME}" 

mkdir -p "$RESULT_DIR" "$SCORE_DIR" 

# 1. Generate model responses (vLLM inference — same flags you've been using)
bfcl generate \
  --model OpenLLM-France/Luciole-1B-FC \
  --local-model-path "$MODEL_PATH" \
  --backend vllm \
  --num-gpus 1 \
  --gpu-memory-utilization 0.9 \
  --test-category "$CATS" \
  --result-dir $RESULT_DIR 
  
# 2. Score the responses
bfcl evaluate \
  --model OpenLLM-France/Luciole-1B-FC \
  --test-category "$CATS" \
  --result-dir $RESULT_DIR \
  --score-dir  $SCORE_DIR
```