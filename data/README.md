# Data

Data preparation pipeline: preprocessing raw datasets, tokenizing them, and computing statistics for datamix creation.

## Directory Structure

```
data/
├── processing/              # 50+ dataset-specific preprocessing scripts
│   ├── pretraining              # All datasets used during pretraining
│   │   ├── utils.py             # Shared utilities (create_executor, create_parser, ...)
│   │   ├── web_utils.py         # Web data pipeline (deduplication, PII, robots.txt, ...)
│   │   ├── fineweb2.py          # FineWeb-2 dataset
│   │   ├── starcoder_data.py    # StarCoder code dataset
│   │   ├── gallica.py           # French Gallica library dataset
│   │   ├── dclm.py              # DCLM dataset
│   │   └── ...                  # One script per dataset source
│   ├── postraining              # # All datasets used during posttraining
│   │   ├── utils.py             # Shared utilities (create_executor, create_parser, ...)
│   │   └── ...                  # One script per dataset source
├── tokenization/            # Tokenization and statistics
│   ├── run_tokenization.py  # Main tokenization script (submits one SLURM job per dataset)
│   ├── merge_stats.py       # Merge per-file statistics
│   ├── create_datamix.py    # Generate datamix JSON from statistics
│   └── ...
├── synthetic_generation/    # Generate synthetic training data with LLMs
│   ├── README.md
│   └── wrap/                # Batch generation launcher
├── tools/                   # Data analysis utilities
│   ├── summary_stats.py     # Compute summary statistics on datasets
│   ├── fasttext_stats.py    # FastText-based quality scoring
│   └── starcoder_language_stats.py  # Language detection for code
├── set_env.sh               # Environment variables setup
└── requirements.txt         # Python dependencies
```

## Environment Setup

### 1. Create environment

```bash
module purge
module load arch/h100
module load anaconda-py3/2024.06
module load cuda/12.4.1
conda create -n datatrove-env python=3.10
conda activate datatrove-env
pip install -r requirements.txt
```

### 2. Clone datatrove (custom fork)

```bash
git clone https://github.com/linagora-labs/datatrove.git
cd datatrove
git checkout lucie_v2
pip install -e .[io,processing,inference]
pip install vllm
pip install --no-build-isolation flash-attn
```

### 3. Set environment variables

```bash
source set_env.sh
```

This sets `$OpenLLM_OUTPUT`, `$HF_HOME`, and activates `datatrove-env`.

You should also configure your SLURM accounts:
```bash
export SLURM_ACCOUNT_GPU="your_account@h100"
export SLURM_ACCOUNT_CPU="your_account@cpu"
```

## Processing Datasets

Each dataset has its own script in `processing/`. All scripts follow the same pattern using the shared `utils.py` utilities:

```bash
source set_env.sh

# Test locally (first 1000 samples)
python processing/fineweb2 --local

# Run on SLURM (full dataset)
python processing/fineweb2

# Run for ablation (5% sample)
python processing/fineweb2 --ablation
```

### Adding a new dataset

Create a new script in `processing/` following the existing patterns. The key utilities from `utils.py` are:
- `create_parser()` — Standard argument parser with `--local`, `--debug`, `--ablation`, `--jz` flags
- `parse_args(parser)` — Parse arguments and set `data_path` from `$OpenLLM_OUTPUT`
- `create_executor(pipeline, ...)` — Create a local or SLURM executor for the processing pipeline
- `add_sampler_filter(pipeline, rate)` — Add a sampling step for ablations (5% by default)

## Tokenization

Once datasets are preprocessed in `$OpenLLM_OUTPUT/data/raw_datasets/`:

### 1. Configure datasets to tokenize

Create a YAML file (e.g. `tokenization/datasets_to_tokenize.yaml`):
```yaml
dataset_groups:
  - root_path: <<OpenLLM_OUTPUT>>/data/raw_data/data_for_ablation
    datasets:
      - name: fineweb2_fra_Latn_cluster_5-100
        path: fineweb2/data/fra_Latn/clusters/cluster_size-5-100
```

### 2. Run tokenization

```bash
python tokenization/run_tokenization.py YAML_FILE OUTPUT_DIR \
    --tokenizer_name OpenLLM-BPI/tokenizer_128k-arab-regional_v2
```
This submits one SLURM job per dataset.

### 3. Compute statistics

```bash
sbatch run_statistics.slurm OUTPUT_DIR
python merge_stats.py OUTPUT_DIR
python visualize_token_stats.py
```

### 4. Create a datamix

```bash
python tokenization/create_datamix.py OUTPUT_DIR/datamix_output \
    --token_dir OUTPUT_DIR --seq_length 4096
```

### 5. Next steps

- [Pretraining](../pretrain/README.md) — use the datamix for training
- [Ablations](../ablations/README.md) — run controlled experiments

## Pre-downloading from HuggingFace

Set a shared HF cache:
```bash
export HF_HOME=$HF_HOME  # or set explicitly
```

Download datasets or models:
```bash
# Dataset
huggingface-cli download open-web-math/open-web-math --repo-type dataset

# Subset only
huggingface-cli download EleutherAI/proof-pile-2 --repo-type dataset --include algebraic-stack/*

# Tokenizer
huggingface-cli download OpenLLM-BPI/tokenizer_128k-arab-regional_v2 --repo-type model
```
