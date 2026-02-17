# OpenLLM-BPI-Training

Training pipeline for the **Luciole** series of multilingual (French-English) large language models, part of the [OpenLLM-France](https://github.com/OpenLLM-France) initiative.

This repository covers the full LLM lifecycle: data preparation, tokenizer training, pretraining, fine-tuning, and evaluation. It is designed to run on the [Jean-Zay](http://www.idris.fr/jean-zay/) HPC cluster (NVIDIA H100 GPUs) via SLURM.

## Pipeline Overview

```
┌──────────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. Data      │───▶│ 2. Tokenizer  │───▶│ 3. Tokenize  │───▶│ 4. Datamix   │
│ Processing   │    │ Training      │    │ Datasets     │    │ Creation     │
│ data/        │    │ tokenizer_    │    │ data/        │    │ pretrain/    │
│ processing/  │    │ train/        │    │ tokenization/│    │ datamix/     │
└──────────────┘    └───────────────┘    └──────────────┘    └──────┬───────┘
                                                                    │
                    ┌───────────────┐    ┌──────────────┐    ┌──────▼───────┐
                    │ 8. Evaluation │◀───│ 7. Convert   │◀───│ 5. Pretrain  │
                    │ evaluation/   │    │ to HF format │    │ pretrain/    │
                    └───────┬───────┘    │ pretrain/    │    │ train/       │
                            │            │ conversion/  │    └──────────────┘
                            │            └──────────────┘
                    ┌───────▼───────┐
                    │ 6. Fine-tune  │
                    │ (optional)    │
                    │ finetune/     │
                    └───────────────┘
```

## Directory Structure

| Directory | Description | Documentation |
|-----------|-------------|---------------|
| `data/processing/` | 50+ dataset-specific preprocessing scripts (FineWeb, DCLM, Gallica, StarCoder, ...) | [data/README.md](data/README.md) |
| `data/tokenization/` | Tokenize processed datasets and compute statistics | [data/README.md](data/README.md) |
| `data/synthetic_generation/` | Generate synthetic training data using LLMs | [data/synthetic_generation/README.md](data/synthetic_generation/README.md) |
| `tokenizer_train/` | Train and evaluate custom BPE tokenizers | [tokenizer_train/README.md](tokenizer_train/README.md) |
| `pretrain/train/` | Distributed pretraining via NeMo + SLURM | [pretrain/README.md](pretrain/README.md) |
| `pretrain/conversion/` | Convert NeMo checkpoints to HuggingFace format | [pretrain/conversion/README.md](pretrain/conversion/README.md) |
| `finetune/` | SFT and RL fine-tuning with NeMo | [finetune/README.md](finetune/README.md) |
| `evaluation/` | Benchmark evaluation with lighteval | [evaluation/README.md](evaluation/README.md) |

## Quickstart

### Prerequisites

- Access to Jean-Zay HPC cluster (or similar SLURM-based cluster with H100 GPUs)
- Set the following environment variables in your `.bashprofile`:

```bash
export OpenLLM_OUTPUT=/path/to/shared/output     # Shared output directory
export SLURM_ACCOUNT_GPU="your_account@h100"      # SLURM GPU account
export SLURM_ACCOUNT_CPU="your_account@cpu"        # SLURM CPU account
```

### Example pipeline (train + convert + evaluate)

```bash
# 1. Process data
cd data/ && source set_env.sh
python processing/fineweb2 --local   # test locally
python processing/fineweb2            # run on SLURM

# 2. Tokenize
python tokenization/run_tokenization.py datasets.yaml $OpenLLM_OUTPUT/data/tokens \
    --tokenizer_name OpenLLM-BPI/tokenizer_128k-arab-regional_v2

# 3. Train
cd ../pretrain/train/
python slurm_launcher.py --output_dir $OpenLLM_OUTPUT/pretrain/my_exp \
    --arch nemotron1b --mode phase1 --datamix datamixes/luciole_phase1.json

# 4. Convert checkpoints to HuggingFace format
cd ../conversion/
sbatch convert.slurm $OpenLLM_OUTPUT/pretrain/my_exp

# 5. Evaluate
cd ../../evaluation/
python evaluate_experiment.py $OpenLLM_OUTPUT/pretrain/my_exp tasks/en.txt
```

## Supported Model Architectures

Recipes in `pretrain/train/recipes/`:

| Family | Sizes |
|--------|-------|
| LLaMA | 1B, 3B, 8B, 21B, 24B, 70B |
| Mistral | 12B, Small 24B |
| Mixtral | 8x7B |
| Nemotron | 1B, 4B, 8B, 22B, 23B + variants |
| NemotronH | 8B, 47B (hybrid Mamba/Transformer) |

## Environment Setup

Each pipeline stage uses its own environment. See the README in each subdirectory for setup instructions:

- **Data processing**: `data/README.md` — conda `datatrove-env` (Python 3.10) + [datatrove](https://github.com/linagora-labs/datatrove) fork
- **Pretraining**: `pretrain/README.md` — NeMo module on Jean-Zay + `zarr`
- **Fine-tuning**: `finetune/README.md` — NeMo-RL with `uv`
- **Evaluation**: `evaluation/README.md` — conda `eval-env` + [lighteval](https://github.com/OpenLLM-France/lighteval) fork

## Key Environment Variables

| Variable | Description |
|----------|-------------|
| `OpenLLM_OUTPUT` | Shared output directory for all pipeline stages |
| `SLURM_ACCOUNT_GPU` | SLURM account for GPU jobs (e.g. `wuh@h100`) |
| `SLURM_ACCOUNT_CPU` | SLURM account for CPU jobs (e.g. `qgz@cpu`) |
| `HF_HOME` | HuggingFace cache directory |
| `HF_MODELS_CACHE` | Local mirror of HuggingFace models |
| `HF_DATASETS_MIRROR` | Local mirror of HuggingFace datasets |

## Linting

```bash
pre-commit run --all-files
# Or directly:
ruff check . --fix
ruff format .
```

## License

This project is licensed under the GNU General Public License v3 — see [LICENSE](LICENSE) for details.
