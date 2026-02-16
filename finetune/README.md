# Fine-tuning

Fine-tune pretrained Luciole models using NeMo, with support for SFT (Supervised Fine-Tuning) and RL-based training.

## Scripts

| Script | Description |
|--------|-------------|
| `finetune.py` | Main fine-tuning script — supports multiple architectures and parallelism strategies |
| `finetune_recipe.py` | Recipe system: trainer, optimizer, precision, and checkpoint configuration |
| `finetune_dataloader.py` | Data loading for fine-tuning |
| `finetune_sft_mix_8b.py` | SFT with mixed datasets on 8B models |
| `finetune_long_context.py` | Long context fine-tuning |
| `finetune_test.py` | Quick fine-tuning test |
| `test_custom_data.py` | Test custom dataset loading |

## Environment Setup

### Option 1: NeMo on Jean-Zay

```bash
module load arch/h100 nemo/2.4.0
pip install --user --no-cache-dir zarr
```

### Option 2: NeMo-RL

See [nemo-rl/README.md](nemo-rl/README.md) for installation instructions.

## Usage

### Basic fine-tuning

```bash
# Debug run
python finetune.py --arch nemotronh8b --mode debug --num_nodes 1

# Production run (10B tokens)
python finetune.py --arch nemotronh8b --mode 10b --num_nodes 4 \
    --batch_size 32 --seq_length 4096 --tensor_parallelism 2
```

### Key arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--arch` | Model architecture (`llama1b`, `llama8b`, `nemotronh8b`, `mamba1b`, `mixtral8x7`, ...) | — |
| `--mode` | Training duration: `debug`, `benchmark`, or token count (`10b`, `100b`, `1m`) | `debug` |
| `--num_nodes` | Number of SLURM nodes | 1 |
| `--batch_size` | Micro batch size | 4 |
| `--seq_length` | Sequence length | 100 |
| `--tensor_parallelism` | Tensor parallelism degree | 1 |
| `--pipeline_parallelism` | Pipeline parallelism degree | 1 |
| `--fp8` | Enable FP8 mixed precision | false |
| `--lr` | Learning rate | (from recipe) |
| `--save_every` | Checkpoint interval (e.g. `4m` = 4 million tokens) | `4m` |

### Submit via SLURM

```bash
sbatch finetune_h100.slurm   # H100 GPUs
sbatch finetune.slurm         # Standard submission
```

## Directory Structure

```
finetune/
├── finetune.py                # Main entry point
├── finetune_recipe.py         # NeMo recipe configuration
├── finetune_dataloader.py     # Data loading
├── finetune_sft_mix_8b.py     # SFT mixed-data 8B variant
├── finetune_long_context.py   # Long context variant
├── finetune.slurm             # SLURM job template
├── finetune_h100.slurm        # H100-specific SLURM template
├── nemo-rl/                   # NeMo-RL installation and config
├── nemo_patch/                # Custom NeMo patches
├── databricks/                # Databricks-specific utilities
└── set_env.sh                 # Environment setup
```
