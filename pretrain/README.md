# Pretraining

Distributed LLM pretraining pipeline using NeMo, with SLURM job orchestration for training, checkpoint conversion, and evaluation.

## Prerequisites

Add to your `.bashprofile`:
```bash
export OpenLLM_OUTPUT=/path/to/shared/output
export SLURM_ACCOUNT_GPU="your_account@h100"
export SLURM_ACCOUNT_CPU="your_account@cpu"
```

## Directory Structure

```
pretrain/
├── train/          # Training scripts and recipes (see train/README.md)
│   ├── slurm_launcher.py    # Submit a single training SLURM job
│   ├── train_model.py       # Core NeMo training loop
│   ├── recipes/             # Model architecture recipes
│   └── datamixes/           # Training data mixture configs
├── conversion/     # Checkpoint conversion (see conversion/README.md)
└── benchmark/      # Performance benchmarking
```

## Install

NeMo is pre-installed on Jean-Zay. You only need:
```bash
pip install --user --no-cache-dir zarr
```

For Mamba/hybrid models:
```bash
pip install --user --no-cache-dir --no-build-isolation mamba-ssm[causal-conv1d]
```

> **Note:** If you encounter errors, try cleaning your `~/.local` directory.

See [train/README.md](train/README.md) for detailed training options.

## Train Only

```bash
cd train/
python slurm_launcher.py \
    --output_dir $OpenLLM_OUTPUT/pretrain/my_experiment \
    --arch nemotron1b \
    --mode phase1 \
    --num_nodes 4 \
    --datamix datamixes/luciole_phase1.json
```

## Create a Datamix

First, tokenize your datasets and compute statistics (see [data/README.md](../data/README.md)).

Then create a datamix:
```bash
cd datamix/
python create_datamix.py --data_path /path/to/tokenized/datasets --starcoder 1.
```

This produces a `datamix_xxx.json` file to pass to training with `--datamix`.

## Convert Checkpoints to HuggingFace Format

```bash
cd conversion/
sbatch convert.slurm $OpenLLM_OUTPUT/pretrain/my_experiment
```

See [conversion/README.md](conversion/README.md) for more options.

## Experiment Output Structure

After a pipeline run:
```
my_experiment/
├── my_experiment/
│   └── checkpoints/          # NeMo distributed checkpoints
├── huggingface_checkpoints/  # Converted HF checkpoints
├── datamix/                  # Datamix configs used
├── job_<id>/                 # Per-job SLURM logs and configs
│   ├── log.out               # Training log
│   ├── failed.out            # Error log
│   ├── command.sh            # Command used
│   └── launch.slurm          # SLURM script
├── conversion/               # Conversion logs
├── evaluation/               # Evaluation results and plots
├── completed.txt             # Success marker
└── failed.out                # Global error log
```

## Performance Benchmarking

Run throughput benchmarks across architectures:
```bash
cd benchmark/
python benchmark_pipeline.py --num_nodes 4 --mode minimal
```

Modes: `debug` (1 config), `minimal` (6), `full` (~10 with FP8), `extra` (15+), `seq_8192`.

### Training Time Estimates (1B model, seq_length=2048, batch_size=512)

| Nodes | Time/step | 20B tokens | 35B tokens |
|-------|-----------|------------|------------|
| 1     | 5.3s      | —          | 49h        |
| 2     | 2.7s      | —          | 25h        |
| 4     | 1.43s     | 7h34       | 13h15      |
