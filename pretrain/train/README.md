# Pretraining

Core training scripts for distributed LLM pretraining with NeMo on SLURM.

## Scripts

| Script | Description |
|--------|-------------|
| `slurm_launcher.py` | Submit a single training SLURM job |
| `train_model.py` | Core NeMo training loop (called by SLURM jobs) |
| `callbacks.py` | Custom training callbacks (timing, checkpointing) |
| `dataloader.py` | Data loading and datamix configuration |


## Launching Training Jobs

```bash
python slurm_launcher.py \
    --output_dir $OpenLLM_OUTPUT/pretrain/my_experiment \
    --arch nemotron1b \
    --mode phase1 \
    --num_nodes 16 \
    --datamix datamixes/luciole_phase1.json
```

### Training modes

| Mode | Description |
|------|-------------|
| `debug` | Short run for testing (1h, uses dev QoS) |
| `benchmark` | Performance benchmarking |
| `ablation` | Ablation run (like `phase1`, but with a shorter 500-step warmup) |
| `phase1` | Initial pretraining phase |
| `phase2` | Secondary training (continued pretraining) |
| `annealing` | Learning rate annealing phase |
| `context_extension` | Extend context window length |

### Key arguments (slurm_launcher.py)

| Argument | Description | Default |
|----------|-------------|---------|
| `--output_dir` | Experiment output directory | — |
| `--arch` | Model architecture (see `recipes/`) | `llama1b` |
| `--mode` | Training mode (see above) | `debug` |
| `--datamix` | Path to datamix JSON/YAML config | — |
| `--num_nodes` | Number of SLURM nodes | 1 |
| `--gpus_per_node` | GPUs per node | 4 |
| `--qos` | SLURM QoS | `qos_gpu_h100-as` |
| `--account` | SLURM account | `$SLURM_ACCOUNT_GPU` or `wuh@h100` |
| `--nemo_version` | NeMo module version | `nemo/2.3.1` |
| `--name_prefix` | Prefix added to the experiment name | (empty) |
| `--email` | Email for SLURM notifications (lines omitted if unset) | — |
| `--dependency` | SLURM job dependency | — |

LR/schedule arguments (from `train_model.py`, also accepted by `slurm_launcher.py`):

| Argument | Description | Default |
|----------|-------------|---------|
| `--seq_length` | Sequence length | 4096 |
| `--scheduler` | LR scheduler (`wsd`, `cosine`) | `wsd` |
| `--max_lr` | Peak learning rate | recipe default |
| `--min_lr` | Final learning rate | — (cosine: `max_lr * 0.1` if unset) |

## Model Recipes

Architecture recipes are in `recipes/`. Each defines model dimensions, attention config, and parallelism defaults:

| Recipe | Parameters |
|--------|-----------|
| `nemotron_1b` | 1B Nemotron |
| `nemotronh_8b` | 8B hybrid Mamba/Transformer |
| `nemotron_23b` | 23B Nemotron |
| `llama_21b` | 21B LLaMA |
| `llama_24b` | 24B LLaMA |
| `mistral_small3_24b` | 24B Mistral Small |
| ... | See `recipes/` for full list |

## Datamixes

Training data mixtures are JSON files in `datamixes/`:

| File | Description |
|------|-------------|
| `luciole_phase1.json` | Phase 1 pretraining mix |
| `luciole_phase2.json` | Phase 2 pretraining mix |
| `luciole_phase3.json` | Phase 3 pretraining mix (annealing) |
| `luciole_context_extension.json` | Context extension mix |
| `mock.json` | Small mock dataset for testing |
| ... | See `datamixes/` for the full list |

To create a new datamix, see `../datamix/` (or `../../data/tokenization/create_datamix.py`).

## Experiment Output Structure

After a pipeline run, the experiment folder contains:

```
my_experiment/
├── my_experiment/
│   └── checkpoints/          # NeMo checkpoints
├── huggingface_checkpoints/  # Converted HF checkpoints
├── job_<id>/                 # Per-job logs
|   ├── failed.out            # SLURM error log
│   ├── log.out               # SLURM output log
│   ├── command.sh            # Command used to launch the job
│   └── launch.slurm          # SLURM submission script
└── evaluation/               # Evaluation results
```
