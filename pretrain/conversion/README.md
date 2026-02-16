# Checkpoint Conversion

Convert NeMo distributed checkpoints to HuggingFace format (and vice versa).

## Scripts

| Script | Description |
|--------|-------------|
| `convert_experiment.py` | Main conversion script — converts all (or filtered) checkpoints in an experiment |
| `convert_dist_to_hf.py` | Low-level: distributed NeMo checkpoint → HuggingFace |
| `hf_to_nemo.py` | Reverse: HuggingFace → NeMo format |
| `convert.slurm` | SLURM job template for batch conversion |

## Usage

### Convert all checkpoints of an experiment

```bash
sbatch convert.slurm $OpenLLM_OUTPUT/pretrain/luciole_serie/my_experiment
```

### Convert with filtering

```bash
# Only convert checkpoints at step multiples of 5000
torchrun --nproc_per_node=1 convert_experiment.py $experiment_path --arch nemotron --multiple_of 5000

# Only convert the last checkpoint
torchrun --nproc_per_node=1 convert_experiment.py $experiment_path --arch nemotronh --last
```

### Key arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `experiment_path` | Path to experiment folder (must contain a checkpoints subfolder) | — |
| `--arch` | Architecture: `llama`, `nemotron`, `nemotronh` | `nemotron` |
| `--multiple_of` | Only convert checkpoints at step multiples of N | 1 |
| `--last` | Only convert the final checkpoint | false |

### Convert HuggingFace → NeMo

```bash
sbatch hf_to_nemo.slurm
# Or directly:
torchrun --nproc_per_node=1 hf_to_nemo.py --model_name path/to/hf/model --output_path path/to/nemo/output
```

## Output

Converted checkpoints are saved in `$experiment_path/huggingface_checkpoints/` with names like:
```
my_experiment-step_0100000
my_experiment-step_0200000
```
