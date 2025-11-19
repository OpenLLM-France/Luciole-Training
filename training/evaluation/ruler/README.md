# RULER evaluation

`config_models.sh`, `config_tasks.sh` and `run.sh` were initially in the RULER repo. We adapted it for our Luciole models.

## Installation

```bash
git clone git@github.com:NVIDIA/RULER.git@ab17b7853df4e0a30b78cd5d2b463ac7dff6ee13
```

We want minimal installation, do let's use the preexistnig nemo modules on JZ...

```bash
module load arch/h100 nemo/2.4.0
pip install --user wonderwords
```

## Run RULER

```bash 
module load arch/h100 nemo/2.4.0
bash run.sh $model_name $benchmark_name
```

see `config_models.sh`, `config_tasks.sh` to see the list of available models and benchmarks

### Add a model

Add a model in `config_models.sh`.

For example:
```bash
luciole-1b-phase1)
    MODEL_PATH="${OpenLLM_OUTPUT}/pretrain/luciole_serie/luciole_nemotron1b/huggingface_checkpoints/luciole_nemotron1b-step_0715786.tmp"
    TOKENIZER_NAME="OpenLLM-BPI/tokenizer_128k-arab-regional_v2"
    MODEL_TEMPLATE_TYPE="base"
    MODEL_FRAMEWORK="vllm"
    ;;
```

### Add a benchmark

