## NeMo-RL installation instructions on Jean-Zay

Go at a location to install NeMo-RL, e.g.,
```bash
cd $SCRATCH
```

Clone the `nano-v3` branch of the NeMo-RL repo:
```bash
git clone -b nano-v3 https://github.com/NVIDIA-NeMo/RL.git nemo-rl --recursive
cd nemo-rl
```

Load the uv module and create a virtual environment:
```bash
module load arch/h100 # Will be needed for later
module load uv/0.8.3
uv venv
source .venv/bin/activate
```

Load the necessary modules for your system (example for Jean-Zay with H100 GPUs):
```bash
module load cudnn/9.10.2.21-12-cuda
module load cuda/12.8.0
module load nccl/2.27.3-1-cuda
export TORCH_CUDA_ARCH_LIST=9.0
```

Install the required dependencies:
```bash
uv run --locked --extra mcore --directory .
```

If you are on Jean-Zay, the build process will probably get killed (`c++: fatal error: Killed signal terminated program cc1plus `),
You have then to compile after connecting to a compute node with sufficient resources:
```bash
srun -p compil_h100 -c 24 --hint=nomultithread --pty -A qgz@cpu bash

# and re-run:
# uv run --locked --extra mcore --directory .
```


**Troubleshooting:**
When running a script in slurm, if you have an import error (on megatron.bridge for instance),
try running the slurm job with the following environment variable set:
```bash
NRL_FORCE_REBUILD_VENVS=true uv run python ...
```

Add paths to model and tokenizer in the yaml file. The original branch is unable to handle multiple train/val files. We made some modifications to ``__init__.py`` and ``oai_format_dataset.py`` in ``nemo_rl/data/datasets/response_datasets/`` to handle this. For multiple files in training folder, use ``dataset_name: openai_format_multifiles`` in your yaml. Then run, 
```bash
sbatch finetune_8B_H100.slurm
```
to test the example sft script.

After the model is trained, run convert.slurm to convert the checkpoint to HF format by providing path to model config and the checkpoint.


