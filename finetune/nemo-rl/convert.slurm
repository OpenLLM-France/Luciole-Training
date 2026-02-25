#!/bin/bash
#SBATCH --job-name=nemorl_convert_tohf
#SBATCH --output=log_%x-%j.stdout
#SBATCH --error=log_%x-%j.stderr
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --constraint h100
#SBATCH --account wuh@h100
#SBATCH --qos qos_gpu_h100-dev
#SBATCH --mail-type=ARRAY_TASKS,BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=achaturvedi@linagora.com,cthompson@linagora.com

## load environment
module purge
module load arch/h100
module load cuda/12.8.0
module load cudnn/9.10.2.21-12-cuda
module load nccl/2.27.3-1-cuda
module load uv/0.8.3

source .venv/bin/activate

export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
export TORCH_CUDA_ARCH_LIST=9.0

#python examples/converters/convert_megatron_to_hf.py \
#  --config results/sft_luciole_nemotronh8B_luciemix/step_1156/config.yaml \
#  --megatron-ckpt-path results/sft_luciole_nemotronh8B_luciemix/step_1156/policy/weights/iter_0000000 \
#  --hf-ckpt-path hf_ckpts_8B_luciemix/

uv run --no-build-isolation python examples/converters/convert_megatron_to_hf.py \
  --config PATH_TO_MODEL_CONFIG \
  --megatron-ckpt-path PATH_TO_CKPT \
  --hf-ckpt-path OUTPUT_FOLDER
