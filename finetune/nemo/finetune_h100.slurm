#!/bin/bash
#SBATCH --job-name=finetune_nemo
#SBATCH --output=log_%x-%j.stdout
#SBATCH --error=log_%x-%j.stderr
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --constraint h100
#SBATCH --account zwy@h100
#SBATCH --qos qos_gpu_h100-dev
#SBATCH --mail-type=ARRAY_TASKS,BEGIN,END,FAIL,REQUEUE

## load environment
module purge
module load arch/h100
module load nemo/2.4.0

export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Set environment variables for distributed training
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
GPUS_PER_NODE=4  # Adjust based on your setup

DISTRIBUTED_ARGS=" \
       --nproc_per_node $GPUS_PER_NODE \
       --nnodes $SLURM_NNODES \
       --node_rank $SLURM_NODEID \
       --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
       --rdzv_backend c10d \
       --max_restarts 0 \
       "

srun torchrun $DISTRIBUTED_ARGS finetune_test.py
