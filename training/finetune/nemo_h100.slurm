#!/bin/bash
#SBATCH --job-name=nemo_launch
#SBATCH --output=log_%x-%j.stdout
#SBATCH --error=log_%x-%j.stderr
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --constraint h100
#SBATCH --account zwy@h100
#SBATCH --qos qos_gpu_h100-t3
#SBATCH --mail-type=ARRAY_TASKS,BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=cthompson@linagora.com,achaturvedi@linagora.com

## load environment
module purge
module load arch/h100
module load nemo/2.4.0
#module load nemo/2.3.1
#module load anaconda-py3/2023.03
#module load pytorch-gpu/py3/2.6.0
#module load pytorch-gpu/py3/2.7.0
#srun rm *.std*
#srun rm *.out
#srun rm core-python*
## launch script on every node
set -x

module list
#srun pip list > $SCRATCH/pipoutput.txt
#MODEL=EleutherAI/pythia-70m
#OUTDIR=$WORK/../commun/Claire/pretrain/lora/$MODEL
export GIT_PYTHON_REFRESH=quiet

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

echo "Arguments: {train_cmd}"
srun torchrun $DISTRIBUTED_ARGS $SCRATCH/nemo_train.py 

# execute script
#srun --output=$SCRATCH/nemo.out --error=$SCRATCH/errors/nemo.err \
#	python $SCRATCH/nemo_train.py 
#--devices 2 \
#--num_nodes 1 \
#--data_dir $SCRATCH/../commun/preprocessed_data/Claire/lit-gpt/padded/$MODEL \
#--checkpoint_dir $WORK/../commun/Claire/checkpoints/$MODEL \
#--out_dir $OUTDIR
