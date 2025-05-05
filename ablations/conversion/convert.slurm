#!/bin/bash
#SBATCH --job-name=convert
#SBATCH --output=log_%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --account=wuh@h100
#SBATCH --constraint=h100

source ../set_env.sh

echo $1

python -m torch.distributed.launch --nproc_per_node 1 convert_experiment.py $1