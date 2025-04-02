#!/bin/bash

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# module purge
# module load arch/h100 nemo/2.1.0

module purge
module load anaconda-py3/2023.09 
module load cuda/12.4.1
module load gcc/11.3.1
module load ffmpeg/6.1.1
module load singularity/3.8.5
conda activate /lustre/fsn1/projects/rech/qgz/commun/envs/nemo-env