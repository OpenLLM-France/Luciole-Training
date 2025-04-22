#!/bin/bash
#SBATCH --job-name=convert 
#SBATCH --output=%x.out 
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

python -m torch.distributed.launch --nproc_per_node 1 convert_llama_nemo_to_hf.py \
    --input_name_or_path "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/train/language_ablations/datamix_1._fra_Latn_4n_20b/datamix_1._fra_Latn_4n_20b/checkpoints/datamix_1._fra_Latn_4n_20b--step=19072.0000-epoch=1-consumed_samples=9765376.0-last" \
    --output_path "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/train/audran/hf/model.bin"

python -m torch.distributed.launch --nproc_per_node 1 convert_dist_ckpt_to_llama.py \
    --input_path "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/train/language_ablations/datamix_1._fra_Latn_4n_20b/datamix_1._fra_Latn_4n_20b/checkpoints/datamix_1._fra_Latn_4n_20b--step=19072.0000-epoch=1-consumed_samples=9765376.0-last" \
    --output_path "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/train/test_conversion"