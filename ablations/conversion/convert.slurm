#!/bin/bash
#SBATCH --job-name=debug 
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

python -m torch.distributed.launch --nproc_per_node 1 \
    megatron_ckpt_to_nemo.py \
    --checkpoint_folder "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/train/audran/mock_1n_debug/mock_1n_debug/2025-04-11_17-13-38/checkpoints" \
    --checkpoint_name "mock_1n_debug--step=9.0000-epoch=0-consumed_samples=5120.0/weights" \
    --nemo_file_path /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/train/audran/model.nemo \
    --hparams_file "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/train/audran/mock_1n_debug/mock_1n_debug/2025-04-11_17-13-38/checkpoints/context/model.yaml" \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --gpus_per_node 1 \
    --model_type gpt \
    # --hparams_file model_old.yaml \
    # --convert_mlm

    # --hparams_file "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/train/audran/mock_1n_debug/mock_1n_debug/2025-04-11_17-13-38/checkpoints/mock_1n_debug--step=9.0000-epoch=0-consumed_samples=5120.0/context/model.yaml" \

# python -m torch.distributed.launch --nproc_per_node=1 megatron_lm_ckpt_to_nemo.py \
#     --checkpoint_folder "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/train/audran/mock_1n_debug/mock_1n_debug/2025-04-11_17-13-38/checkpoints/" \
#     --checkpoint_name "mock_1n_debug--step=9.0000-epoch=0-consumed_samples=5120.0/weights" \
#     --nemo_file_path /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/train/model.nemo \
#     --hparams_file "model.yaml" \
#     --model_type gpt \
#     --tensor_model_parallel_size 1 \
#     --pipeline_model_parallel_size 1 \
#     --gpus_per_node 1 \
#     --mcore_input

# python -m torch.distributed.launch --nproc_per_node=1 megatron_lm_ckpt_to_nemo.py \
#     --checkpoint_folder "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/train/audran/mock_1n_debug/mock_1n_debug/2025-04-11_17-13-38/checkpoints/mock_1n_debug--step=9.0000-epoch=0-consumed_samples=5120.0/weights" \
#     --checkpoint_name "common.pt" \
#     --nemo_file_path /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/train/model.nemo \
#     --model_type gpt \
#     --tensor_model_parallel_size 1 \
#     --pipeline_model_parallel_size 1 \
#     --gpus_per_node 1

# python test.py