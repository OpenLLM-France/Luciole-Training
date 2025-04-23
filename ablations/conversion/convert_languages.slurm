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

python -m torch.distributed.launch --nproc_per_node 1 convert_experiment.py ablations/train/language_ablations/datamix_1._eng_Latn_4n_20b
python -m torch.distributed.launch --nproc_per_node 1 convert_experiment.py ablations/train/language_ablations/datamix_1._fra_Latn_4n_20b
python -m torch.distributed.launch --nproc_per_node 1 convert_experiment.py ablations/train/language_ablations/datamix_.25_fra_Latn_.75_eng_Latn_4n_20b
python -m torch.distributed.launch --nproc_per_node 1 convert_experiment.py ablations/train/language_ablations/datamix_.5_fra_Latn_.5_eng_Latn_4n_20b
python -m torch.distributed.launch --nproc_per_node 1 convert_experiment.py ablations/train/language_ablations/datamix_.75_fra_Latn_.25_eng_Latn_4n_20b
python -m torch.distributed.launch --nproc_per_node 1 convert_experiment.py ablations/train/language_ablations/datamix_.95_fra_Latn_.05_eng_Latn_4n_20b
# python -m torch.distributed.launch --nproc_per_node 1 convert_experiment.py ablations/train/language_ablations/datamix_dclm_dolmino_4n_20b