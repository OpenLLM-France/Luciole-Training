#!/bin/bash

# Loss benchmarks

python ../train/slurm_launcher.py --output_dir audran/benchmark100 --arch llama8b --num_nodes 8 --mode benchmark100 --tp 1
python ../train/slurm_launcher.py --output_dir audran/benchmark100 --arch llama8b --num_nodes 8 --mode benchmark100 --fp8 --tp 1
python ../train/slurm_launcher.py --output_dir audran/benchmark100 --arch llama8b --num_nodes 8 --mode benchmark100 --fp8 --tp 1 --batch_size 512 --seq_length 4096

# tp/cp/pp

python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 32 --mode benchmark --tp 1
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 64 --mode benchmark --tp 1
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 128 --mode benchmark --tp 1

python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 32 --mode benchmark --fp8 --tp 1
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 64 --mode benchmark --fp8 --tp 1
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 128 --mode benchmark --fp8 --tp 1

python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 32 --mode benchmark --tp 4
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 64 --mode benchmark --tp 4

python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 32 --mode benchmark --fp8 --tp 4
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 64 --mode benchmark --fp8 --tp 4

python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 32 --mode benchmark --tp 2
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 32 --mode benchmark --tp 1 --pp 2

# seq length
python ../train/slurm_launcher.py --output_dir audran/benchmark --name_prefix seq_length8192 --seq_length 8192 --arch llama8b --num_nodes 32 --mode benchmark --tp 1 --pp 2
python ../train/slurm_launcher.py --output_dir audran/benchmark --name_prefix seq_length8192 --seq_length 8192 --arch llama8b --num_nodes 32 --mode benchmark --tp 1 --pp 2 --cp 2
python ../train/slurm_launcher.py --output_dir audran/benchmark --name_prefix seq_length8192 --seq_length 8192 --arch llama8b --num_nodes 32 --mode benchmark --tp 1 --cp 2

# batch size
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 32 --mode benchmark --name_prefix batch_size512 --tp 1 --batch_size 512 --seq_length 4096
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 64 --mode benchmark --name_prefix batch_size512 --tp 1 --batch_size 512 --seq_length 4096

# python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 32 --mode benchmark --name_prefix batch_size512 --fp8 --tp 1 --batch_size 512 --seq_length 4096
# python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 64 --mode benchmark --name_prefix batch_size512 --fp8 --tp 1 --batch_size 512 --seq_length 4096
# python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 128 --mode benchmark --name_prefix batch_size512 --fp8 --tp 1 --batch_size 512 --seq_length 4096

python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 32 --mode benchmark --name_prefix batch_size512_20s --fp8 --tp 1 --batch_size 512 --seq_length 4096
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 64 --mode benchmark --name_prefix batch_size512_20s --fp8 --tp 1 --batch_size 512 --seq_length 4096
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 128 --mode benchmark --name_prefix batch_size512_20s --fp8 --tp 1 --batch_size 512 --seq_length 4096

# llama 1b
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama1b --num_nodes 32 --mode benchmark --tp 1
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama1b --num_nodes 64 --mode benchmark --tp 1

python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama1b --num_nodes 32 --mode benchmark --fp8 --tp 1

# mamba
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch mambahybrid8b --num_nodes 32 --mode benchmark --tp 1
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch mambahybrid8b --num_nodes 64 --mode benchmark --tp 1

python ../train/slurm_launcher.py --output_dir audran/benchmark --arch mambahybrid8b --num_nodes 32 --mode benchmark --fp8 --tp 1
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch mambahybrid8b --num_nodes 32 --mode benchmark --tp 4


