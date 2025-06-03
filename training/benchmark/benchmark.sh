#!/bin/bash

# nodes 32
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 32 --mode benchmark --tp 1
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 32 --mode benchmark --fp8 --tp 1
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 32 --mode benchmark --tp 4
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 32 --mode benchmark --fp8 --tp 4
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 32 --mode benchmark --tp 2
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 32 --mode benchmark --tp 1 --pp 2

python ../train/slurm_launcher.py --output_dir audran/benchmark --name_prefix seq_length8192 --seq_length 8192 --arch llama8b --num_nodes 32 --mode benchmark --tp 1 --pp 2
python ../train/slurm_launcher.py --output_dir audran/benchmark --name_prefix seq_length8192 --seq_length 8192 --arch llama8b --num_nodes 32 --mode benchmark --tp 1 --pp 2 --cp 2
python ../train/slurm_launcher.py --output_dir audran/benchmark --name_prefix seq_length8192 --seq_length 8192 --arch llama8b --num_nodes 32 --mode benchmark --tp 1 --cp 2

python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama1b --num_nodes 32 --mode benchmark --tp 1
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama1b --num_nodes 32 --mode benchmark --fp8 --tp 1

python ../train/slurm_launcher.py --output_dir audran/benchmark --arch mambahybrid8b --num_nodes 32 --mode benchmark --tp 1
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch mambahybrid8b --num_nodes 32 --mode benchmark --fp8 --tp 1
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch mambahybrid8b --num_nodes 32 --mode benchmark --tp 4

# nodes 64
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 64 --mode benchmark --tp 1
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 64 --mode benchmark --fp8 --tp 1
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 64 --mode benchmark --tp 4
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 64 --mode benchmark --fp8 --tp 4

python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama1b --num_nodes 64 --mode benchmark --tp 1

python ../train/slurm_launcher.py --output_dir audran/benchmark --arch mambahybrid8b --num_nodes 64 --mode benchmark --tp 1

# nodes 128
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 128 --mode benchmark --tp 1
python ../train/slurm_launcher.py --output_dir audran/benchmark --arch llama8b --num_nodes 128 --mode benchmark --fp8 --tp 1
