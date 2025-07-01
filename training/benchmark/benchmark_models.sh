#!/bin/bash

# tp/cp/pp

# python ../train/slurm_launcher.py --output_dir audran/benchmark_models --arch llama8b --num_nodes 128 --mode benchmark --seq_length 4096 --batch_size 1024
python ../train/slurm_launcher.py --output_dir audran/benchmark_models --arch llama8b --num_nodes 128 --mode benchmark --seq_length 4096 --batch_size 1024 --fp8

python ../train/slurm_launcher.py --output_dir audran/benchmark_models --arch llama1b --num_nodes 128 --mode benchmark --seq_length 4096 --batch_size 1024

# python ../train/slurm_launcher.py --output_dir audran/benchmark_models --arch llama3b --num_nodes 128 --mode benchmark --seq_length 4096 --batch_size 1024
python ../train/slurm_launcher.py --output_dir audran/benchmark_models --arch llama3b --num_nodes 128 --mode benchmark --seq_length 4096 --batch_size 1024 --fp8

#  tp 4 pp 4 vpp 5 cp 2
python ../train/slurm_launcher.py --output_dir audran/benchmark_models --arch llama70b --num_nodes 128 --mode benchmark --seq_length 8192 --batch_size 512 --fp8 

# tp 1 cp 2 vpp 8 pp 4 ep 8
# python ../train/slurm_launcher.py --output_dir audran/benchmark_models --arch mixtral8x7 --num_nodes 128 --mode benchmark --seq_length 4096 --batch_size 1024
# python ../train/slurm_launcher.py --output_dir audran/benchmark_models --arch mixtral8x7 --num_nodes 128 --mode benchmark --seq_length 4096 --batch_size 1024 --fp8

# tp2 test batch size
# python ../train/slurm_launcher.py --output_dir audran/benchmark_models --arch nemotronh8b --num_nodes 128 --mode benchmark --seq_length 8192 --batch_size 768 --name_prefix s8192_b768 --tp 2
python ../train/slurm_launcher.py --output_dir audran/benchmark_models --arch nemotronh8b --num_nodes 128 --mode benchmark --seq_length 8192 --batch_size 768 --name_prefix s8192_b768 --tp 2 --fp8

# # tp8  test batch size
# python ../train/slurm_launcher.py --output_dir audran/benchmark_models --arch nemotronh56b --num_nodes 128 --mode benchmark --seq_length 8192 --batch_size 768
# python ../train/slurm_launcher.py --output_dir audran/benchmark_models --arch nemotronh56b --num_nodes 128 --mode benchmark --seq_length 8192 --batch_size 768 --fp8