#!/bin/bash
#SBATCH --job-name=auto_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_cpu-dev
#SBATCH --account=qgz@cpu

if [ -z "$1" ]; then
    python auto_eval.py --help
else
    python auto_eval.py "$1"
fi