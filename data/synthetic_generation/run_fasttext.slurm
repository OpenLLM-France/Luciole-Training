#!/bin/bash
#SBATCH --job-name=fasttext 
#SBATCH --output=slurm/log_%x-%j.out 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16  
#SBATCH --time=01:00:00 
#SBATCH --hint=nomultithread 
#SBATCH --qos=qos_cpu-dev
#SBATCH --account=qgz@cpu

module purge
module load anaconda-py3/2024.06
conda activate distilabel-test

export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface

input_path=/linkhome/rech/gendjf01/uzq54wg/.cache/distilabel/pipelines/pipeline_text_generation_0/53bedcbb53ca1db3fed4f15c1fb88e4d8a6089c8/executions/9225776c9aa457fa8ede8b3581ce5944615b6a72/data/steps_outputs/text_generation_0
output_path=$OpenLLM_OUTPUT/fasttext_classifiers/Qwen3-32B_content_edu_400k

python train_fasttext.py --input_path $input_path --output_path $output_path --label educational_score --from_parquet