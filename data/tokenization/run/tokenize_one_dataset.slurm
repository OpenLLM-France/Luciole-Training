#!/bin/bash
#SBATCH --output=slurm/%x.out 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=20:00:00 
#SBATCH --hint=nomultithread 
#SBATCH --account=qgz@cpu
#SBATCH --partition=cpu_p1

# Requires nemo installation - on Jeanzay for now
module purge
module load arch/a100 nemo/2.2.1
export OpenLLM_OUTPUT=$qgz_ALL_CCFRSCRATCH/OpenLLM-BPI-output
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface

export TOKENIZERS_PARALLELISM=false

INPUT_FOLDER=$1
OUTPUT_PATH=$2
TOKENIZER_NAME=$3 
REGEX_FILTER=$4

echo "Starting tokenization for $INPUT_FOLDER"
echo "Output will be saved to $OUTPUT_PATH"

python preprocess_data_for_megatron.py \
    --input=$INPUT_FOLDER \
    --output-prefix=$OUTPUT_PATH \
    --preproc-folder \
    --json-keys=text \
    --dataset-impl=mmap \
    --tokenizer-library=huggingface \
    --tokenizer-model-name=$TOKENIZER_NAME \
    --use-fast \
    --append-eod \
    --apply-ftfy \
    --workers=40 \
    --regex-filter $REGEX_FILTER \
    --log-interval 10000
