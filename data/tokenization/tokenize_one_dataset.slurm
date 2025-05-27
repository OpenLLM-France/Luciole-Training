#!/bin/bash
#SBATCH --output=slurm/%x.out 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --time=10:00:00 
#SBATCH --hint=nomultithread 
#SBATCH --account=qgz@cpu
#SBATCH --partition=prepost

# Requires nemo installation - on Jeanzay for now
module purge
module load arch/h100 nemo/2.1.0
export TOKENIZERS_PARALLELISM=false

INPUT_FOLDER=$1
OUTPUT_PATH=$2
TOKENIZER_NAME=$3 

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
    --workers=25 \
    --log-interval 10000
