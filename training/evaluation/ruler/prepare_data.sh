#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# container: docker.io/cphsieh/ruler:0.1.0
# bash run.sh MODEL_NAME BENCHMARK_NAME

if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> $1 <benchmark_name>"
    exit 1
fi

RULER_DIR=~/RULER/scripts/

# Root Directories
GPUS="1" # GPU size for tensor_parallel.
ROOT_DIR="${OpenLLM_OUTPUT}/ruler_evaluation" # the path that stores generated task samples and model predictions.
MODEL_DIR="../.." # the path that contains individual model folders from HUggingface.
ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
BATCH_SIZE=1  # increase to improve GPU utilization


# Model and Tokenizer
source config_models.sh
MODEL_NAME=${1}
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_NAME TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi


export OPENAI_API_KEY=${OPENAI_API_KEY}
export GEMINI_API_KEY=${GEMINI_API_KEY}
export AZURE_API_ID=${AZURE_ID}
export AZURE_API_SECRET=${AZURE_SECRET}
export AZURE_API_ENDPOINT=${AZURE_ENDPOINT}


# Benchmark and Tasks
source config_tasks.sh
BENCHMARK=${2}
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

# Add this at the top of your script or pass it as an env variable:
# DATA_PREP_ONLY=true   # or false

total_time=0
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    if [ "${TOKENIZER_NAME}" != "" ]; then
        DATA_DIR="${ROOT_DIR}/data/${TOKENIZER_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    else
        DATA_DIR="${ROOT_DIR}/data/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    fi
    PRED_DIR="${ROOT_DIR}/pred/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}
    
    for TASK in "${TASKS[@]}"; do
        # -------------------------
        # DATA PREPARATION
        # -------------------------
        python ${RULER_DIR}/data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB}
    done
done

echo "Total time spent on call_api: $total_time seconds"
