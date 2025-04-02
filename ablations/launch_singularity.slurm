#!/bin/bash

cwd=$(pwd)

source $cwd/set_env.sh

# set -x

(cd $cwd && python launch_pretraining_singularity.py --data_path /lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped/Wikipedia--fr_text_document --output_path /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output)
