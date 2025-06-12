#!/bin/bash

export PAGER='less -r'

data_path=$1

python ~/datatrove/src/datatrove/tools/inspect_data.py $data_path -r jsonl -s 1 -l $SCRATCH/labeling_web.tmp 