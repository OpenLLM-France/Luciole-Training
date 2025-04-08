#!/bin/bash

export PAGER='less -r'

data_path=$OpenLLM_OUTPUT/data/raw_datasets/fineweb2/data/fra_Latn/clusters/cluster_size:1000+

python ~/datatrove/src/datatrove/tools/inspect_data.py $data_path -r jsonl -s 1 -l $SCRATCH/labeling_web.tmp 