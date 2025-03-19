#!/bin/bash

export PAGER='less -r'

data_path=$DATA/fineweb2/data/fra_Latn/clusters/cluster_size:1000+
data_path=$DATA/algebraic_stack/output

python ~/datatrove/src/datatrove/tools/inspect_data.py $data_path -r jsonl -s 1 -l $SCRATCH/labeling_web.tmp 