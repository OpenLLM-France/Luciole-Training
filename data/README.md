# Data

All about preprocessing datasets.

## Environment setup

### Create environment 
```
module purge
module load anaconda-py3/2023.09 
conda create -n datatrove-env python=3.10
conda activate datatrove-env
```

### Clone datatrove
```
git clone https://github.com/linagora-labs/datatrove.git
git checkout lucie_v2
cd datatrove
pip install -e .[io,processing]
pip install rich
pip install matplotlib
pip install spacy
```

## Download a dataset from HF

Set a common HF cache dir
```
export HF_HOME=$ALL_CCFRSCRATCH/.cache/huggingface
```

Load a dataset with huggingface-cli:
```
dataset_name=open-web-math/open-web-math
huggingface-cli download $dataset_name --repo-type dataset 
```

To load a specific subset you can use incluse and/or exclude
```
dataset_name=EleutherAI/proof-pile-2
huggingface-cli download $dataset_name --repo-type dataset --include algebraic-stack/*
```
