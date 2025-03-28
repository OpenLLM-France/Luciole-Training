# Data

All about preprocessing datasets.

- processing
- scripts

## Environment setup

### Create environment 
```
module purge
module load anaconda-py3/2023.09 
conda create -n datatrove-env python=3.12
conda activate datatrove-env
```

### Clone datatrove
```
git clone https://github.com/linagora-labs/datatrove.git
cd datatrove
git checkout lucie_v2
pip install -e .[io,processing]
pip install rich
pip install matplotlib
pip install spacy
pip install slugify
```

You can add a hostname in `set_env.sh` and set your `$DATA`. Then you can use `source set_env.sh`.

## Processing Datasets

### Local

For example, to process FineWeb-2, on your local machine:
```
source set_env.sh
python processing/fineweb2 --local
```
It will load only the first 1000 samples of the fineweb2 data

### On Jeanzay

Similarly on Jean Zay, you can use:
```
source set_env.sh
python processing/fineweb2
```
Use the `--debug` flag if you want to do some tests (it will use only one task with a dev qos.)

## Useful tools and script

See code in `scripts/`

## Tips

### Download a dataset from HF

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
