# Data

All about preprocessing datasets.

- processing
- scripts

## Environment setup

### Create environment 
```
module purge
module load anaconda-py3/2024.06
conda create -n datatrove-env python=3.10
conda activate datatrove-env
pip install -r requirements.txt
```

### Clone datatrove
```
git clone https://github.com/linagora-labs/datatrove.git
cd datatrove
git checkout lucie_v2
pip install -e .[io,processing]
```

You can add a hostname in `set_env.sh` and set your `$OpenLLM_OUTPUT` variable. Then you can use `source set_env.sh`.

## Processing Datasets

### Local

For example, to process FineWeb-2, on your local machine:
```
source set_env.sh
python processing/fineweb2 --local
```
It will load only the first 1000 samples of the fineweb2 data

### Ablation

For ablation, use the ablation argument. You can add the code line:
```
from utils import add_sampler_filter
pipeline = add_sampler_filter(pipeline) if args.ablation else pipeline
```
to add sampling step after reading the data.

### On Jeanzay

Similarly on Jean Zay, you can use:
```
source set_env.sh
python processing/fineweb2
```

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
