# Data

All about preprocessing datasets.

## Processing Datasets

### Environment setup

#### Create environment 
```bash
module purge
module load anaconda-py3/2024.06
conda create -n datatrove-env python=3.10
conda activate datatrove-env
pip install -r requirements.txt
```

#### Clone datatrove
```bash
git clone https://github.com/linagora-labs/datatrove.git
cd datatrove
git checkout lucie_v2
pip install -e .[io,processing]
```

You can add a hostname in `set_env.sh` and set your `$OpenLLM_OUTPUT` variable. Then you can use `source set_env.sh`.

### Run Processing

#### Locally

For example, to process FineWeb-2, on your local machine:
```
source set_env.sh
python processing/fineweb2 --local
```
It will load only the first 1000 samples of the fineweb2 data

#### On Jeanzay

Similarly on Jean Zay, you can use:
```bash
source set_env.sh
python processing/fineweb2
```

#### ... for ablation

For ablation, use the ablation argument. You can add the code line:
```python
from utils import add_sampler_filter
pipeline = add_sampler_filter(pipeline) if args.ablation else pipeline
```
to add sampling step (it only processes 5\% of the full dataset) after reading the data.  

## Tokenization

You have some preprocessed datasets in `$OpenLLM_OUTPUT/data/raw_datasets` or in `$OpenLLM_OUTPUT/data/raw_datasets_ablation` and you want to tokenize them...

1. Specify the datasets you want to tokenize in `datasets_to_tokenize.yaml` (you can duplicate and rename this file if you want). There should have two entries for each dataset:
- path: the relative path of the dataset (for example in `$OpenLLM_OUTPUT/data/raw_datasets(_ablation)`)
- name: the associated name of the dataset after tokenization

For example:
```
  - name: fineweb2_fra_Latn_cluster_5-100
    path: fineweb2/data/fra_Latn/clusters/cluster_size-5-100
```

2. Run tokenzation by using the script `run_tokenization.py`
```
usage: run_tokenization.py [-h] [--yaml_file YAML_FILE] [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--tokenizer_name TOKENIZER_NAME]

options:
  -h, --help            show this help message and exit
  --yaml_file YAML_FILE
                        .yaml file that contains the datasets you want to tokenize. See for example datasets_to_tokenize.yaml. (default: datasets_to_tokenize.yaml)
  --input_dir INPUT_DIR
                        Input directory that contains the processed datasets you want to tokenize. (default: $OpenLLM_OUTPUT/data/raw_datasets_ablation)
  --output_dir OUTPUT_DIR
                        Output directory that will contain all your tokenized datasets, with name provided by your yaml file. You cannot use different tokenizer in one output_dir (it will raise an error). (default: $OpenLLM_OUTPUT/data/tokens_ablation)
  --tokenizer_name TOKENIZER_NAME
                        The tokenizer you want to use to tokenize the data. This name will be saved in your output_dir. (default: OpenLLM-France/Lucie-7B)
```
It will create one sbatch per dataset, using prepost partition.

3. Run statistics: 
```bash
sbatch run_statistics.slurm OUTPUT_DIR
```
where `OUTPUT_DIR` is the absolute path of your tokenized datasets.
It will create a folder `stats` in the tokenized data folder, with the statistics of each tokens file.

4. Next step is in [`../ablations`](../ablations/README.md) or in [`../training`](../training/README.md).

## Tips... 

### Download a dataset from HF

Set a common HF cache dir
```bash
export HF_HOME=$qgz_ALL_CCFRSCRATCH/.cache/huggingface
```

Load a dataset with huggingface-cli:
```bash
dataset_name=open-web-math/open-web-math
huggingface-cli download $dataset_name --repo-type dataset 
```

To load a specific subset you can use incluse and/or exclude
```bash
dataset_name=EleutherAI/proof-pile-2
huggingface-cli download $dataset_name --repo-type dataset --include algebraic-stack/*
```

Load a tokenizer:
```bash
huggingface-cli download OpenLLM-BPI/tokenizer_128k_latin --repo-type model
```