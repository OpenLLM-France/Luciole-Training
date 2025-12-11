# Data

All about preprocessing datasets.

## Processing Datasets

### Environment setup

#### Create environment 
```bash
module purge
module load arch/h100 
module load anaconda-py3/2024.06
module load cuda/12.4.1
conda create -n datatrove-env python=3.10
conda activate datatrove-env
pip install -r requirements.txt
# pip install lighteval[extended_tasks,math,multilingual]
```

#### Clone datatrove
```bash
git clone https://github.com/linagora-labs/datatrove.git
cd datatrove
git checkout lucie_v2
pip install -e .[io,processing,inference]
pip install vllm
pip install --no-build-isolation flash-attn
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
dataset_groups:
  - root_path: <<OpenLLM_OUTPUT>>/data/raw_data/data_for_ablation
    datasets:
      - name: fineweb2_fra_Latn_cluster_5-100
        path: fineweb2/data/fra_Latn/clusters/cluster_size-5-100
```

2. Run tokenzation by using the script `run/run_tokenization.py`
```bash
run_tokenization.py YAML_FILE OUTPUT_DIR --tokenizer_name OpenLLM-BPI/tokenizer_128k-arab-regional_v2
```
It will create one sbatch per dataset.

3. Run statistics: 
```bash
sbatch run_statistics.slurm OUTPUT_DIR
```
where `OUTPUT_DIR` is the absolute path of your tokenized datasets.
It will create a folder `stats` in the tokenized data folder, with the statistics of each tokens file.

Then add your datasets in `link_datasets.sh`
and run it to create symbolic links in a common folder.
Then
```bash
python merge_stats.py OUTPUT_DIR
```
and to visualize
```bash
python visualize_token_stats.py

```

4. Next step is in [`../ablations`](../ablations/README.md) or in [`../training`](../training/README.md).

## Tips... 

### Pre-download a dataset or a tokenizer from HuggingFace

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
huggingface-cli download OpenLLM-BPI/tokenizer_128k-arab-regional --repo-type model
```
