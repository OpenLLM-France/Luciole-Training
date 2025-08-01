# Ablations

## Ablation pipeline

```
cd training/ablation
python regmix_datamix.py french_datamix --regex ".*fr" --max 5
bash ablation.sh french_datamix fineweb2.txt
cd ../evaluation
python analyze_results.py $OpenLLM_OUTPUT/ablations/train/regmix/french_datamix
cd ../ablation
python fit_exp_law.py --dir /lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/train/regmix/french_datamix
```

Low correlation is better than high correlation. Tasks evaluate the perplexity and we are interested in the log proba so we want to minimize the correlation. See https://arxiv.org/abs/2403.16952
Plot made by fit_exp_law show the model weights matrix T (see figure 4 in the paper and equation 1)

## Install requirements

You should create a new environment for evaluation.
```bash
module purge
module load anaconda-py3/2024.06
conda create -n eval-env python=3.10
conda activate eval-env
pip install -U lighteval[multilingual,vllm]
pip install seaborn
pip install "datasets<4.0.0"
```