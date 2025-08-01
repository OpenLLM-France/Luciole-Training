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

## Install requirements

You should create a new environment for evaluation.
```bash
module purge
module load anaconda-py3/2024.06
conda create -n eval-env python=3.10
conda activate eval-env
pip install -U lighteval[multilingual,vllm]
pip install seaborn 
```