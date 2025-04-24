# Evaluation

All about evaluation

## Installation

```
module purge
module load anaconda-py3/2024.06
conda create -n eval-env python=3.10
conda activate eval-env
pip install lighteval[extended_tasks,math,multilingual]
pip install hf-xet
```
