# Evaluation

All about evaluation

## Installation
git clone https://github.com/hynky1999/lighteval.git
cd lighteval
conda create -n lighteval python=3.10 && conda activate lighteval
git checkout new-multi-lang-branch
pip installv -e  '.[accelerate]'
```
module purge
module load anaconda-py3/2024.06
conda create -n eval-env python=3.10
conda activate eval-env
pip install lighteval[extended_tasks,math]
```
