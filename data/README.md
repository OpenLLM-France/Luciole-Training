# Data

All about preprocessing datasets.

## Environment setup

### Create environment 
```
module purge
module load anaconda-py3/2023.09 
conda create -n data-env python=3.10
conda activate data-env
```

### Clone datatrove
```
git clone https://github.com/linagora-labs/datatrove.git
git checkout lucie_v2
cd datatrove
pip install -e .[io,processing]
pip install rich
```