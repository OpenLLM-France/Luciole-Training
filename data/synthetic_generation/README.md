### Install env

```
module load arch/h100 
module load anaconda-py3/2024.06
module load cuda/12.4.1
conda create -n distilabel-env python=3.12
conda activate distilabel-env
pip install -U distilabel[vllm]
pip install flash-attn --no-build-isolation
pip install pynvml
pip install hf_xet
pip install bs4
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install fasttext
```

### Use it

```
module load arch/h100 
module load anaconda-py3/2024.06
module load cuda/12.4.1
conda activate distilabel-env
```
