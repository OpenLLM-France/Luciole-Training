### Install env

```
module load arch/h100 
module load anaconda-py3/2024.06
module load cuda/12.4.1
<!-- module load gcc/11.4.1 -->
conda create -n distilabel-env python=3.12
conda activate distilabel-env
pip install --force-reinstall distilabel[vllm,hf-transformers] 
pip install flash-attn --no-build-isolation
pip install hf_xet
pip install pynvml
pip install bs4
pip install scikit-learn
pip install matplotlib
```

### Use it

```
module load arch/h100 
module load anaconda-py3/2024.06
module load cuda/12.4.1
conda activate distilabel-env
```
