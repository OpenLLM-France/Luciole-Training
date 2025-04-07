# Ablations

Running ablations.

`pip install --user --no-cache-dir  zarr`

## Singularity

Make a custom env:

```
module purge
module load anaconda-py3/2023.09 
module load cuda/12.4.1
module load gcc/11.3.1
module load ffmpeg/6.1.1
conda create --prefix /lustre/fsn1/projects/rech/qgz/commun/envs/nemo-env python==3.12.7
conda activate nemo-env
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install Cython packaging
pip install nemo_toolkit['nlp'] # or pip install "git+https://github.com/NVIDIA/NeMo@${REF:-'main'}#egg=nemo_toolkit[nlp]"
pip install git+https://github.com/NVIDIA/NeMo-Run.git
pip install megatron-core
pip install lightning
pip install cloudpickle
pip install fiddle
pip install hydra-core==1.3.2
pip install lightning==2.4.0
pip install omegaconf==2.3
pip install peft
pip install torchmetrics==0.11.0
pip install transformers==4.48.3
pip install wandb
pip install webdataset==0.2.86
pip install datasets
pip install einops
pip install inflect
pip install mediapy==1.1.6
pip install pandas
pip install sacremoses==0.0.43
pip install sentencepiece==0.2.0
pip install typing
```


### Build and use a Singularity image

See [the jean-zay documentation](http://www.idris.fr/jean-zay/cpu/jean-zay-utilisation-singularity.html)

To make the image, go on prepost
```
srun --pty --ntasks=1 --cpus-per-task=16 --hint=nomultithread --account=qgz@cpu --partition=prepost --qos=qos_cpu-dev --time=00:20:00 bash
```
then launch (on scratch or work)
```
singularity build image-singularity-tensorflow.sif docker://nvcr.io/nvidia/nemo:25.02
```
A file is already available at 
```
/lustre/fswork/projects/rech/qgz/commun/image-singularity-nemo.sif
```

You need to copy the file in $SINGULARITY_ALLOWED_DIR (this is a personal folder)
```
idrcontmgr cp /lustre/fswork/projects/rech/qgz/commun/image-singularity-nemo.sif
```
You can see your images with 
```
idrcontmgr ls
```