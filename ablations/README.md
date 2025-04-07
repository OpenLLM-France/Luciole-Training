# Ablations

Running ablations.

`pip install --user --no-cache-dir  zarr`

## Singularity

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