# Ablations

Running ablations.

`pip install --user --no-cache-dir  zarr`
`pip install --user --no-cache-dir  python-slugify[unidecode]`

## Estimate training time for 35B tokens on Llama 32 1b

`Number of tokens per step: seq_length (2048) * global_batch_size (512) = 1 048 576`

`Number of steps in order to see 35B: 33 378 steps`

`Time to see 35b tokens on 1 node: (33 378 * 5.3)/3600 = 49h `

| Number of nodes | 1 step | 20B tokens | 35B tokens  | 
|-----------------|--------|------------|-------------|
| 1               | 5.3s   |            | 49h         |
| 2               | 2.7s   |            | 25h         |
| 4               | 1.43s  | 7h34       | 13h15       |

## Language ablations

Example of command training:
```
python slurm_launcher.py --config datamix_dclm_dolmino.json --output_dir test --num_nodes 1 --mode debug
python slurm_launcher.py --config datamix_dclm_dolmino.json --output_dir language_ablations --num_nodes 4 --mode 20b
```

Converting your checkpoints:
```
sbatch convert.slurm $OpenLLM_OUTPUT/ablations/train/languages_ablations/datamix_dclm_dolmino_4n_20b
```