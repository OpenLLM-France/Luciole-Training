# 1B model

## Phase 1

[Repeeat](../../../data/tokenization/run/chronicles/phase_1/repeats.csv)
[Datamix](../../../data/tokenization/run/chronicles/phase_1/datamix.json)

```bash 
python slurm_launcher.py 
```

```bash
cd train/
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/train --mode phase1 --num_nodes 128 --arch llama1b --slurm_array 3 --email ogouvert@linagora.com
```

## Phase 2

```bash
cd train/
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/train --mode phase2 --num_nodes 128 --arch llama1b --slurm_array 3 --email ogouvert@linagora.com
```

## Annealing

```bash
cd train/
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/train --mode annealing --num_nodes 128 --arch llama1b  --config datamix.json --email ogouvert@linagora.com
```

# 7B model

# 20 B model