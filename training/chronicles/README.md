# 1B model

## OLMO 2 Baseline

Evaluate
```bash
cd evaluation/

module load anaconda-py3/2024.06
conda activate eval-env
path=/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/pretrain/OLMo-2-0425-1B

python evaluate_experiment.py $path --olmo2 tasks/en.txt --command accelerate
python evaluate_experiment.py $path --olmo2 tasks/fr.txt --custom_tasks multilingual --command accelerate
```

## Phase 1

[Repeeat](../../../data/tokenization/run/chronicles/phase_1/repeats.csv)
[Datamix](../../../data/tokenization/run/chronicles/phase_1/datamix.json)

```bash
cd train/
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix luciole --mode phase1 --num_nodes 128 --arch llama1b --email ogouvert@linagora.com
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix luciole --mode phase1 --time 16:00:00 --num_nodes 128 --arch llama1b --email ogouvert@linagora.com
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix luciole --mode phase1 --time 12:00:00 --num_nodes 128 --arch llama1b --email ogouvert@linagora.com
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix luciole --mode phase1 --time 48:00:00 --num_nodes 64 --arch llama1b --email ogouvert@linagora.com
```

Convert
```bash
cd conversion/
path=/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/pretrain/luciole_llama1b
sbatch convert.slurm $path --no_completion
python rope_scaling_correction.py 
```

Evaluate
```bash
cd evaluation/

module load anaconda-py3/2024.06
conda activate eval-env
path=/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/pretrain/luciole_llama1b

python evaluate_experiment.py $path tasks/en.txt --command accelerate
python evaluate_experiment.py $path tasks/fr.txt --custom_tasks multilingual --command accelerate
```

Plot results
```bash
cd evaluation/

module load anaconda-py3/2024.06
conda activate eval-env
path=/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/pretrain/luciole_llama1b
olmo=/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/pretrain/OLMo-2-0425-1B
python plot_results.py $path $olmo --group agg --output_path $path/figs --seq_length 4096
python plot_results.py $path $olmo --group fr --output_path $path/figs --seq_length 4096
python plot_results.py $path $olmo --group en --output_path $path/figs --seq_length 4096
```

## Phase 2

```bash
cd train/
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix luciole --mode phase2 --num_nodes 128 --arch llama1b --email ogouvert@linagora.com
```

## Annealing

```bash
cd train/
python slurm_launcher.py --output_path $OpenLLM_OUTPUT/pretrain --name_prefix luciole --mode annealing --num_nodes 128 --arch llama1b --config datamix.json --email ogouvert@linagora.com
```

# 7B model

# 20 B model