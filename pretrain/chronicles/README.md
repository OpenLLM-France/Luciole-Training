# Environmental Impact

This directory estimates the environmental impact of training the Luciole series of multilingual language models, trained on the [Jean-Zay](http://www.idris.fr/jean-zay/) HPC cluster (NVIDIA H100 GPUs) as part of the [OpenLLM-France](https://github.com/OpenLLM-France) initiative.

## Data Sources

Energy consumption and carbon emissions data are collected from Jean-Zay's SLURM accounting system, which tracks per-job metrics including energy usage and CO2 estimates.

## Methodology

We aggregated per-job metrics into per-project totals:

- **Walltime**: the sum of elapsed wall-clock time across all jobs.
- **GPU Time**: for each job, the walltime multiplied by the number of allocated nodes (derived from the SLURM `Nodelist` field). This represents the total node-hours consumed.
- **Energy**: reported in kWh, split into three components:
  - **Total Energy** = Host Energy + GPU Energy
  - **Host Energy**: energy consumed by the host components (CPUs, RAM, network, storage, cooling).
  - **GPU Energy**: energy consumed by the GPUs.
- **Carbon Emissions**: reported in grams of CO2-equivalent, using three different carbon intensity sources:
  - **RTE**: based on real-time carbon intensity data from [RTE](https://www.rte-france.com/) (the French electricity transmission operator).
  - **EMaps**: based on [Electricity Maps](https://www.electricitymaps.com/) carbon intensity data.
  - **OWID**: based on [Our World in Data](https://ourworldindata.org/energy) country-level averages.

  Each emission metric is further decomposed into Host and GPU contributions.

## Results

The aggregated results are in [`consumption.csv`](consumption.csv), sorted by decreasing total energy consumption. Job types are:

- **Training 1B / 8B / 23B**: main pretraining runs for the 1B, 8B and 23B parameter Luciole models.
- **Expes for 1B / 8B / 23B**: additional runs that were not part of the main pretraining.
- **Ablation Experiments**: controlled experiments to validate data choices.
- **Time Benchmark**: performance benchmarking runs.
