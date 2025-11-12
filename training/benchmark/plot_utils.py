import json
import pandas as pd
import os
import glob
import datetime


def load_data(input_folder):
    data = []
    # iterate on all input_folder/**/job_*
    for job_folder in glob.glob(
        os.path.join(input_folder, "**", "job_*"), recursive=True
    ):
        try:
            # Check log.out exists
            log_file = os.path.join(job_folder, "log.out")
            if not os.path.exists(log_file):
                continue
            # Get config
            config_files = glob.glob(os.path.join(job_folder, "config_*.json"))
            if config_files:
                config_file = config_files[0]
            else:
                continue
            with open(config_file, "r") as f:
                config = json.load(f)
            # Calculate stats
            stats_data = get_stats(job_folder)
            if not stats_data:
                continue
            # creation date
            creation_time = os.path.getctime(job_folder)
            creation_date = datetime.datetime.fromtimestamp(creation_time)
            # Process
            data.append(
                dict(
                    creation_date=creation_date,
                    job_id=os.path.basename(job_folder).removeprefix("job_"),
                    **stats_data,
                    **config,
                )
            )
        except Exception:
            pass
    return data


def convert_data(data):
    records = []
    for entry in data:
        try:
            number_of_steps_per_trillion_tokens = 1e12 / (
                entry["data"]["global_batch_size"] * entry["data"]["seq_length"]
            )

            if "error" in entry:
                stats = dict(
                    error=entry["error"],
                    mean_step_timing=entry["error"],
                    training_time=entry["error"],
                    consumed_gpu_hours=entry["error"],
                )
            else:
                stats = dict(
                    mean_step_timing=entry["mean_step_timings"],
                    training_time=entry["mean_step_timings"]
                    * number_of_steps_per_trillion_tokens
                    / (3600 * 24),
                    consumed_gpu_hours=(
                        entry["mean_step_timings"]
                        * number_of_steps_per_trillion_tokens
                        * entry["trainer"]["num_nodes"]
                        * entry["trainer"]["devices"]
                    )
                    / 3600,
                )

            try:
                fp8_recipe = entry["trainer"]["plugins"].get("fp8_recipe", "bf16")
            except AttributeError:
                # entry["trainer"]["plugins"] is probably a list
                fp8_recipe = entry["trainer"]["plugins"][0].get("fp8_recipe", "bf16")

            records.append(
                {
                    "num_nodes": entry["trainer"]["num_nodes"],
                    "num_gpus": entry["trainer"]["num_nodes"]
                    * entry["trainer"]["devices"],
                    **stats,
                    "arch": entry["args"]["arch"],
                    "tp": entry["trainer"]["strategy"]["tensor_model_parallel_size"],
                    "pp": entry["trainer"]["strategy"]["pipeline_model_parallel_size"],
                    "precision": "fp8" if entry["args"]["fp8"] else "bf16",
                    "seq_length": entry["data"]["seq_length"],
                    "cp": entry["trainer"]["strategy"]["context_parallel_size"],
                    "batch_size": entry["data"]["global_batch_size"],
                    "micro_batch_size": entry["data"]["micro_batch_size"],
                    "fp8_recipe": fp8_recipe,
                    "grad_reduce_in_fp32": entry["trainer"]["strategy"]["ddp"][
                        "grad_reduce_in_fp32"
                    ],
                    "sequence_parallel": entry["trainer"]["strategy"][
                        "sequence_parallel"
                    ],
                    "note": "\n" + entry.get("info", ""),
                }
            )
        except Exception as e:
            raise RuntimeError(f"error on {entry}") from e

    df = pd.DataFrame(records)
    return df


def get_stats(output_dir):
    import re
    import numpy as np

    steps = dict()
    pattern = r"iteration (\d+)/\d+.*?train_step_timing in s: ([\d.]+)"
    with open(os.path.join(output_dir, "log.out"), "r") as f:
        log_content = f.read()
    matches = re.findall(pattern, log_content)
    if not matches:
        return None

    iterations = [int(m[0]) for m in matches]
    times = [float(m[1]) for m in matches]
    valid_times = times[5:]  # skip warmup

    steps = {
        "min_iteration": min(iterations),
        "max_iteration": max(iterations),
        # "step_timings": times,
        "step_timings_mean": np.mean(valid_times),
        "step_timings_std": np.std(valid_times),
        "total_time": np.sum(times),
    }
    return steps


def setup_data(input_folder):
    data = load_data(input_folder)
    df = convert_data(data)
    return df
