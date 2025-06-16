import os
import torch
import logging
import torch.distributed as dist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def read_datamix_file(file):
    loaded_data = None
    if file.endswith(".json"):
        import json

        with open(file, "r") as f:
            loaded_data = json.load(f)
    elif file.endswith(".yaml"):
        import yaml

        with open(file, "r") as f:
            loaded_data = yaml.safe_load(f)
    elif file.endswith(".bin"):
        loaded_data = {
            "data_path": os.path.dirname(file),
            "train": [
                {
                    "name": os.path.splitext(os.path.basename(file))[0],
                    "weight": 1.0,
                }
            ],
        }
    else:
        raise RuntimeError(f"Config should be a json or a yaml, got {file}")

    def make_data_flattened_list(split="train"):
        data_paths = []
        for dataset in loaded_data.get(split, []):
            data_paths.append(str(dataset["weight"]))
            data_paths.append(os.path.join(loaded_data["data_path"], dataset["name"]))
        return data_paths

    logger.info(loaded_data)
    if "validation" in loaded_data:
        data_paths = {
            "train": make_data_flattened_list("train"),
            "validation": make_data_flattened_list("validation"),
            "test": make_data_flattened_list("validation"),
        }
    else:
        data_paths = make_data_flattened_list("train")
    # logger.info(">>>>>>>>>>>")
    # logger.info(data_paths)

    # Read tokenizer
    try:
        with open(
            os.path.join(loaded_data["data_path"], "tokenizer_name.txt"), "r"
        ) as f:
            tokenizer_name = f.read().strip()
            logger.info(f"Find tokenizer: {tokenizer_name}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"tokenizer_name.txt not found in {loaded_data['data_path']}. Please rerun the tokenization step."
        )
    return data_paths, tokenizer_name


def save_stats(output_dir, args, strategy_args, data_args, write_step_timings=True):
    import re
    import json

    strategy_args.pop("ddp")
    strategy_args["pipeline_dtype"] = str(strategy_args["pipeline_dtype"])
    job_id = os.environ.get("SLURM_JOB_ID")
    steps = dict()
    model_size = dict()
    if write_step_timings:
        pattern = r"iteration (\d+)/\d+.*?train_step_timing in s: ([\d.]+)"
        file = f"log_{job_id}.out"
        with open(os.path.join(output_dir, file), "r") as f:
            log_content = f.read()
        iteration_timing = {
            int(match[0]): float(match[1]) for match in re.findall(pattern, log_content)
        }
        mean = sum(list(iteration_timing.values())[2:]) / (len(iteration_timing) - 2)
        steps = {
            "step_timings": list(iteration_timing.values()),
            "mean_step_timings": mean,
        }

        pattern = r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[MB])\s+(?P<label>Trainable params|Total params)"

        matches = re.findall(pattern, log_content)

        model_size = {
            label: float(value) if unit == "B" else float(value) / 1000
            for value, unit, label in matches
        }
    with open(
        os.path.join(output_dir, f"stats_{args['name']}_{job_id}.json"), "w"
    ) as jsonfile:
        json_data = {**args, **data_args, **strategy_args, **steps, **model_size}
        json.dump(json_data, jsonfile, indent=2)


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def write_completion(output_dir):
    with open(os.path.join(output_dir, "completed.txt"), "w") as f:
        f.write("")


def suppress_non_main_logging():
    """Sets all existing and future loggers to only log from the main process."""
    dist.init_process_group(backend="nccl")
    if is_main_process():
        logging_level = logging.INFO
    else:
        logging_level = logging.CRITICAL + 1  # Effectively disables logging

    logging.getLogger().setLevel(logging.CRITICAL)

    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(logging_level)
