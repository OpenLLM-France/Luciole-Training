import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_ARCHITECTURES = [
    "llama1b",
    "llama3b",
    "llama8b",
    "llama24b",
    "llama70b",
    "mamba1b",
    "mixtral8x7",
    "mambahybrid8b",
    "nemotronh8b",
    "nemotronh47b",
    "nemotron22b",
    "nemotron4b",
    "nemotron8b",
    "nemotron1b",
    "qwen32b",
    "mistral12b",
    # "ablation_llama90m",
    # "ablation_llama210m",
    # "ablation_llama400m",
    # "ablation_llama530m",
    "qwen30ba3b",
]


def to_nb_tokens(x):
    if x in ["debug", "benchmark", "benchmark100", "phase1", "phase2", "annealing"]:
        return x
    x = x.replace("b", " * 1_000_000_000")
    x = x.replace("m", " * 1_000_000")
    try:
        return int(eval(x))
    except Exception as e:
        raise ValueError(
            f"Invalid value for --mode: {x} (expect 'debug' or a number of tokens)"
        ) from e


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
    return loaded_data


def get_data_paths(loaded_data):
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
    return data_paths


def get_tokenizer(loaded_data):
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
    return tokenizer_name


def check_tokenizer(tokenizer_name, base_checkpoint):
    # loaded_data = read_datamix_file(config)
    # data_paths, tokenizer_name, total_tokens = get_data_paths_and_tokenizer(loaded_data)
    if base_checkpoint:
        if os.path.exists(
            os.path.join(base_checkpoint, "context", "tokenizer_name.txt")
        ):
            with open(
                os.path.join(base_checkpoint, "context", "tokenizer_name.txt"), "r"
            ) as f:
                base_model_tokenizer = f.read().strip()
            if tokenizer_name != base_model_tokenizer:
                raise ValueError(
                    f"Datamix tokenizer : {tokenizer_name} and base model tokenizer : {base_model_tokenizer} are different!"
                )
    return None


def serialize_fdl(config):
    import fiddle as fdl

    if isinstance(config, fdl.Buildable):
        result = {
            "__type__": type(config).__name__,
            "__fn_or_cls__": str(config.__fn_or_cls__),
        }
        for k, v in config.__arguments__.items():
            try:
                result[k] = serialize_fdl(v)
            except Exception:
                result[k] = f"<non-serializable: {type(v).__name__}>"
        return result
    elif isinstance(config, (list, tuple)):
        return [serialize_fdl(x) for x in config]
    elif isinstance(config, dict):
        return {k: serialize_fdl(v) for k, v in config.items()}
    elif isinstance(config, (str, int, float, bool, type(None))):
        return config
    else:
        # Fallback for non-serializable objects
        return f"<non-serializable: {type(config).__name__}>"


def save_config(output_dir, args, data_args, recipe):
    import json
    from importlib.metadata import version
    from git import Repo

    recipe_dict = {
        "data": data_args,
        "trainer": serialize_fdl(recipe.trainer),
        "model": serialize_fdl(recipe.model),
        "optim": serialize_fdl(recipe.optim),
        "resume": serialize_fdl(recipe.resume),
        "log": serialize_fdl(recipe.log),
    }
    args_dict = {
        "mode": args.mode,
        "name": args.name,
        "output_dir": args.output_dir,
        "arch": args.arch,
        "fp8": args.fp8,
        "config": args.config,
        "performance_mode": args.performance_mode,
    }
    repo = Repo(".", search_parent_directories=True)
    commit_hash = repo.head.commit.hexsha
    toolkit_version = dict(
        nemo_version=version("nemo_toolkit"), open_llm_training_version=commit_hash
    )
    file_path = os.path.join(output_dir, f"config_{args.name}.json")
    with open(file_path, "w") as jsonfile:
        json_data = {
            **recipe_dict,
            **toolkit_version,
            "args": args_dict,
        }
        json.dump(json_data, jsonfile, indent=2)
    logger.info(f"Config saved to {file_path}")


def save_stats(output_dir, name):
    import re
    import json
    import numpy as np

    steps = dict()
    model_size = dict()
    pattern = r"iteration (\d+)/\d+.*?train_step_timing in s: ([\d.]+)"
    with open(os.path.join(output_dir, "log.out"), "r") as f:
        log_content = f.read()
    iteration_timing = {
        int(match[0]): float(match[1]) for match in re.findall(pattern, log_content)
    }
    mean_list = list(iteration_timing.values())[5:]
    mean = np.mean(mean_list)
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
    with open(os.path.join(output_dir, f"stats_{name}.json"), "w") as jsonfile:
        json_data = {
            **steps,
            **model_size,
        }
        json.dump(json_data, jsonfile, indent=2)


def write_completion(output_dir):
    with open(os.path.join(output_dir, "completed.txt"), "w") as f:
        f.write("")
