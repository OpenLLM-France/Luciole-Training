import torch
from nemo.utils import logging
import logging  # noqa: F811
import os
from argparse import ArgumentParser
from tqdm import tqdm

import convert_dist_to_llama

for name in logging.root.manager.loggerDict:
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
torch.set_float32_matmul_precision("high")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_checkpoint_folder(input_path, ouput_path):
    checkpoints = os.listdir(os.path.join(input_path, "checkpoints"))
    os.makedirs(os.path.join(ouput_path), exist_ok=True)

    for checkpoint in tqdm(
        checkpoints,
        total=len(checkpoints),
        desc=f"Converting {os.path.basename(input_path)}",
    ):
        checkpoint_path = os.path.join(input_path, "checkpoints", checkpoint)
        print("\nProcessing", checkpoint_path)
        checkpoint_output_path = os.path.join(ouput_path, checkpoint).replace("=", "_")
        print("Output to", checkpoint_output_path)
        if os.path.isfile(checkpoint_path):
            print("Skipping file", checkpoint)
            continue
        if checkpoint.endswith("-last"):
            print("Skipping", checkpoint)
            continue
        if os.path.isfile(checkpoint_path + "-unfinished"):
            print("Skipping unfinished", checkpoint)
            continue
        if os.path.exists(checkpoint_output_path):
            print("Skipping existing", checkpoint_output_path)
            continue
        else:
            convert_dist_to_llama.convert_checkpoint(
                checkpoint_path, checkpoint_output_path
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "experiment_path",
        type=str,
        default=None,
        help="Path to an experiment",
    )
    parser.add_argument("--local-rank")
    parser.add_argument("--no_completion", action="store_true")
    args = parser.parse_args()
    experiment_path = args.experiment_path

    logger.info(f"Converting experiment {experiment_path}")

    xp_name = os.path.basename(experiment_path)
    xp_path = os.path.join(experiment_path, xp_name)
    xp_output_path = os.path.join(experiment_path, "huggingface_checkpoints")

    if not os.path.exists(xp_path):
        raise FileNotFoundError(
            f"No {xp_name} in {experiment_path}, please put the root experiment folder"
        )

    if os.path.exists(os.path.join(xp_path, "checkpoints")):
        convert_checkpoint_folder(xp_path, xp_output_path)
    else:
        runs = os.listdir(xp_path)
        for run in runs:
            run_path = os.path.join(xp_path, run)
            if os.path.exists(os.path.join(run_path, "checkpoints")):
                run_output_path = os.path.join(xp_output_path, run)
                convert_checkpoint_folder(run_path, xp_output_path)

    logger.info(f"Finished converting {experiment_path}!")

    if not args.no_completion:
        with open(
            os.path.join(experiment_path, "conversion", "completed.txt"), "w"
        ) as f:
            f.write("")
