import torch
import logging
import os
from argparse import ArgumentParser
from tqdm import tqdm

import convert_dist_to_llama

torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_checkpoint_folder(input_path, ouput_path):
    checkpoints = os.listdir(os.path.join(input_path, "checkpoints"))
    os.makedirs(os.path.join(ouput_path), exist_ok=True)

    for checkpoint in tqdm(checkpoints, desc=f"Converting run {input_path}"):
        checkpoint_path = os.path.join(input_path, "checkpoints", checkpoint)
        checkpoint_output_path = os.path.join(ouput_path, checkpoint)
        convert_dist_to_llama.convert_checkpoint(
            checkpoint_path, checkpoint_output_path
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "experiment",
        type=str,
        default=None,
        help="Path to an experiment",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=os.getenv("OpenLLM_OUTPUT"),
        help="",
    )
    parser.add_argument("--local-rank")
    args = parser.parse_args()

    experiment_path = os.path.join(args.input_path, args.experiment)

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
            run_output_path = os.path.join(xp_output_path, run)
            convert_checkpoint_folder(run_path, run_output_path)

    logger.info(f"Finished converting {experiment_path}!")
