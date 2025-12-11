import os
from argparse import ArgumentParser
from tqdm import tqdm
import re


def convert_checkpoint_folder(input_path, output_path, arch, multiple_of=None, last=False):
    checkpoints = sorted(os.listdir(os.path.join(input_path, "checkpoints")))
    os.makedirs(os.path.join(output_path), exist_ok=True)
    import convert_dist_to_hf

    if last:
        checkpoints = checkpoints[-1:]

    for checkpoint in tqdm(
        checkpoints,
        total=len(checkpoints),
        desc=f"Converting {os.path.basename(input_path)}",
    ):
        checkpoint_path = os.path.join(input_path, "checkpoints", checkpoint)
        checkpoint_output_path = os.path.join(output_path, checkpoint).replace("=", "_").replace("-last", "")
        # print("Output to", checkpoint_output_path)
        if os.path.isfile(checkpoint_path):
            print("\nSkipping file", checkpoint)
            continue
        if checkpoint.endswith("-last") and multiple_of is None and not last:
            print("\nSkipping last", checkpoint)
            continue
        if os.path.isfile(checkpoint_path + "-unfinished"):
            print("\nSkipping unfinished", checkpoint)
            continue
        if multiple_of is not None:
            num_step = get_step(checkpoint)
            if num_step and ((num_step + 1) % multiple_of == 0):
                pass
            else:
                print(
                    f"Skipping {checkpoint} because it is not a multiple of {multiple_of}"
                )
                continue
        convert_dist_to_hf.convert_checkpoint(
            checkpoint_path, checkpoint_output_path, arch=arch
        )


def get_step(text):
    match = re.search(r"-step=(\d+)", text)
    if match:
        step_number = int(match.group(1))
        return step_number
    else:
        return None


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "experiment_path",
        type=str,
        default=None,
        help="Path to an experiment",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="llama",
        choices=["llama", "nemotron", "nemotronh"],
    )
    parser.add_argument(
        "--multiple_of",
        type=int,
        default=1,
        help="Convert only checkpoints that are multiple of this value",
    )
    parser.add_argument("--last", action="store_true")
    parser.add_argument("--local-rank")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    experiment_path = args.experiment_path

    import torch
    from nemo.utils import logging
    import logging  # noqa: F811

    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
    torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Converting experiment {experiment_path}")

    xp_name = os.path.basename(experiment_path)
    xp_path = os.path.join(experiment_path, xp_name)
    xp_output_path = os.path.join(experiment_path, "huggingface_checkpoints")

    if not os.path.exists(xp_path):
        raise FileNotFoundError(
            f"No {xp_name} in {experiment_path}, please put the root experiment folder"
        )

    if os.path.exists(os.path.join(xp_path, "checkpoints")):
        convert_checkpoint_folder(xp_path, xp_output_path, args.arch, args.multiple_of, args.last)
    else:
        runs = os.listdir(xp_path)
        for run in runs:
            run_path = os.path.join(xp_path, run)
            if os.path.exists(os.path.join(run_path, "checkpoints")):
                run_output_path = os.path.join(xp_output_path, run)
                convert_checkpoint_folder(run_path, run_output_path, args.arch, args.multiple_of, args.last)

    logger.info(f"Finished converting {experiment_path}!")
