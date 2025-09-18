import torch
import logging
import os
from argparse import ArgumentParser

import nemotron_exporter    # noqa: F401
from nemo.collections import llm

torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_checkpoint(input_path, output_path, arch="llama"):
    if arch == "llama":
        exporter = llm.LlamaModel.exporter("hf", input_path)
    elif arch == "nemotron":
        exporter = llm.NemotronModel.exporter("hf", input_path)
    elif arch == "nemotronh":
        exporter = llm.MambaModel.exporter("hf", input_path)
    exporter.init()
    exporter.apply(output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "input_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=os.getenv("OpenLLM_OUTPUT"),
        help="",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="model.bin",
        help="Path to HF folder",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="llama",
        choices=["llama", "nemotron", "nemotronh"],
    )
    parser.add_argument("--local-rank")
    args = parser.parse_args()

    checkpoint_path = os.path.join(args.input_path, args.input_checkpoint)
    convert_checkpoint(checkpoint_path, args.output_path, args.arch)
