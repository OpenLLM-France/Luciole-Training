import torch
import logging
import os
from argparse import ArgumentParser

from nemo.collections import llm

torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_checkpoint(input_path, output_path):
    exporter = llm.LlamaModel.exporter("hf", input_path)
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
    parser.add_argument("--local-rank")
    args = parser.parse_args()

    checkpoint_path = os.path.join(args.input_path, args.input_checkpoint)
    convert_checkpoint(checkpoint_path, args.output_path)
