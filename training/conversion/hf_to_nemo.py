import torch
import logging
import os
from argparse import ArgumentParser

from nemo.collections import llm

torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_checkpoint(input_path, output_path):
    exporter = llm.LlamaModel.importer(input_path)
    exporter.init()
    exporter.apply(output_path)
    with open(os.path.join(output_path, "context", "tokenizer_name.txt"), "w") as f:
        f.write(input_path.replace("hf://", ""))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--hf_model",
        type=str,
        default="hf://meta-llama/Llama-3.2-1B",
        help="Path to a hf model",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(os.getenv("OpenLLM_OUTPUT"), "hf_models", "model.nemo"),
        help="Path to nemo ckpt",
    )
    parser.add_argument("--local-rank")
    args = parser.parse_args()

    convert_checkpoint(args.hf_model, args.output_path)
