import torch
import logging
import os
from argparse import ArgumentParser

from nemo.collections import llm

torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_checkpoint(input_path, output_path, arch):
    logger.info(f"Converting {input_path} to {output_path}")
    if arch == "llama":
        importer = llm.LlamaModel.importer(input_path)
    elif arch == "nemotron":
        importer = llm.NemotronModel.importer(input_path)
    elif arch == "nemotronh":
        importer = llm.MambaModel.importer(input_path)
    importer.init()
    importer.apply(output_path)
    with open(os.path.join(output_path, "context", "tokenizer_name.txt"), "w") as f:
        f.write(input_path.replace("hf://", ""))
    logger.info(f"Model converted to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--hf_model",
        type=str,
        default="hf://OpenLLM-France/Lucie-7B-Instruct-v1.1",
        help="Path to a hf model id. If you use a local path to a hf model use: hf:///path/to/model/Lucie-7B-Instruct-v1.1",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(os.getenv("OpenLLM_OUTPUT"), "hf_models", "model"),
        help="Path to nemo ckpt",
    )
    parser.add_argument(
        "--nemo_file", action="store_true", help="Will generate a .nemo file"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="llama",
        choices=["llama", "nemotron", "nemotronh"],
        help="Model architecture",
    )
    parser.add_argument("--local-rank")
    args = parser.parse_args()

    convert_checkpoint(args.hf_model, args.output_path, args.arch)

    if args.nemo_file:
        import tarfile

        with tarfile.open(args.output_path + ".nemo", "w") as tar:
            tar.add(args.output_path, arcname="context")
            tar.add(args.output_path, arcname="weights")
        logger.info(f"Nemo file written to {args.output_path+'.nemo'}")
