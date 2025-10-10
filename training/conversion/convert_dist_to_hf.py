import torch
import logging
import os
from argparse import ArgumentParser
import shutil
import json
import custom_exporter
import custom_exporter.nemotron_exporter  # noqa: F401
import custom_exporter.nemotronh_exporter  # noqa: F401
from nemo.collections import llm

torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_checkpoint(input_path, output_path, arch="llama"):
    if os.path.exists(output_path):
        # Skip conversion if existing
        print("\nSkipping existing", output_path)
    else:
        # Convert
        print("\nProcessing", input_path)
        print("Output to", output_path)
        if arch == "llama":
            exporter = llm.LlamaModel.exporter("hf", input_path)
        elif arch == "nemotron":
            exporter = llm.NemotronModel.exporter("hf", input_path)
        elif arch == "nemotronh":
            exporter = llm.MambaModel.exporter("hf", input_path)
        exporter.init()
        exporter.apply(output_path)
    # Nemotron-H specific
    if arch == "nemotronh":
        # Modify config.json
        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        if "auto_map" not in config:
            config["auto_map"] = {
                "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
                "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
            }
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            pass
        # Copy config and modeling files
        custom_exporter_path = os.path.dirname(custom_exporter.__file__)
        if not os.path.exists(os.path.join(output_path, "configuration_nemotron_h.py")):
            shutil.copy(
                os.path.join(custom_exporter_path, "configuration_nemotron_h.py"),
                os.path.join(output_path, "configuration_nemotron_h.py"),
            )
        if not os.path.exists(os.path.join(output_path, "modeling_nemotron_h.py")):
            shutil.copy(
                os.path.join(custom_exporter_path, "modeling_nemotron_h.py"),
                os.path.join(output_path, "modeling_nemotron_h.py"),
            )


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
