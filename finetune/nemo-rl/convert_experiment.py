import yaml
import os
from argparse import ArgumentParser
from nemo_rl.models.megatron.community_import import export_model_from_megatron


def convert_one_checkpoint(
    config, hf_model_name=None, megatron_ckpt_path=None, hf_ckpt_path=None
):
    """Main entry point."""
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    # Use hf_model_name override, if available.
    model_name = hf_model_name if hf_model_name else config["policy"]["model_name"]
    tokenizer_name = config["policy"]["tokenizer"]["name"]
    hf_overrides = config["policy"].get("hf_overrides", {}) or {}

    export_model_from_megatron(
        hf_model_name=model_name,
        input_path=megatron_ckpt_path,
        output_path=hf_ckpt_path,
        hf_tokenizer_path=tokenizer_name,
        hf_overrides=hf_overrides,
        overwrite=True,
    )

    # Copy tokenizer folder if exists
    if os.path.exists(tokenizer_name):
        for file in os.listdir(tokenizer_name):
            src_file = os.path.join(tokenizer_name, file)
            dst_file = os.path.join(hf_ckpt_path, file)
            if os.path.isfile(src_file):
                with open(src_file, "rb") as src_f, open(dst_file, "wb") as dst_f:
                    dst_f.write(src_f.read())

    # Copy Generation config
    src_gen_config_path = os.path.join(
        os.path.dirname(__file__), "generation_config.json"
    )
    dst_gen_config_path = os.path.join(hf_ckpt_path, "generation_config.json")
    with open(src_gen_config_path, "rb") as src_f, open(
        dst_gen_config_path, "wb"
    ) as dst_f:
        dst_f.write(src_f.read())


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "experiment_path",
        type=str,
        default=None,
        help="Path to an experiment",
    )
    parser.add_argument(
        "prefix_name",
        type=str,
        help="Prefix name for the output HuggingFace checkpoints. Format MUST contains: luciole_{{arch}}{{size}} e.g. luciole_nemotron1b_baseline etc.",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    experiment_path = args.experiment_path
    prefix_name = args.prefix_name

    checkpoints_path = os.path.join(experiment_path, "checkpoints")
    assert os.path.exists(
        checkpoints_path
    ), f"Checkpoints path {checkpoints_path} does not exist"

    for step_folder in os.listdir(checkpoints_path):
        step_path = os.path.join(checkpoints_path, step_folder)
        if not os.path.isdir(step_path):
            continue
        config_path = os.path.join(step_path, "config.yaml")
        if not os.path.exists(config_path):
            print(f"Config path {config_path} does not exist, skipping {step_folder}")
            continue
        megatron_ckpt_path = os.path.join(step_path, "policy/weights/iter_0000000")
        if not os.path.exists(megatron_ckpt_path):
            print(
                f"Megatron checkpoint path {megatron_ckpt_path} does not exist, skipping {step_folder}"
            )
            continue
        hf_ckpt_path = os.path.join(
            experiment_path, "huggingface_checkpoints", f"{prefix_name}-{step_folder}"
        )

        convert_one_checkpoint(config_path, None, megatron_ckpt_path, hf_ckpt_path)
