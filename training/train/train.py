import argparse
import torch
import logging

from recipe_llama import (
    create_trainer,
    distributed_fused_adam_with_cosine_annealing,
    create_logger,
    create_autoresume,
)
from utils import (
    read_datamix_file,
)

from dataloader import create_data

from nemo.collections import llm
from nemo.utils.exp_manager import TimingCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    def to_nb_tokens(x):
        if x == "debug" or x == "benchmark":
            return x
        x = x.replace("b", " * 1_000_000_000")
        x = x.replace("m", " * 1_000_000")
        try:
            return int(eval(x))
        except Exception as e:
            raise ValueError(
                f"Invalid value for --mode: {x} (expect 'debug' or a number of tokens)"
            ) from e

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--arch",
        default="llama1b",
        type=str,
        choices=["llama1b", "llama8b", "mamba1b", "mixtral8x7", "mambahybrid8b"],
    )
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--num_gpus_per_node", default=4, type=int)
    parser.add_argument("--mode", default="debug", type=to_nb_tokens)
    parser.add_argument(
        "--output_dir",
        default="/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/train",
    )
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--seq_length", default=None, type=int)
    parser.add_argument("--tensor_parallelism", default=None, type=int)
    parser.add_argument("--pipeline_parallelism", default=None, type=int)
    parser.add_argument("--context_parallelism", default=1, type=int)
    parser.add_argument("--virtual_pipeline_parallelism", default=None, type=int)
    parser.add_argument("--fp8", default=False, action="store_true")
    args = parser.parse_args()

    arch = args.arch
    num_nodes = args.num_nodes
    name = args.name
    output_dir = args.output_dir

    data_paths, tokenizer_name = read_datamix_file(args.config)

    batch_size = args.batch_size
    seq_length = args.seq_length

    if batch_size is None and seq_length is None:
        if arch == "llama1b" or arch == "mamba1b":
            batch_size = 1024
            seq_length = 2048
        elif arch == "llama8b" or arch == "mambahybrid8b":
            batch_size = 1024
            seq_length = 4096
        elif arch == "mixtral8x7":
            batch_size = 512
            seq_length = 4096
        else:
            raise ValueError(f"Unsupported model : {arch}")
    elif batch_size is None:
        batch_size = 4_194_304 // seq_length
    elif seq_length is None:
        seq_length = 4_194_304 // batch_size

    pipeline_parallelism = args.pipeline_parallelism if args.pipeline_parallelism else 1
    tensor_parallelism = args.tensor_parallelism if args.tensor_parallelism else 1

    data = create_data(
        data_paths,
        tokenizer_name=tokenizer_name,
        batch_size=batch_size,
        seq_length=seq_length,
    )

    if args.mode in ["debug", "benchmark"]:
        max_steps = 1 if args.mode == "debug" else 10
        resume_if_exists = False
        every_n_train_steps = max_steps
    else:
        number_of_tokens = args.mode
        max_steps = number_of_tokens // (seq_length * batch_size)
        resume_if_exists = True
        every_n_train_steps = 1_000_000_000 // (seq_length * batch_size)

    logger.info(f"Job name: {args.name}")
    logger.info(f"Architecture: {arch}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Sequence length: {seq_length}")
    logger.info(f"Number of nodes: {args.num_nodes}")
    logger.info(f"Tokenizer: {tokenizer_name}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Saving checkpoints every {every_n_train_steps} train steps")
    logger.info(f"Resume training if possible: {resume_if_exists}")
    logger.info(f"Tensor_parallelism: {tensor_parallelism}")
    logger.info(f"Pipeline_parallelism: {pipeline_parallelism}")

    expert_parallelism = None
    virtual_pipeline_parallelism = args.virtual_pipeline_parallelism
    optimizer_warmup_steps = 2000
    if arch.startswith("llama"):
        if arch == "llama1b":
            from nemo.collections.llm.gpt.model.llama import (
                Llama32Config1B as LlamaConfig,
            )

            optimizer_warmup_steps = 500
        elif arch == "llama8b":
            from nemo.collections.llm.gpt.model.llama import (
                Llama31Config8B as LlamaConfig,
            )
        else:
            raise ValueError(f"Unsupported llama model : {arch}")
        model_config = LlamaConfig()
        model = llm.LlamaModel(model_config, tokenizer=data.tokenizer)
    elif arch.startswith("mamba"):
        if arch == "mamba1b":
            from nemo.collections.llm.gpt.model.ssm import (
                BaseMambaConfig1_3B as MambaConfig,
            )

            model_config = MambaConfig(
                tokenizer_library="huggingface",
                tokenizer_name=tokenizer_name,
                share_embeddings_and_output_weights=True,
            )
        elif arch == "mambahybrid8b":
            from nemo.collections.llm.gpt.model.ssm import (
                NVIDIAMambaHybridConfig8B as MambaConfig,
            )

            model_config = MambaConfig(
                tokenizer_library="huggingface",
                tokenizer_name=tokenizer_name,
                hybrid_override_pattern="*-".join(["M-" * 5] * 5),
                num_layers=58,
            )
            tensor_parallelism = 4
        model = llm.GPTModel(model_config, tokenizer=data.tokenizer)
    elif arch == "mixtral8x7":
        from nemo.collections.llm.gpt.model.mixtral import (
            MixtralConfig8x7B,
            MixtralModel,
        )

        virtual_pipeline_parallelism = 8  # 8
        pipeline_parallelism = 4  # 4
        expert_parallelism = 8  # 8

        model_config = MixtralConfig8x7B()
        model = MixtralModel(model_config, tokenizer=data.tokenizer)
    else:
        raise NotImplementedError(f"Architecture {arch} not implemented")

    opt = distributed_fused_adam_with_cosine_annealing(
        max_lr=3e-4, warmup_steps=optimizer_warmup_steps
    )

    trainer = create_trainer(
        tensor_parallelism=tensor_parallelism,
        pipeline_parallelism=pipeline_parallelism,
        pipeline_parallelism_type=torch.bfloat16,
        context_parallelism=args.context_parallelism,
        virtual_pipeline_parallelism=virtual_pipeline_parallelism,
        max_steps=max_steps,
        num_gpus_per_node=args.num_gpus_per_node,
        num_nodes=num_nodes,
        callbacks=[TimingCallback()],
        val_check_interval=5 if args.mode in ["debug", "benchmark"] else 1000,
        limit_val_batches=0.0,  # 1 if args.mode == "debug" else 0,
        fp8=args.fp8,
        expert_parallelism=expert_parallelism,
    )

    nemo_logger = create_logger(
        dir=output_dir,
        name=name,
        every_n_train_steps=every_n_train_steps,
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer="data",
        optim=opt,
        resume=create_autoresume(resume_if_exists=resume_if_exists),
    )

    if args.mode in ["debug", "benchmark"]:
        import os
        import re
        import json

        files = os.listdir(output_dir)
        pattern = r"iteration (\d+)/\d+.*?train_step_timing in s: ([\d.]+)"
        for file in files:
            if file.startswith("log_"):
                with open(os.path.join(output_dir, file), "r") as f:
                    log_content = f.read()
                iteration_timing = {
                    int(match[0]): float(match[1])
                    for match in re.findall(pattern, log_content)
                }
                mean = sum(list(iteration_timing.values())[2:]) / (
                    len(iteration_timing) - 2
                )
                log_id = file.replace("log_", "")
                log_id = log_id.replace(".out", "")
                with open(
                    os.path.join(output_dir, f"stats_{name}_{log_id}.json"), "w"
                ) as jsonfile:
                    json_data = {
                        **vars(args),
                        "step_timings": list(iteration_timing.values()),
                        "mean_step_timings": mean,
                    }
                    json_data["batch_size"], json_data["seq_length"] = (
                        batch_size,
                        seq_length,
                    )
                    (
                        json_data["tensor_parallelism"],
                        json_data["pipeline_parallelism"],
                    ) = tensor_parallelism, pipeline_parallelism
                    json.dump(json_data, jsonfile, indent=2)
