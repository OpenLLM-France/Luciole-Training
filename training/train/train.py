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
        if x == "debug":
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
        "--arch", default="llama1b", type=str, choices=["llama1b", "llama8b", "mamba"]
    )
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--num_gpus_per_node", default=4, type=int)
    parser.add_argument("--mode", default="debug", type=to_nb_tokens)
    parser.add_argument(
        "--output_dir",
        default="/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/train",
    )
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--seq_length", default=2048, type=int)
    parser.add_argument("--fp8", default=False, action="store_true")
    args = parser.parse_args()

    arch = args.arch
    num_nodes = args.num_nodes
    name = args.name
    output_dir = args.output_dir

    data_paths, tokenizer_name = read_datamix_file(args.config)

    data = create_data(
        data_paths,
        tokenizer_name=tokenizer_name,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
    )

    if args.mode == "debug":
        max_steps = 5
        resume_if_exists = False
        every_n_train_steps = max_steps
    else:
        number_of_tokens = args.mode
        max_steps = number_of_tokens // (args.seq_length * args.batch_size)
        resume_if_exists = True
        every_n_train_steps = 2_500_000_000 // (args.seq_length * args.batch_size)

    logger.info(f"Job name: {args.name}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Sequence length: {args.seq_length}")
    logger.info(f"Number of nodes: {args.num_nodes}")
    logger.info(f"Tokenizer: {tokenizer_name}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Saving checkpoints every {every_n_train_steps} train steps")
    logger.info(f"Resume training if possible: {resume_if_exists}")

    if arch.startswith("llama"):
        # Llama config
        if arch == "llama1b":
            from nemo.collections.llm.gpt.model.llama import (
                Llama32Config1B as LlamaConfig,
            )
        elif arch == "llama8b":
            from nemo.collections.llm.gpt.model.llama import (
                Llama31Config8B as LlamaConfig,
            )
        else:
            raise ValueError(f"Unsupported llama model : {arch}")
        model_config = LlamaConfig()
        model = llm.LlamaModel(model_config, tokenizer=data.tokenizer)
    elif arch == "mamba":
        # Mamba Config
        from nemo.collections.llm.gpt.model.ssm import BaseMambaConfig1_3B

        model_config = BaseMambaConfig1_3B(
            tokenizer_library="huggingface",
            tokenizer_name=tokenizer_name,
            share_embeddings_and_output_weights=True,
        )
        model = llm.GPTModel(model_config, tokenizer=data.tokenizer)
    else:
        raise NotImplementedError(f"Architecture {arch} not implemented")

    opt = distributed_fused_adam_with_cosine_annealing(max_lr=3e-4)

    trainer = create_trainer(
        tensor_parallelism=1,
        pipeline_parallelism=1,
        pipeline_parallelism_type=torch.bfloat16,
        max_steps=max_steps,
        num_gpus_per_node=args.num_gpus_per_node,
        num_nodes=num_nodes,
        callbacks=[TimingCallback()],
        val_check_interval=5 if args.mode == "debug" else 1000,
        limit_val_batches=0.0,  # 1 if args.mode == "debug" else 0,
        fp8=args.fp8,
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
