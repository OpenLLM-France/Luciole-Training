import argparse
import torch
import logging
from nemo.collections.llm.gpt.model.ssm import BaseMambaConfig1_3B

from recipe_llama import (
    create_trainer,
    distributed_fused_adam_with_cosine_annealing,
    create_logger,
    create_autoresume,
)
from utils import (
    read_datamix_file,
    get_config,
)

from dataloader import create_data

from nemo.collections import llm
from nemo.utils.exp_manager import TimingCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--num_gpus_per_node", default=4, type=int)
    parser.add_argument("--mode", choices=["debug", "20b", "35b"], default="debug")
    parser.add_argument(
        "--output_dir",
        default="/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/train",
    )
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--seq_length", default=2048, type=int)
    args = parser.parse_args()

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
        max_steps = 1
        resume_if_exists = False
        every_n_train_steps = 5
    else:
        number_of_tokens = int(args.mode.replace("b", "")) * 1_000_000_000
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

    model_config = BaseMambaConfig1_3B(
        tokenizer_library="huggingface",
        tokenizer_name=tokenizer_name
    )
    model = llm.MambaModel(model_config, tokenizer=data.tokenizer)

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
        limit_val_batches=1 if args.mode == "debug" else 0,
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
