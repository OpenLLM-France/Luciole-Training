import argparse
import torch
import logging
import nemo_run as run
import fiddle as fd
from llama32_config import convert_to_llama32_1b
from utils import read_datamix_file, create_data, create_logger

from nemo.collections import llm
from nemo.utils.exp_manager import TimingCallback
from nemo.collections.llm.recipes.log.default import default_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import (
    distributed_fused_adam_with_cosine_annealing,
)
from nemo.collections.llm.recipes.llama3_8b import trainer  # , model
from nemo.collections.llm.gpt.model.llama import LlamaModel, Llama31Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--mode", choices=["debug", "20b", "35b"], default="debug")
    parser.add_argument(
        "--output_dir",
        default="/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/ablations/train",
    )
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--seq_length", default=2048, type=int)
    parser.add_argument("--tokenizer", default="OpenLLM-France/Lucie-7B", type=str)
    args = parser.parse_args()

    num_nodes = args.num_nodes
    name = args.name
    output_dir = args.output_dir

    data_paths = read_datamix_file(args.config)
    data = create_data(
        data_paths,
        tokenizer_name=args.tokenizer,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
    )

    if args.mode == "debug":
        max_steps = 10
        resume_if_exists = False
        every_n_train_steps = 5
    else:
        number_of_tokens = int(args.mode.replace("b", "")) * 1_000_000_000
        max_steps = number_of_tokens // (data.seq_length * data.global_batch_size)
        resume_if_exists = True
        every_n_train_steps = 5_000_000_000 // (
            data.seq_length * data.global_batch_size
        )

    logger.info(f"Job name: {args.name}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Sequence length: {args.seq_length}")
    logger.info(f"Number of nodes: {args.num_nodes}")
    logger.info(f"Tokenizer: {args.tokenizer}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Saving checkpoints every {every_n_train_steps} train steps")
    logger.info(f"Resume training if possible: {resume_if_exists}")

    # model = model()   # default recipe function but use wrong model config and it is not an arg
    model = run.Config(LlamaModel, config=run.Config(Llama31Config))
    model = convert_to_llama32_1b(model)

    trainer = trainer(
        tensor_parallelism=1,
        pipeline_parallelism=1,
        pipeline_parallelism_type=torch.bfloat16,
        max_steps=max_steps,
        num_gpus_per_node=4,
        num_nodes=num_nodes,
        callbacks=[run.Config(TimingCallback)],
    )
    # default recipe function but use wrong way for checkpointing (time_interval), we could edit afterwards
    # nemo_logger=default_log(dir=output_dir, name=name, tensorboard_logger=tensorboard_logger(name=name))
    nemo_logger = create_logger(
        dir=output_dir,
        name=name,
        every_n_train_steps=every_n_train_steps,
        tensorboard_logger=fd.build(tensorboard_logger(name=name)),
    )

    optim = distributed_fused_adam_with_cosine_annealing(max_lr=3e-4)
    create_autoresume = default_resume(resume_if_exists=resume_if_exists)

    model = fd.build(model)
    trainer = fd.build(trainer)
    # nemo_logger = fd.build(nemo_logger)
    optim = fd.build(optim)
    create_autoresume = fd.build(create_autoresume)

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer="data",
        optim=optim,
        resume=create_autoresume,
    )
