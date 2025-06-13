import argparse
import torch
import json
import logging
import pytorch_lightning as pl

from training.train.recipe_pretrain import (
    create_trainer,
    distributed_fused_adam_with_cosine_annealing,
    create_logger,
    create_autoresume,
)
from utils import read_datamix_file, save_stats, write_completion

from dataloader import create_data

from nemo.collections import llm
from nemo.utils.exp_manager import TimingCallback
from megatron.core.distributed import DistributedDataParallelConfig

if __name__ == "__main__":

    def to_nb_tokens(x):
        if x == "debug" or x == "benchmark" or x == "benchmark100":
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
        choices=[
            "llama1b",
            "llama8b",
            "llama70b",
            "mamba1b",
            "mixtral8x7",
            "mambahybrid8b",
        ],
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
    parser.add_argument("--seed", default=1234, type=int)
    args = parser.parse_args()

    # suppress_non_main_logging()
    logger = logging.getLogger(__name__)

    arch = args.arch
    num_nodes = args.num_nodes
    output_dir = args.output_dir

    # try:
    data_paths, tokenizer_name = read_datamix_file(args.config)

    batch_size = args.batch_size
    seq_length = args.seq_length

    if batch_size is None and seq_length is None:
        if arch in ["llama1b", "mamba1b"]:
            batch_size = 512
            seq_length = 2048
        elif arch in ["llama8b", "mambahybrid8b"]:
            batch_size = 1024
            seq_length = 4096
        elif arch == "mixtral8x7":
            batch_size = 512
            seq_length = 4096
        elif arch in ["llama70b"]:
            batch_size = 512
            seq_length = 8192
        else:
            raise ValueError(f"Unsupported model : {arch}")
    elif batch_size is None:
        batch_size = 4_194_304 // seq_length
    elif seq_length is None:
        seq_length = 4_194_304 // batch_size

    pl.seed_everything(args.seed, workers=True)

    data_args = dict(
        paths=data_paths,
        global_batch_size=batch_size,
        seq_length=seq_length,
        tokenizer_name=tokenizer_name,
        seed=args.seed,
    )

    data = create_data(data_args)

    if args.mode in ["debug", "benchmark", "benchmark100"]:
        max_steps = 1 if args.mode == "debug" else 10
        max_steps = 100 if args.mode == "benchmark100" else max_steps
        resume_if_exists = args.mode.startswith("benchmark")
        every_n_train_steps = max_steps
    else:
        number_of_tokens = args.mode
        max_steps = number_of_tokens // (seq_length * batch_size)
        resume_if_exists = True
        every_n_train_steps = 1_000_000_000 // (seq_length * batch_size)

    strategy_args = dict(
        tensor_model_parallel_size=args.tensor_parallelism
        if args.tensor_parallelism
        else 1,
        pipeline_model_parallel_size=args.pipeline_parallelism
        if args.pipeline_parallelism
        else 1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=args.virtual_pipeline_parallelism,
        context_parallel_size=args.context_parallelism,
        sequence_parallel=False,
        gradient_as_bucket_view=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
        ),
    )

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
        elif arch == "llama70b":
            from nemo.collections.llm.gpt.model.llama import (
                Llama31Config70B as LlamaConfig,
            )

            strategy_args["sequence_parallel"] = True
            strategy_args["tensor_model_parallel_size"] = 2
            strategy_args["pipeline_model_parallel_size"] = 2
            strategy_args["virtual_pipeline_model_parallel_size"] = 5
            strategy_args["context_parallel_size"] = 2
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
            strategy_args["tensor_model_parallel_size"] = (
                args.tensor_parallelism if args.tensor_parallelism else 4
            )
        model = llm.GPTModel(model_config, tokenizer=data.tokenizer)
    elif arch == "mixtral8x7":
        from nemo.collections.llm.gpt.model.mixtral import (
            MixtralConfig8x7B,
            MixtralModel,
        )

        model_config = MixtralConfig8x7B()
        model = MixtralModel(model_config, tokenizer=data.tokenizer)

        strategy_args["virtual_pipeline_model_parallel_size"] = 8
        strategy_args["pipeline_model_parallel_size"] = 4
        strategy_args["expert_model_parallel_size"] = 8
    else:
        raise NotImplementedError(f"Architecture {arch} not implemented")

    args_dict = vars(args)
    for key in [
        "tensor_parallelism",
        "pipeline_parallelism",
        "context_parallelism",
        "virtual_pipeline_parallelism",
        "batch_size",
        "seq_length",
    ]:
        args_dict.pop(key, None)

    logger.info("Args:\n" + json.dumps(args_dict, indent=2))
    logger.info("Strategy Args:\n" + json.dumps(strategy_args, indent=2, default=str))
    logger.info("Data Args:\n" + json.dumps(data_args, indent=2, default=str))

    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Saving checkpoints every {every_n_train_steps} train steps")
    logger.info(f"Resume training if possible: {resume_if_exists}")

    opt = distributed_fused_adam_with_cosine_annealing(
        max_lr=3e-4, warmup_steps=optimizer_warmup_steps
    )

    trainer = create_trainer(
        strategy_args=strategy_args,
        max_steps=max_steps,
        num_gpus_per_node=args.num_gpus_per_node,
        num_nodes=num_nodes,
        callbacks=[TimingCallback()],
        val_check_interval=5
        if args.mode in ["debug", "benchmark", "benchmark100"]
        else 1000,
        limit_val_batches=0.0,  # 1 if args.mode == "debug" else 0,
        fp8=args.fp8,
        log_every_n_steps=1
        if args.mode in ["debug", "benchmark", "benchmark100"]
        else 10,
    )

    nemo_logger = create_logger(
        dir=output_dir,
        name=args.name,
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

    if args.mode in ["debug"] or str(args.mode).startswith("benchmark"):
        save_stats(
            output_dir,
            args=args_dict,
            strategy_args=strategy_args,
            data_args=data_args,
        )
    write_completion(output_dir)
    # finally:
    #     if dist.is_available() and dist.is_initialized():
    #         dist.barrier()
    #         dist.destroy_process_group()
