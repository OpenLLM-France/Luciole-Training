import argparse
import torch
import json
import logging
import os
import transformers

from finetune_recipe import (
    create_trainer,
    distributed_fused_adam_with_cosine_annealing,
    create_logger,
    create_autoresume,
)
from utils import read_datamix_file, save_stats, write_completion

from finetune_dataloader import create_data

from nemo.collections import llm
from nemo.utils.exp_manager import TimingCallback
from nemo.collections.common.metrics.perf_metrics import FLOPsMeasurementCallback
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.lightning.pytorch.callbacks.pytorch_profiler import PytorchProfilerCallback
from megatron.core.distributed import DistributedDataParallelConfig
from lightning.pytorch.loggers import TensorBoardLogger
from nemo.collections.llm.modelopt import set_modelopt_spec_if_exists_in_ckpt
from nemo.collections.llm.api import _setup

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
    #parser.add_argument("config")
    parser.add_argument(
        "--arch",
        default="llama1b",
        type=str,
        choices=["llama1b", "llama8b", "mamba1b", "mixtral8x7", "mambahybrid8b", "qwen25-7B", "qwen3-8B"],
    )
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--num_gpus_per_node", default=4, type=int)
    parser.add_argument("--mode", default="debug", type=to_nb_tokens)
    parser.add_argument("--save_every", default="4m", type=to_nb_tokens)
    parser.add_argument(
        "--output_dir",
        default=f"/lustre/fsn1/projects/rech/qgz/{os.environ['USER']}/nemo_test",
    )
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--seq_length", default=100, type=int)
    parser.add_argument("--packed_seq_length", default=100, type=int)
    parser.add_argument("--tensor_parallelism", default=1, type=int)
    parser.add_argument("--pipeline_parallelism", default=1, type=int)
    parser.add_argument("--context_parallelism", default=1, type=int)
    parser.add_argument("--virtual_pipeline_parallelism", default=1, type=int)
    parser.add_argument("--sequence_parallelism", default=False, action="store_true")
    parser.add_argument("--fp8", default=False, action="store_true")
    parser.add_argument("--n_recompute", default=0, type=int)
    parser.add_argument("--warmup", default=50, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--dataset_name", default="databricks/", type=str)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-1B", type=str)
    parser.add_argument("--profile", default=False, action="store_true")
    parser.add_argument("--val_steps", default=30, type=int)
    args = parser.parse_args()

    # suppress_non_main_logging()
    logger = logging.getLogger(__name__)

    arch = args.arch
    num_nodes = args.num_nodes
    output_dir = args.output_dir

    model_name = args.model_name
    data_paths = f"{os.environ['SCRATCH']}/" + args.dataset_name
    tokenizer_name = f"{os.environ['DSDIR']}/HuggingFace_Models/{model_name}"

    batch_size = args.batch_size
    seq_length = args.seq_length
    packed_seq_length = args.packed_seq_length

    # From training (not used)
    # if batch_size is None and packed_seq_length is None:
    #     if arch == "llama1b" or arch == "mamba1b":
    #         batch_size = 512
    #         packed_seq_length = 2048
    #     elif arch == "llama8b" or arch == "mambahybrid8b" or arch == "qwen25-7B" or arch == "qwen3-8B":
    #         batch_size = 1024
    #         packed_seq_length = 4096
    #     elif arch == "mixtral8x7":
    #         batch_size = 512
    #         packed_seq_length = 4096
    #     else:
    #         raise ValueError(f"Unsupported model : {arch}")
    # elif batch_size is None:
    #     batch_size = 4_194_304 // packed_seq_length
    # elif packed_seq_length is None:
    #     packed_seq_length = 4_194_304 // batch_size

    data_args = dict(
        batch_size=batch_size, seq_length=seq_length, packed_seq_length=packed_seq_length, tokenizer_name=tokenizer_name
    )

    data = create_data(data_paths, **data_args)

    if args.mode in ["debug", "benchmark"]:
        max_steps = 1 if args.mode == "debug" else 10
        resume_if_exists = args.mode == "benchmark"
        every_n_train_steps = max_steps
    else:
        number_of_tokens = args.mode
        max_steps = number_of_tokens // (packed_seq_length * batch_size)
        resume_if_exists = True
        every_n_train_steps = args.save_every // (packed_seq_length * batch_size)

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
        sequence_parallel=args.sequence_parallelism,
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

    optimizer_warmup_steps = args.warmup

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
    elif arch.startswith("qwen25"):
        if arch == "qwen25-7B":
            from nemo.collections.llm.gpt.model.qwen2 import (
                Qwen25Config7B as QwenConfig,
            )
        else:
            raise ValueError(f"Unsupported llama model : {arch}")
        model_config = QwenConfig()
        model = llm.Qwen2Model(model_config, tokenizer=data.tokenizer)
    elif arch.startswith("qwen3"):
        if arch == "qwen3-8B":
            from nemo.collections.llm.gpt.model.qwen3 import (
                Qwen3Config8B as QwenConfig,
            )
        else:
            raise ValueError(f"Unsupported qwen model : {arch}")
        model_config = QwenConfig()
        model = llm.Qwen3Model(model_config, tokenizer=data.tokenizer)
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

    if args.n_recompute:
        model_config.recompute_granularity = "full"
        model_config.recompute_method = "block"
        model_config.recompute_num_layers = args.n_recompute
 
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
        max_lr=args.lr, warmup_steps=optimizer_warmup_steps, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, min_lr=0.0, adam_eps=1e-8
    )

    callbacks = [TimingCallback()]

    if args.profile:
        callbacks += [
            GarbageCollectionCallback(
                gc_interval=10,
                gc_interval_val=10,
            ),
            PytorchProfilerCallback(
                start_step=0,
                end_step=max_steps,
                warmup_steps=0,
                active_steps=max_steps,
                trace_dir=f"/lustre/fsn1/projects/rech/qgz/{os.environ['USER']}/Training_OpenLLM/torch",
                profiler_kwargs={
                    'with_stack': True,
                    'profile_memory': True,
                }
            )
        ]

    trainer = create_trainer(
        strategy_args=strategy_args,
        max_steps=max_steps,
        num_gpus_per_node=args.num_gpus_per_node,
        num_nodes=num_nodes,
        callbacks=callbacks,
        val_check_interval=5 if args.mode in ["debug", "benchmark"] else args.val_steps,
        limit_val_batches=0.0 if args.mode in ["debug", "benchmark"] else 1.0,
        fp8=args.fp8,
    )

    nemo_logger = create_logger(
        dir=output_dir,
        name=args.name,
        every_n_train_steps=every_n_train_steps,
    )

    llm.finetune(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        optim=opt,
        #resume=create_autoresume(path=f"nemo://{model_name.split('/')[-1]}", resume_if_exists=resume_if_exists),
    )

    if args.mode in ["debug", "benchmark"]:
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
