import argparse
from recipes.recipe_utils import SUPPORTED_ARCHITECTURES
import os
import datetime


def setup_parallelism(
    recipe,
    tensor_parallelism=None,
    pipeline_parallelism=None,
    context_parallelism=None,
    # sequence_parallelism=None,
):
    # Set custom values
    if tensor_parallelism:
        recipe.trainer.strategy.tensor_model_parallel_size = tensor_parallelism
    if pipeline_parallelism:
        recipe.trainer.strategy.pipeline_model_parallel_size = pipeline_parallelism
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = None
    logger.warning("virtual_pipeline_model_parallel_size is set to None")
    if context_parallelism:
        recipe.trainer.strategy.context_parallel_size = context_parallelism
    if (recipe.trainer.strategy.tensor_model_parallel_size == 1) or (
        recipe.data.seq_length <= 4096
    ):
        logger.warning("TP=1, setting sequence_parallel to False")
        recipe.trainer.strategy.sequence_parallel = False
    # Assert
    assert not (
        recipe.data.seq_length <= 4096
        and recipe.trainer.strategy.context_parallel_size > 1
    ), "seq_length <= 4096, and context_parallel_size > 1"
    assert recipe.trainer.strategy.tensor_model_parallel_size <= 4, (
        f"Tensor parallelism is set to {recipe.trainer.strategy.tensor_model_parallel_size}, "
        "which is greater than 4. We only have 4 GPUs per node."
    )
    return recipe


def get_time_limit(time_limit, buffer_minutes: int = 30) -> str:
    logging.info(time_limit)
    h, m, s = map(int, time_limit.split(":"))
    slurm_time_limit = datetime.timedelta(hours=h, minutes=m, seconds=s)
    td = slurm_time_limit - datetime.timedelta(minutes=buffer_minutes)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    return f"{td.days:02}:{hours:02}:{minutes:02}:00"


def get_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--datamix",
        default="datamixes/mock.json",
        help="Path to the datamix, should be a json or yaml file.",
        type=str,
    )
    parser.add_argument(
        "--arch",
        default="llama1b",
        type=str,
        choices=SUPPORTED_ARCHITECTURES,
    )
    parser.add_argument("--name", default="", type=str)
    parser.add_argument(
        "--mode",
        default="debug",
        choices=["debug", "benchmark", "phase1", "phase2", "annealing"],
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.getenv("OpenLLM_OUTPUT"), "test_run"),
    )
    parser.add_argument("--batch_size", "--gbs", default=1024, type=int)
    parser.add_argument("--micro_batch_size", "--mbs", default=None, type=int)
    parser.add_argument("--seq_length", default=4096, type=int)
    parser.add_argument("--tensor_parallelism", "--tp", default=None, type=int)
    parser.add_argument("--pipeline_parallelism", "--pp", default=None, type=int)
    parser.add_argument("--context_parallelism", "--cp", default=None, type=int)
    parser.add_argument(
        "--virtual_pipeline_parallelism", "--vpp", default=None, type=int
    )
    parser.add_argument("--fp8", default=False, action="store_true")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--base_checkpoint", default=None, type=str)
    parser.add_argument("--performance_mode", default=False, action="store_true")
    parser.add_argument("--time", default="100:00:00", type=str)
    parser.add_argument(
        "--ckpt_intervals",
        default="{1: 1, 50_000: 1000, 100_000: 5_000}",
        type=str,
    )
    parser.add_argument(
        "--fp8_recipe", default="tensorwise", choices=["delayed", "tensorwise"]
    )
    parser.add_argument("--fp8_layers_bf16", default=4, type=int)
    parser.add_argument(
        "--no_grad_reduce_in_fp32",
        action="store_true",
        help="Disable gradient reduction in fp32",
    )
    parser.add_argument(
        "--scheduler",
        default="wsd",
        choices=["wsd", "cosine"],
        type=str,
    )
    parser.add_argument("--max_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--no_load_optim_state", default=False, action="store_true")
    return parser


if __name__ == "__main__":
    import ast
    import logging
    import math
    import sys
    from functools import partial

    import fiddle
    import pytorch_lightning as pl
    import torch

    from nemo import lightning as nl
    from nemo.collections.llm.gpt.data import PreTrainingDataModule
    from nemo.collections.llm.recipes.precision.mixed_precision import (
        bf16_with_fp8_current_scaling_mixed,
        bf16_with_fp8_mixed,
    )
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
    from nemo.lightning.pytorch.callbacks import (
        GarbageCollectionCallback,
    )  # , SpikeDetection
    from nemo.lightning.pytorch.optim import WarmupAnnealingScheduler

    from callbacks import (
        ProgressiveIntervalCheckpoint,
        checkpoint_along_step_curve,
        StatelessTimer,
        CustomTimingCallback,
        StopAtEndOfPhaseCallback,
    )
    from recipes.recipe_utils import get_recipe

    from utils import (
        check_tokenizer,
        process_datamix_file,
        save_config,
    )

    import nemo_run as run

    # from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
    # from .callbacks import PytorchProfilerCallback
    # from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback

    parser = get_parser()
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(__name__)

    pl.seed_everything(args.seed, workers=True)

    # Set up base recipe
    recipe_args = dict(
        dir=args.output_dir,
        name=args.name,
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        num_gpus_per_node=int(os.environ.get("SLURM_GPUS", 4)),
    )
    recipe = get_recipe(
        arch=args.arch,
        recipe_args=recipe_args,
        performance_mode_if_possible=args.performance_mode,
    )

    ### DATA SETUP
    # Read datamix config
    tokenizer_name, data_paths, total_tokens = process_datamix_file(args.datamix)
    check_tokenizer(tokenizer_name, args.base_checkpoint)

    # Set up batch size and seq length
    global_batch_size = (
        args.batch_size if args.batch_size else recipe.data.global_batch_size
    )
    seq_length = args.seq_length if args.seq_length else recipe.data.seq_length
    tokens_per_batch = seq_length * global_batch_size
    if args.scheduler == "cosine" and args.mode == "phase2":
        max_steps = math.floor(2 * 1e12 / tokens_per_batch)  # 2T horizon for phase 2
        max_steps_phase2 = math.floor(total_tokens / tokens_per_batch)
    else:
        max_steps = math.floor(total_tokens / tokens_per_batch)
    logger.info(f"Global batch size: {global_batch_size}")
    logger.info(f"Sequence length: {seq_length}")
    logger.info(f"Tokens per batch: {tokens_per_batch}")
    logger.info(f"Total tokens in your datamix: {total_tokens}")

    # Set up recipe data
    micro_batch_size = (
        recipe.data.micro_batch_size
        if args.micro_batch_size is None
        else args.micro_batch_size
    )
    logger.info(f"Micro batch size: {micro_batch_size}")

    data_args = dict(
        num_workers=8,
        pin_memory=True,
        split="1,0,0",
        paths=data_paths,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        seq_length=seq_length,
        seed=args.seed,
        index_mapping_dir=os.path.join(args.output_dir, "index_mapping"),
    )
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, use_fast=True)
    recipe.data = run.Config(PreTrainingDataModule, tokenizer=tokenizer, **data_args)

    ### MODEL SETUP
    recipe.model.tokenizer = recipe.data.tokenizer
    recipe.model.config.seq_length = recipe.data.seq_length
    if args.arch.startswith("llama") and args.base_checkpoint is None:
        recipe.model.config.old_context_len = recipe.data.seq_length

    ### TRAINER SETUP
    if args.mode == "debug":
        max_steps = 2
    elif args.mode == "benchmark":
        max_steps = 25
    recipe.trainer.max_steps = max_steps
    recipe.trainer.val_check_interval = max_steps
    recipe.trainer.limit_val_batches = 0.0
    recipe.trainer.log_every_n_steps = 1 if args.mode in ["debug", "benchmark"] else 5
    recipe.trainer.strategy.pipeline_dtype = torch.bfloat16
    recipe.trainer.strategy.ckpt_async_save = True

    # FP8 setup
    if args.fp8:
        if args.fp8_recipe == "delayed":
            fp8_plugin = bf16_with_fp8_mixed()
        else:
            fp8_plugin = bf16_with_fp8_current_scaling_mixed()
            fp8_plugin.first_last_layers_bf16 = True
            fp8_plugin.num_layers_at_start_in_bf16 = args.fp8_layers_bf16
            fp8_plugin.num_layers_at_end_in_bf16 = args.fp8_layers_bf16
        recipe.trainer.plugins = fp8_plugin
        recipe.trainer.plugins.grad_reduce_in_fp32 = not args.no_grad_reduce_in_fp32
        recipe.trainer.strategy.ddp.grad_reduce_in_fp32 = (
            not args.no_grad_reduce_in_fp32
        )
        logger.info(f"Using FP8 with config: {fp8_plugin}")

    # Parallelism setup
    recipe = setup_parallelism(
        recipe,
        tensor_parallelism=args.tensor_parallelism,
        pipeline_parallelism=args.pipeline_parallelism,
        context_parallelism=args.context_parallelism,
        # sequence_parallelism=args.sequence_parallelism,
    )

    # Callbacks setup
    time_limit = get_time_limit(
        args.time, 5 if args.mode in ["debug", "benchmark"] else 30
    )
    recipe.trainer.callbacks = []
    recipe.trainer.callbacks.append(run.Config(StatelessTimer, duration=time_limit))
    logger.info("Added StatelessTimer")
    recipe.trainer.callbacks.append(
        run.Config(CustomTimingCallback)
    )  # , max_training_time_per_step=args.max_training_time_per_step))
    logger.info("Added TimingCallback")
    # recipe.trainer.callbacks.append(run.Config(StopAtEndOfPhaseCallback, end_step=args.end_step))
    # logger.info("Added StopAtEndOfPhaseCallback")
    recipe.trainer.callbacks.append(
        run.Config(
            GarbageCollectionCallback,
            gc_interval_train=100,
            gc_interval_val=100,
        )
    )
    logger.info("Added GarbageCollectionCallback")
    # run.Config(MegatronCommOverlapCallback, tp_comm_overlap=True),
    # os.makedirs(f"{args.output_dir}/traces", exist_ok=True)
    # run.Config(PytorchProfilerCallback, start_step=15, end_step=20, warmup_steps=1, active_steps=5, trace_dir=f"{args.output_dir}/traces")

    # Custom checkpointing method
    every_n_train_steps = (
        30 if args.mode in ["debug", "benchmark"] else min(max_steps, 10_000)
    )
    intervals = (
        {}
        if args.mode in ["debug", "benchmark"]
        else ast.literal_eval(args.ckpt_intervals)
    )
    every_function_train_steps = partial(
        checkpoint_along_step_curve,
        intervals=intervals,
        else_interval=5_000,
    )

    recipe.log.ckpt = run.Config(
        ProgressiveIntervalCheckpoint,
        filename=args.name + "-{step:07.0f}",
        save_last=True,
        save_top_k=-1,
        every_function_train_steps=every_function_train_steps,
        every_n_train_steps=every_n_train_steps,
        monitor="step",
        mode="max",
        every_n_epochs=None,
        save_optim_on_train_end=True,  # set to True if you want to continue training even if max_steps was reached
        # async_save = not args.sync_ckpt,
    )

    ### OPTIM SETUP
    recipe.optim.config.lr = (
        args.max_lr if args.max_lr is not None else recipe.optim.config.lr
    )
    max_lr = recipe.optim.config.lr
    if args.mode in ["debug", "benchmark"]:
        warmup = 5
    elif args.mode == "phase1":
        warmup = 2000
    elif args.mode in ["phase2", "annealing"]:
        warmup = 0
    # Scheduler setup
    if args.scheduler == "wsd":
        min_lr = max_lr if args.mode != "annealing" else 0
        recipe.optim.lr_scheduler = run.Config(
            WarmupAnnealingScheduler, warmup_steps=warmup, min_lr=min_lr
        )
        logger.info(
            f"Setting WarmupAnnealingScheduler with max_steps: {max_steps}, warmup: {warmup}, max_lr: {max_lr}, min_lr: {min_lr}"
        )
    elif args.scheduler == "cosine":
        recipe.optim.lr_scheduler.warmup_steps = 0
        recipe.optim.lr_scheduler.min_lr = max_lr * 0.1
        logger.info(
            f"Setting Cosine Scheduler with max_steps: {max_steps} (2T tokens), warmup: {warmup}"
        )
        if args.mode == "phase2":
            recipe.trainer.callbacks.append(
                run.Config(StopAtEndOfPhaseCallback, end_step=max_steps_phase2)
            )
            logger.info(
                f"Setting StopAtEndOfPhaseCallback with end_step: {max_steps_phase2}"
            )
    else:
        raise ValueError(
            f"Scheduler {args.scheduler} not supported in mode {args.mode}"
        )
    # Weight decay
    recipe.optim.weight_decay = args.weight_decay
    logger.info(f"Setting weight decay to {args.weight_decay}")

    # Resume from base_checkpoint
    restore_config = (
        nl.RestoreConfig(
            path=args.base_checkpoint, load_optim_state=not args.no_load_optim_state
        )
        if args.base_checkpoint
        else None
    )

    resume_if_exists = False if args.mode == "debug" else True
    resume_ignore_no_checkpoint = True
    # resume_ignore_no_checkpoint = False if args.mode in ["phase2", "annealing"] else True
    recipe.resume = run.Config(
        nl.AutoResume,
        resume_if_exists=resume_if_exists,
        resume_ignore_no_checkpoint=resume_ignore_no_checkpoint,
        resume_past_end=True,  # set to True if you want to continue training even if max_steps was reached
        restore_config=restore_config,
    )

    # Save config
    job_id = os.environ.get("SLURM_JOB_ID", "0")
    job_output = os.path.join(args.output_dir, f"job_{job_id}")
    os.makedirs(job_output, exist_ok=True)
    save_config(
        job_output,
        args,
        recipe,
    )

    recipe_obj = fiddle.build(recipe)
    recipe_obj()

    logger.info("Finished training.")
