import argparse
import torch
import os
import sys
import logging
import fiddle
import pytorch_lightning as pl
from nemo import lightning as nl
import nemo_run as run

from dataloader import create_data
from utils import (
    get_check_data_and_tokenizer,
    save_stats,
    save_config,
    write_completion,
    to_nb_tokens,
    SUPPORTED_ARCHITECTURES,
)

from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
from nemo.utils.exp_manager import TimingCallback

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--arch",
        default="llama1b",
        type=str,
        choices=SUPPORTED_ARCHITECTURES,
    )
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--mode", default="debug", type=to_nb_tokens)
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.getenv("OpenLLM_OUTPUT"), "ablations", "train"),
    )
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--seq_length", default=None, type=int)
    parser.add_argument("--tensor_parallelism", default=None, type=int)
    parser.add_argument("--pipeline_parallelism", default=None, type=int)
    parser.add_argument("--context_parallelism", default=None, type=int)
    parser.add_argument("--virtual_pipeline_parallelism", default=None, type=int)
    parser.add_argument("--fp8", default=False, action="store_true")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--base_checkpoint", default=None, type=str)
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(__name__)

    arch = args.arch
    output_dir = args.output_dir

    data_paths, tokenizer_name = get_check_data_and_tokenizer(
        args.config, args.base_checkpoint
    )

    pl.seed_everything(args.seed, workers=True)

    recipe_args = dict(
        dir=output_dir,
        name=args.name,
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        num_gpus_per_node=int(os.environ.get("SLURM_GPUS", 4)),
    )
    if arch.startswith("llama"):
        from recipes.recipe_llama import get_recipe
    elif arch.startswith("nemotronh"):
        from recipes.recipe_nemotronh import get_recipe

        if arch == "nemotronh47b":
            args.fp8 = True
    elif arch.startswith("mixtral") or arch.startswith("mistral"):
        from recipes.recipe_mistral import get_recipe
    elif arch.startswith("qwen"):
        from recipes.recipe_qwen import get_recipe
    else:
        raise NotImplementedError(f"Architecture {arch} not implemented")
    pretrain_recipe, recipes_args = get_recipe(
        arch=arch, recipe_args=recipe_args, performance_mode_if_possible=True
    )
    recipe = pretrain_recipe(**recipes_args)

    batch_size = args.batch_size if args.batch_size else recipe.data.global_batch_size
    seq_length = args.seq_length if args.seq_length else recipe.data.seq_length

    data_args = dict(
        paths=data_paths,
        global_batch_size=batch_size,
        micro_batch_size=recipe.data.micro_batch_size,
        seq_length=seq_length,
        tokenizer_name=tokenizer_name,
        seed=args.seed,
    )

    data = create_data(data_args)

    if args.mode in ["debug", "benchmark", "benchmark100"]:
        max_steps = 1 if args.mode == "debug" else 20
        max_steps = 100 if args.mode == "benchmark100" else max_steps
        resume_if_exists = args.mode.startswith("benchmark")
        every_n_train_steps = max_steps
    else:
        number_of_tokens = args.mode
        max_steps = number_of_tokens // (seq_length * batch_size)
        resume_if_exists = True
        every_n_train_steps = 1_000_000_000 // (seq_length * batch_size)

    # TRAINER
    recipe.trainer.max_steps = max_steps
    recipe.trainer.val_check_interval = (
        5 if args.mode in ["debug", "benchmark", "benchmark100"] else 1000
    )
    recipe.trainer.limit_val_batches = 0.0
    recipe.trainer.log_every_n_steps = (
        1 if args.mode in ["debug", "benchmark", "benchmark100"] else 1
    )
    recipe.trainer.callbacks = [run.Config(TimingCallback)]
    if args.fp8:
        if arch == "nemotronh47b":
            logger.info("FP8 is always activated on nemotronh47b")
        else:
            recipe.trainer.plugins = bf16_with_fp8_mixed()

    # MODEL
    # recipe.model.config.seq_length = seq_length

    # OPTIM
    if (
        isinstance(args.mode, str) or args.mode <= 50_000_000_000
    ):  # if less than 50B tokens, shorter warmup
        recipe.optim.lr_scheduler.warmup_steps = 500
    # optimizer_warmup_steps = 2000

    # STRATEGY
    if args.tensor_parallelism:
        recipe.trainer.strategy.tensor_model_parallel_size = args.tensor_parallelism
    if args.pipeline_parallelism:
        recipe.trainer.strategy.pipeline_model_parallel_size = args.pipeline_parallelism
    recipe.trainer.strategy.pipeline_dtype = torch.bfloat16
    if args.virtual_pipeline_parallelism:
        recipe.trainer.strategy.virtual_pipeline_model_parallel_size = (
            args.virtual_pipeline_parallelism
        )
    if args.context_parallelism:
        recipe.trainer.strategy.context_parallel_size = args.context_parallelism
    # if args.sequence_parallelism is not None:
    #     recipe.trainer.strategy.sequence_parallel = args.sequence_parallelism

    if (
        recipe.trainer.strategy.tensor_model_parallel_size > 4
        and args.tensor_parallelism is None
    ):
        logger.warning(
            f"Tensor parallelism is set to {recipe.trainer.strategy.tensor_model_parallel_size} which is greater than 4. We only have 4 GPUs per node. Setting tensor parallelism to 4."
        )
        recipe.trainer.strategy.tensor_model_parallel_size = 4
        if recipe.data.micro_batch_size > 1:
            recipe.data.micro_batch_size = recipe.data.micro_batch_size // 2
        else:
            recipe.trainer.strategy.pipeline_model_parallel_size = (
                recipe.trainer.strategy.pipeline_model_parallel_size * 2
            )
    # LOGGER
    # recipe.log.log_dir = output_dir
    # recipe.log.name = args.name
    # recipe.log.tensorboard = tensorboard_logger(name=args.name)

    # CKPT
    recipe.log.ckpt = run.Config(
        nl.ModelCheckpoint,
        save_last=True,
        save_top_k=-1,
        every_n_train_steps=every_n_train_steps,
        monitor="step",
        mode="max",
        every_n_epochs=None,
    )

    # RESUME
    recipe.resume = run.Config(
        nl.AutoResume,
        resume_if_exists=resume_if_exists,
        resume_ignore_no_checkpoint=True,
        restore_config=nl.RestoreConfig(path=args.base_checkpoint)
        if args.base_checkpoint
        else None,
    )

    # DATA
    recipe.data = data
    if arch.startswith("nemotronh"):
        recipe.model.tokenizer = data.tokenizer

    job_id = os.environ.get("SLURM_JOB_ID", "0")
    save_config(
        os.path.join(output_dir, f"job_{job_id}"),
        args.name,
        data_args,
        recipe=recipe,
    )

    recipe_obj = fiddle.build(recipe)
    recipe_obj()

    if str(args.mode).startswith("benchmark"):
        save_stats(os.path.join(output_dir, f"job_{job_id}"), args.name)
    write_completion(output_dir)
