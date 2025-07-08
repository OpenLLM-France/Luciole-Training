import argparse
import torch
import os
import logging
import fiddle
import pytorch_lightning as pl
from nemo import lightning as nl
import nemo_run as run
from pprint import pprint

from dataloader import create_data
from recipes.recipe_utils import (
    create_autoresume,
)
from utils import (
    get_check_data_and_tokenizer,
    save_stats,
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
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--num_gpus_per_node", default=4, type=int)
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

    # suppress_non_main_logging()
    logger = logging.getLogger(__name__)

    arch = args.arch
    num_nodes = args.num_nodes
    output_dir = args.output_dir

    data_paths, tokenizer_name = get_check_data_and_tokenizer(
        args.config, args.base_checkpoint
    )

    batch_size = args.batch_size
    seq_length = args.seq_length

    if batch_size is None and seq_length is None:
        if arch in ["llama1b", "llama3b", "mamba1b"]:
            batch_size = 512
            seq_length = 2048
        elif arch in ["llama8b", "mambahybrid8b", "nemotronh8b"]:
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

    # optimizer_warmup_steps = 2000
    recipe_args = dict(
        dir=output_dir,
        name=args.name,
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)),
        num_gpus_per_node=int(os.environ.get("SLURM_GPUS", 4)),
    )
    if arch.startswith("llama"):
        from recipes.recipe_llama import get_recipe

        pretrain_recipe, recipes_args = get_recipe(
            arch=arch, recipe_args=recipe_args, performance_mode_if_possible=True
        )
    elif arch.startswith("nemotronh"):
        from recipes.recipe_nemotronh import get_recipe

        pretrain_recipe, recipes_args = get_recipe(
            arch=arch, recipe_args=recipe_args, performance_mode_if_possible=True
        )
        if arch == "nemotronh47b":
            args.fp8 = True
    elif arch.startswith("mixtral"):
        from recipes.recipe_mixtral import get_recipe

        pretrain_recipe, recipes_args = get_recipe(
            arch=arch, recipe_args=recipe_args, performance_mode_if_possible=True
        )
    else:
        raise NotImplementedError(f"Architecture {arch} not implemented")

    recipe = pretrain_recipe(**recipes_args)

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

    # OPTIM
    recipe.optim.lr_scheduler.warmup_steps = 500

    # STRATEGY
    if args.tensor_parallelism:
        recipe.trainer.strategy.tensor_model_parallel_size = args.tensor_parallelism
    if args.pipeline_parallelism:
        recipe.trainer.strategy.pipeline_model_parallel_size = args.pipeline_parallelism
    # recipe.trainer.strategy.pipeline_dtype = torch.bfloat16
    if args.virtual_pipeline_parallelism:
        recipe.trainer.strategy.virtual_pipeline_model_parallel_size = (
            args.virtual_pipeline_parallelism
        )
    if args.context_parallelism:
        recipe.trainer.context_parallel_size = args.context_parallelism
    # if args.sequence_parallelism is not None:
    #     recipe.trainer.strategy.sequence_parallel = args.sequence_parallelism

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
    recipe.resume = create_autoresume(
        resume_if_exists=resume_if_exists, base_checkpoint=args.base_checkpoint
    )

    # DATA
    recipe.data = data
    if arch.startswith("nemotronh"):
        recipe.model.tokenizer = data.tokenizer
    pprint(recipe)

    recipe_obj = fiddle.build(recipe)
    recipe_obj()

    save_stats(
        output_dir,
        args.name,
        data_args,
        recipe=recipe,
        write_step_timings=True
        if args.mode in ["debug"] or str(args.mode).startswith("benchmark")
        else False,
    )
    write_completion(output_dir)
