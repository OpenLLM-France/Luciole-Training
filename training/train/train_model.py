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
from recipes.recipe_utils import set_recipe_trainer
from utils import (
    get_check_data_and_tokenizer,
    save_stats,
    save_config,
    write_completion,
    to_nb_tokens,
    SUPPORTED_ARCHITECTURES,
)


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
    parser.add_argument("--performance_mode", default=False, action="store_true")
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
    elif arch.startswith("nemotron"):
        from recipes.recipe_nemotronh import get_recipe

        if arch == "nemotronh47b":
            args.fp8 = True
    elif arch.startswith("mixtral") or arch.startswith("mistral"):
        from recipes.recipe_mistral import get_recipe
    elif arch.startswith("qwen"):
        from recipes.recipe_qwen import get_recipe
    elif arch.startswith("ablation"):
        from recipes.recipe_ablations import get_recipe
    else:
        raise NotImplementedError(f"Architecture {arch} not implemented")
    pretrain_recipe, recipes_args = get_recipe(
        arch=arch,
        recipe_args=recipe_args,
        performance_mode_if_possible=args.performance_mode,
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
    recipe.data = data
    recipe.model.tokenizer = data.tokenizer
    
    if args.mode in ["debug", "benchmark", "benchmark100"]:
        max_steps = 1 if args.mode == "debug" else 25
        max_steps = 100 if args.mode == "benchmark100" else max_steps
        resume_if_exists = args.mode.startswith("benchmark")
        every_n_train_steps = max_steps
    else:
        number_of_tokens = args.mode
        max_steps = number_of_tokens // (seq_length * batch_size)
        resume_if_exists = True
        every_n_train_steps = 1_000_000_000 // (seq_length * batch_size)

    recipe = set_recipe_trainer(recipe, args, max_steps)

    if arch == "llama24b":
        from recipes.recipe_llama import set_llama24b_recipe
        recipe = set_llama24b_recipe(recipe, args)
    elif arch.startswith("ablation"):
        from recipes.recipe_ablations import set_ablation_recipe
        recipe = set_ablation_recipe(recipe, arch)

    # MODEL
    # recipe.model.config.seq_length = seq_length

    # OPTIM
    if (
        isinstance(args.mode, str) or args.mode <= 50_000_000_000
    ):  # if less than 50B tokens, shorter warmup
        recipe.optim.lr_scheduler.warmup_steps = 500
    # optimizer_warmup_steps = 2000

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

    job_id = os.environ.get("SLURM_JOB_ID", "0")
    job_output = os.path.join(output_dir, f"job_{job_id}")
    os.makedirs(job_output, exist_ok=True)
    save_config(
        job_output,
        args,
        data_args,
        recipe=recipe,
    )

    recipe_obj = fiddle.build(recipe)
    recipe_obj()

    if str(args.mode).startswith("benchmark"):
        save_stats(job_output, args.name)
    write_completion(output_dir)
