import nemo_run as run
import torch
import logging

from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
from nemo.utils.exp_manager import TimingCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_recipe_trainer(recipe, args):
    recipe.trainer.val_check_interval = (
        5 if args.mode in ["debug", "benchmark", "benchmark100"] else 1000
    )
    recipe.trainer.limit_val_batches = 0.0
    recipe.trainer.log_every_n_steps = (
        1 if args.mode in ["debug", "benchmark", "benchmark100"] else 1
    )
    recipe.trainer.callbacks = [run.Config(TimingCallback)]
    if args.fp8:
        if args.arch == "nemotronh47b":
            logger.info("FP8 is always activated on nemotronh47b")
        else:
            recipe.trainer.plugins = bf16_with_fp8_mixed()
            recipe.trainer.plugins.grad_reduce_in_fp32 = False

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

    num_gpus = recipe.trainer.devices * recipe.trainer.num_nodes
    if recipe.data.micro_batch_size > 1 and recipe.data.global_batch_size >= num_gpus:
        logger.warning(
            f"Micro batch size is set to {recipe.data.micro_batch_size} which is greater than 1 and global batch size is greater than number of GPUs. This is not supported for Megat. Setting micro batch size to 1."
        )
        recipe.data.micro_batch_size = 1

    if (
        recipe.trainer.strategy.tensor_model_parallel_size > 4
        and args.tensor_parallelism is None
    ):
        logger.warning(
            f"Tensor parallelism is set to {recipe.trainer.strategy.tensor_model_parallel_size} which is greater than 4. We only have 4 GPUs per node. Setting tensor parallelism to 4."
        )
        recipe.trainer.strategy.tensor_model_parallel_size = 4
        # if recipe.data.micro_batch_size > 1:
        #     recipe.data.micro_batch_size = recipe.data.micro_batch_size // 2
        # else:
        recipe.trainer.strategy.pipeline_model_parallel_size = (
            recipe.trainer.strategy.pipeline_model_parallel_size * 2
        )
    return recipe
