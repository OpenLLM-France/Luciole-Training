import logging
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_ARCHITECTURES = [
    "llama1b",
    "llama3b",
    "llama8b",
    "llama21b",
    "llama24b",
    "llama70b",
    "mistral12b",
    "mistral_small3_24b",
    "mixtral8x7",
    "nemotronh8b",
    "nemotronh47b",
    "nemotron_nano9b",
    "nemotron1b",
    "nemotron4b",
    "nemotron8b",
    "nemotron22b",
    "nemotron20b_wider",
    "nemotron20b_deeper",
    "qwen32b",
]


def set_performance_mode_if_possible(arch):
    if arch in [
        "llama8b",
        "llama24b",
        "llama70b",
        "mixtral8x7",
        "nemotronh47b",
        "nemotron22b",
        "nemotron8b",
    ]:
        return True
    return False


def get_recipe(arch, recipe_args, performance_mode_if_possible=False):
    # Setup base recipe
    if arch == "mixtral8x7":
        from nemo.collections.llm.recipes.mixtral_8x7b import pretrain_recipe
    elif arch == "mistral12b":
        if performance_mode_if_possible:
            from nemo.collections.llm.recipes.mistral_nemo_12b import (
                pretrain_recipe_performance as pretrain_recipe,
            )
        else:
            from nemo.collections.llm.recipes.mistral_nemo_12b import pretrain_recipe
    elif arch == "mistral_small3_24b":
        from .mistral_small3_24b import pretrain_recipe
    elif arch == "nemotron1b":
        from .nemotron_1b import pretrain_recipe
    elif arch == "nemotron4b":
        from nemo.collections.llm.recipes.nemotron3_4b import pretrain_recipe
    elif arch == "nemotron8b":
        from nemo.collections.llm.recipes.nemotron3_8b import pretrain_recipe
    elif arch == "nemotron22b":
        from nemo.collections.llm.recipes.nemotron3_22b import pretrain_recipe
    elif arch == "nemotron20b_wider":
        from .nemotron_20b_wider import pretrain_recipe
    elif arch == "nemotron20b_deeper":
        from .nemotron_20b_deeper import pretrain_recipe
    elif arch == "nemotronh8b":
        from .nemotronh_8b import pretrain_recipe
    elif arch == "nemotron_nano9b":
        from .nemotron_nano_9b import pretrain_recipe
    elif arch == "nemotronh47b":
        from nemo.collections.llm.recipes.nemotronh_47b import pretrain_recipe
    elif arch == "qwen32b":
        from nemo.collections.llm.recipes.qwen25_32b import pretrain_recipe
    # elif arch == "qwen30ba3b":
    #     from .qwen3 import pretrain_recipe
    elif arch == "llama1b":
        from nemo.collections.llm.recipes.llama32_1b import pretrain_recipe
    elif arch == "llama3b":
        from nemo.collections.llm.recipes.llama32_3b import pretrain_recipe
    elif arch == "llama8b":
        from nemo.collections.llm.recipes.llama31_8b import pretrain_recipe
    elif arch == "llama21b":
        from .llama_21b import pretrain_recipe
    elif arch == "llama24b":
        from .llama_24b import pretrain_recipe
    elif arch == "llama70b":
        from nemo.collections.llm.recipes.llama31_70b import pretrain_recipe
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Set up performance mode if possible
    if performance_mode_if_possible:
        recipe_args["performance_mode"] = set_performance_mode_if_possible(arch)

    recipe = pretrain_recipe(**recipe_args)
    return recipe


def setup_parallelism(
    recipe,
    tensor_parallelism=None,
    pipeline_parallelism=None,
    context_parallelism=None,
    # sequence_parallelism=None,
):
    import torch

    if tensor_parallelism:
        recipe.trainer.strategy.tensor_model_parallel_size = tensor_parallelism
    if pipeline_parallelism:
        recipe.trainer.strategy.pipeline_model_parallel_size = pipeline_parallelism
    recipe.trainer.strategy.pipeline_dtype = torch.bfloat16
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = None
    if context_parallelism:
        recipe.trainer.strategy.context_parallel_size = context_parallelism
    if recipe.trainer.strategy.tensor_model_parallel_size == 1:
        recipe.trainer.strategy.sequence_parallel = False
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
        and tensor_parallelism is None
    ):
        logger.warning(
            f"Tensor parallelism is set to {recipe.trainer.strategy.tensor_model_parallel_size} which is greater than 4. We only have 4 GPUs per node. Setting tensor parallelism to 4."
        )
    recipe.trainer.strategy.ckpt_async_save = True
    return recipe


def get_time_limit(time_limit, buffer_minutes: int = 30) -> str:
    logging.info(time_limit)
    h, m, s = map(int, time_limit.split(":"))
    slurm_time_limit = datetime.timedelta(hours=h, minutes=m, seconds=s)
    td = slurm_time_limit - datetime.timedelta(minutes=buffer_minutes)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    return f"{td.days:02}:{hours:02}:{minutes:02}:00"
