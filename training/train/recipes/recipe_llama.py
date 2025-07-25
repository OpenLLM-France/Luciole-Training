import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_recipe(arch, recipe_args, performance_mode_if_possible=False):
    if arch == "llama1b":
        from nemo.collections.llm.recipes.llama32_1b import pretrain_recipe
    elif arch == "llama3b":
        from nemo.collections.llm.recipes.llama32_3b import pretrain_recipe
    elif arch in ["llama8b", "llama24b"]:
        from nemo.collections.llm.recipes.llama31_8b import pretrain_recipe

        if performance_mode_if_possible:
            recipe_args["performance_mode"] = True
    elif arch == "llama70b":
        from nemo.collections.llm.recipes.llama31_70b import pretrain_recipe

        if performance_mode_if_possible:
            recipe_args["performance_mode"] = True
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return pretrain_recipe, recipe_args


def set_llama24b_recipe(recipe, args):
    # use llama8b recipe as base and use nemotron22b for model size
    recipe.model.config.num_layers = 40
    recipe.model.config.num_attention_heads = 48
    recipe.model.config.num_query_groups = 8
    recipe.model.config.hidden_size = 6144
    recipe.model.config.ffn_hidden_size = 24576
    if recipe.data.seq_length >= 8192 and not not args.context_parallelism:
        recipe.trainer.strategy.context_parallel_size = 2
    elif not args.context_parallelism:
        recipe.trainer.strategy.context_parallel_size = 1
    if not args.tensor_parallelism:
        recipe.trainer.strategy.tensor_model_parallel_size = 2
    if not args.pipeline_parallelism:
        recipe.trainer.strategy.pipeline_model_parallel_size = 4
    return recipe
