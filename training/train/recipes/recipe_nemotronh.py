import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_recipe(arch, recipe_args, performance_mode_if_possible=False):
    if arch == "nemotronh8b":
        from nemo.collections.llm.recipes.nemotronh_8b import pretrain_recipe
    elif arch == "nemotronh47b":
        from nemo.collections.llm.recipes.nemotronh_47b import pretrain_recipe

        if performance_mode_if_possible:
            recipe_args["performance_mode"] = True
    elif arch == "nemotron22b":
        from nemo.collections.llm.recipes.nemotron3_22b import pretrain_recipe

        if performance_mode_if_possible:
            recipe_args["performance_mode"] = True
    elif arch == "nemotron4b" or arch == "nemotron1b":
        from nemo.collections.llm.recipes.nemotron3_4b import pretrain_recipe
    elif arch == "nemotron8b":
        from nemo.collections.llm.recipes.nemotron3_8b import pretrain_recipe

        if performance_mode_if_possible:
            recipe_args["performance_mode"] = True
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return pretrain_recipe, recipe_args


def set_nemotron1b_recipe(recipe, args):
    recipe.model.config.num_layers = 16
    recipe.model.config.num_attention_heads = 32
    recipe.model.config.num_query_groups = 8
    recipe.model.config.hidden_size = 2048
    # recipe.model.config.ffn_hidden_size = 8192
    recipe.model.config.ffn_hidden_size = 12288
    recipe.model.config.kv_channels = None
    recipe.model.config.share_embeddings_and_output_weights = True
    recipe.data.seq_length = 4096
    recipe.data.global_batch_size = 1024
    recipe.trainer.strategy.context_parallel_size = 1
    recipe.trainer.strategy.tensor_model_parallel_size = 1
    return recipe
