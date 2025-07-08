import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_recipe(arch, recipe_args, performance_mode_if_possible=True):
    if arch == "llama1b":
        from nemo.collections.llm.recipes.llama32_1b import pretrain_recipe
    elif arch == "llama3b":
        from nemo.collections.llm.recipes.llama32_3b import pretrain_recipe
    elif arch == "llama8b":
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
