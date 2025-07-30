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
    elif arch == "nemotron4b":
        from nemo.collections.llm.recipes.nemotron3_4b import pretrain_recipe
    elif arch == "nemotron8b":
        from nemo.collections.llm.recipes.nemotron3_8b import pretrain_recipe
        if performance_mode_if_possible:
            recipe_args["performance_mode"] = True
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return pretrain_recipe, recipe_args
