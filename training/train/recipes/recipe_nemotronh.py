import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_recipe(arch, recipe_args, performance_mode_if_possible=True):
    if arch == "nemotronh8b":
        from nemo.collections.llm.recipes.nemotronh_8b import pretrain_recipe
    elif arch == "nemotronh47b":
        from nemo.collections.llm.recipes.nemotronh_47b import pretrain_recipe
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return pretrain_recipe, recipe_args
