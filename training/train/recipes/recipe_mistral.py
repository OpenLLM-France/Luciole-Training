import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_recipe(arch, recipe_args, performance_mode_if_possible=True):
    if arch == "mixtral8x7":
        from nemo.collections.llm.recipes.mixtral_8x7b import pretrain_recipe

        if performance_mode_if_possible:
            recipe_args["performance_mode"] = True
    elif arch == "mistral12b":
        from nemo.collections.llm.recipes.mistral_nemo_12b import pretrain_recipe

        if performance_mode_if_possible:
            from nemo.collections.llm.recipes.mistral_nemo_12b import (
                pretrain_recipe_performance as pretrain_recipe,
            )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return pretrain_recipe, recipe_args
