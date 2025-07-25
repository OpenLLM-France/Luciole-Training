import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_recipe(arch, recipe_args, performance_mode_if_possible=False):
    if arch == "qwen32b":
        from nemo.collections.llm.recipes.qwen25_32b import pretrain_recipe
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return pretrain_recipe, recipe_args
