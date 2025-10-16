import logging

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
    "nemotron20b_wider_v2",
    "nemotron20b_deeper",
    "nemotron23b_wider_v2",
    "nemotron23b",
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
    elif arch == "nemotron20b_wider_v2":
        from .nemotron_20b_wider_v2 import pretrain_recipe
    elif arch == "nemotron20b_deeper":
        from .nemotron_20b_deeper import pretrain_recipe
    elif arch == "nemotron23b_wider_v2":
        from .nemotron_23b_wider_v2 import pretrain_recipe
    elif arch == "nemotron23b":
        from .nemotron_23b import pretrain_recipe
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
