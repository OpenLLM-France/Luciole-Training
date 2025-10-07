from nemo.collections.llm.recipes.nemotron3_22b import (
    pretrain_recipe as pretrain_base_recipe,
)


def pretrain_recipe(**kwargs):
    recipe = pretrain_base_recipe(**kwargs)
    recipe.model.config.num_layers = 48
    recipe.model.config.num_query_groups = 8
    return recipe
