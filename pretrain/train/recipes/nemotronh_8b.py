from nemo.collections.llm.recipes.nemotronh_8b import (
    pretrain_recipe as pretrain_base_recipe,
)


def pretrain_recipe(**kwargs):
    recipe = pretrain_base_recipe(**kwargs)
    recipe.trainer.strategy.tensor_model_parallel_size = 1
    return recipe
