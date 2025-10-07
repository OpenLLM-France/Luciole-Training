from nemo.collections.llm.recipes.nemotron3_22b import (
    pretrain_recipe as pretrain_base_recipe,
)


def pretrain_recipe(**kwargs):
    recipe = pretrain_base_recipe(**kwargs)
    recipe.model.config.num_attention_heads = 52
    recipe.model.config.num_query_groups = 8
    recipe.model.config.hidden_size = 6656
    recipe.model.config.ffn_hidden_size = 26624
    return recipe
