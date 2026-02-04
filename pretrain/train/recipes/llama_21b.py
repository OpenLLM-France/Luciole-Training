from nemo.collections.llm.recipes.llama31_8b import (
    pretrain_recipe as pretrain_base_recipe,
)


def pretrain_recipe(**kwargs):
    recipe = pretrain_base_recipe(**kwargs)
    recipe.model.config.num_layers = 40
    recipe.model.config.num_attention_heads = 48
    recipe.model.config.num_query_groups = 8
    recipe.model.config.hidden_size = 6144
    recipe.model.config.ffn_hidden_size = 21504
    return recipe
