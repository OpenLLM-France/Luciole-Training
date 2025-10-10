from nemo.collections.llm.recipes.nemotron3_22b import (
    pretrain_recipe as pretrain_base_recipe,
)


def pretrain_recipe(**kwargs):
    recipe = pretrain_base_recipe(**kwargs)
    recipe.model.config.num_attention_heads = 56 # 48
    recipe.model.config.num_query_groups = 8
    recipe.model.config.hidden_size = 7168 # 6656
    recipe.model.config.ffn_hidden_size = 28672 # 26624
    # Parallelism
    recipe.trainer.strategy.tensor_model_parallel_size = 4
    recipe.trainer.strategy.pipeline_model_parallel_size = 2
    return recipe
