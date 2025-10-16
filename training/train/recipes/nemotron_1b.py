from nemo.collections.llm.recipes.nemotron3_4b import (
    pretrain_recipe as pretrain_base_recipe,
)


def pretrain_recipe(**kwargs):
    recipe = pretrain_base_recipe(**kwargs)
    recipe.model.config.num_layers = 24
    recipe.model.config.num_attention_heads = 32
    recipe.model.config.num_query_groups = 8
    recipe.model.config.hidden_size = 2048
    recipe.model.config.ffn_hidden_size = 8192
    recipe.model.config.kv_channels = None
    recipe.model.config.share_embeddings_and_output_weights = True
    # Parallelism
    recipe.trainer.strategy.context_parallel_size = 1
    recipe.trainer.strategy.tensor_model_parallel_size = 1
    return recipe
