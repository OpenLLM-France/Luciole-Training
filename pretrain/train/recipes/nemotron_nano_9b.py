from nemo.collections.llm.recipes.nemotronh_8b import (
    pretrain_recipe as pretrain_base_recipe,
)


def pretrain_recipe(**kwargs):
    recipe = pretrain_base_recipe(**kwargs)
    recipe.model.config.hybrid_override_pattern = (
        "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"
    )
    recipe.model.config.num_layers = 56
    recipe.model.config.hidden_size = 4480
    recipe.model.config.mamba_num_heads = 128
    recipe.model.config.kv_channels = 128
    recipe.model.config.mamba_state_dim = 128
    recipe.model.config.ffn_hidden_size = 15680
    recipe.model.config.num_attention_heads = 40
    recipe.model.config.mamba_head_dim = 80
    return recipe
