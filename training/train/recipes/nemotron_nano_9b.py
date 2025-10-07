from nemo.collections.llm.recipes.nemotronh_8b import (
    pretrain_recipe as pretrain_base_recipe,
)


def pretrain_recipe(**kwargs):
    recipe = pretrain_base_recipe(**kwargs)
    recipe.hybrid_override_pattern = (
        "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"
    )
    recipe.num_layers = 56
    recipe.hidden_size = 4480
    recipe.mamba_num_heads = 128
    recipe.kv_channels = 128
    recipe.mamba_state_dim = 128
    recipe.ffn_hidden_size = 15680
    recipe.num_attention_heads = 40
    recipe.mamba_head_dim = 80
    return recipe
