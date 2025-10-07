from nemo.collections.llm.recipes.mistral_7b import (
    pretrain_recipe as pretrain_base_recipe,
)
import torch


def pretrain_recipe(**kwargs):
    recipe = pretrain_base_recipe(**kwargs)
    recipe.num_layers = 40
    recipe.hidden_size = 5120
    recipe.ffn_hidden_size = 32768
    recipe.num_attention_heads = 32
    recipe.kv_channels = 128
    # recipe.seq_length = 32768
    recipe.window_size = None
    recipe.cp_comm_type = None
    recipe.rotary_percent = 1.0
    recipe.rotary_base = 100000000.0
    recipe.params_dtype = torch.bfloat16
    return recipe
