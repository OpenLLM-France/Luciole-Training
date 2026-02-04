from nemo.collections.llm.recipes.mistral_7b import (
    pretrain_recipe as pretrain_base_recipe,
)
import torch


def pretrain_recipe(**kwargs):
    recipe = pretrain_base_recipe(**kwargs)
    # Model
    recipe.model.config.num_layers = 40
    recipe.model.config.hidden_size = 5120
    recipe.model.config.ffn_hidden_size = 32768
    recipe.model.config.num_attention_heads = 32
    recipe.model.config.kv_channels = 128
    # recipe.model.config.seq_length = 32768
    recipe.model.config.window_size = None
    recipe.model.config.cp_comm_type = None
    recipe.model.config.rotary_percent = 1.0
    recipe.model.config.rotary_base = 100000000.0
    recipe.model.config.params_dtype = torch.bfloat16
    # Parallelism
    recipe.trainer.strategy.tensor_model_parallel_size = 4
    recipe.trainer.strategy.pipeline_model_parallel_size = 2
    return recipe
