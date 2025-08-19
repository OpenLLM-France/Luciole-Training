def get_recipe(arch, recipe_args, performance_mode_if_possible=False):
    from nemo.collections.llm.recipes.llama32_1b import pretrain_recipe
    return pretrain_recipe, recipe_args

def set_ablation_recipe(recipe, arch):
    if arch=="ablation_llama90m":
        recipe.model.config.num_layers = 6
        recipe.model.config.hidden_size = 512
        recipe.model.config.ffn_hidden_size = 2048
        recipe.model.config.num_attention_heads = 8
        recipe.model.config.num_query_groups = 8
    elif arch=="ablation_llama210m":
        recipe.model.config.num_layers = 12
        recipe.model.config.hidden_size = 768
        recipe.model.config.ffn_hidden_size = 3072
        recipe.model.config.num_attention_heads = 12
        recipe.model.config.num_query_groups = 12
    elif arch=="ablation_llama400m":
        recipe.model.config.num_layers = 16
        recipe.model.config.hidden_size = 1024
        recipe.model.config.ffn_hidden_size = 4096
        recipe.model.config.num_attention_heads = 16
        recipe.model.config.num_query_groups = 16
    elif arch=="ablation_llama530m":
        recipe.model.config.num_layers = 24
        recipe.model.config.hidden_size = 1024
        recipe.model.config.ffn_hidden_size = 4096
        recipe.model.config.num_attention_heads = 16
        recipe.model.config.num_query_groups = 16
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    recipe.data.seq_length = 4096
    recipe.data.global_batch_size = 256
    return recipe