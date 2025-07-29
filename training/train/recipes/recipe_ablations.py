def get_recipe(arch, recipe_args, performance_mode_if_possible=False):
    if arch == "ablation_llama90M":
        from nemo.collections.llm.recipes.llama32_1b import pretrain_recipe
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return pretrain_recipe, recipe_args

def set_ablation_recipe(recipe, args):
    recipe.model.config.num_layers = 6
    recipe.model.config.hidden_size = 512
    recipe.model.config.ffn_hidden_size = 2048
    recipe.model.config.num_attention_heads = 8
    recipe.model.config.num_query_groups = 8
    return recipe