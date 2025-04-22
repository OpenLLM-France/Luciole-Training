from collections import OrderedDict
import torch
import json
import os
from safetensors.torch import save_file as safe_save_file

from huggingface_hub import split_torch_state_dict_into_shards
from transformers import LlamaConfig
import argparse

from megatron.core import dist_checkpointing
from megatron.core import dist_checkpointing
from nemo.collections.llm.gpt.model.llama import Llama32Config1B
# from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer

# # print('🥳 model')
# print(model)

# common_state_dict = dist_checkpointing.load_common_state_dict(dist_ckpt_root)
# print("\n🥳 common_state_dict")
# print(common_state_dict)

# tensors_metadata = dist_checkpointing.load_tensors_metadata(dist_ckpt_root)
# print("\n🥳 tensors_metadata")
# print(tensors_metadata)

# plain_tensors = dist_checkpointing.load_plain_tensors(dist_ckpt_root)
# print("\n🥳 plain_tensors")
# print(plain_tensors)

# embedding = plain_tensors["module.embedding.word_embeddings.weight"]
# print("\n🥳 embedding")
# print(embedding.shape)

def convert(model_config, model_state_dict):
    dtype = torch.bfloat16
    param_to_weights = lambda param: param.to(dtype)
    checkpoint = OrderedDict()

    hidden_size = model_config.hidden_size
    head_num = model_config.num_attention_heads
    num_layers = model_config.num_layers
    ffn_hidden_size = model_config.ffn_hidden_size
    num_query_groups = model_config.num_query_groups  # different num_query_groups for 70B

    head_size =hidden_size // head_num  # equivalent to hf's head_dim
    heads_per_group = head_num // num_query_groups
    qkv_total_dim = head_num + 2 * num_query_groups

    # Embedding
    embed_weight = model_state_dict[f'module.embedding.word_embeddings.weight']
    embed_weights_base_name = f'embed_tokens.weight'
    checkpoint[embed_weights_base_name] = param_to_weights(embed_weight)

    for l in range(int(num_layers)):
        # print(f"converting layer {l}")

        qkv_weights = model_state_dict[f'module.decoder.layers.self_attention.linear_qkv.weight'][l, ...]
        qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

        q_slice = torch.cat(
            [
                torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                for i in range(num_query_groups)
            ]
        )
        k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
        v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))
        ## Example of slices
        ## 7b: num_query_groups = head_num = 32,
        ## q_slice = [0, 3, 6, 9 , ... 90, 93]
        ## k_slice = [1, 4, 7, 10, ... 91, 94]
        ## v_slice = [2, 5, 8, 11, ... 92, 95]
        ## 70b (with GQA): num_query_groups = 8, head_num = 64
        ## q_slice = [0, 1, .. 6, 7, 10, 11, .. 16, 17, 20, 21, .. 67, 70, ... 76, 77]
        ## k_slice = [8, 18, 28, ... 68, 78]
        ## v_slice = [9, 19, 29, ... 69, 79]

        q_weights_base_name = f'layers.{l}.self_attn.q_proj.weight'
        k_weights_base_name = f'layers.{l}.self_attn.k_proj.weight'
        v_weights_base_name = f'layers.{l}.self_attn.v_proj.weight'

        checkpoint[q_weights_base_name] = param_to_weights(qkv_weights[q_slice].reshape(-1, hidden_size))
        checkpoint[k_weights_base_name] = param_to_weights(qkv_weights[k_slice].reshape(-1, hidden_size))
        checkpoint[v_weights_base_name] = param_to_weights(qkv_weights[v_slice].reshape(-1, hidden_size))

        # attention dense
        o_weight = model_state_dict[f'module.decoder.layers.self_attention.linear_proj.weight'][l, ...]
        o_weight_base_name = f'layers.{l}.self_attn.o_proj.weight'
        checkpoint[o_weight_base_name] = param_to_weights(o_weight)

        # mlp
        mlp_weights = model_state_dict[f'module.decoder.layers.mlp.linear_fc1.weight'][l, ...]
        mlp_down_proj_weight = mlp_weights[:ffn_hidden_size, :]
        mlp_gate_proj_weight = mlp_weights[ffn_hidden_size:, :]

        mlp_down_proj_base_name = f'layers.{l}.mlp.gate_proj.weight'
        mlp_gate_proj_base_name = f'layers.{l}.mlp.up_proj.weight'

        checkpoint[mlp_down_proj_base_name] = param_to_weights(mlp_down_proj_weight)
        checkpoint[mlp_gate_proj_base_name] = param_to_weights(mlp_gate_proj_weight)

        mlp_up_proj_weight = model_state_dict[f'module.decoder.layers.mlp.linear_fc2.weight'][l, ...]
        mlp_up_proj_base_name = f'layers.{l}.mlp.down_proj.weight'
        checkpoint[mlp_up_proj_base_name] = param_to_weights(mlp_up_proj_weight)

        # layernorm
        input_ln_weight = model_state_dict[f'module.decoder.layers.self_attention.linear_qkv.layer_norm_weight'][l, ...]
        input_ln_base_name = f'layers.{l}.input_layernorm.weight'
        checkpoint[input_ln_base_name] = param_to_weights(input_ln_weight)

        post_attn_ln_weight = model_state_dict[f'module.decoder.layers.mlp.linear_fc1.layer_norm_weight'][l, ...]
        post_attn_ln_base_name = f'layers.{l}.post_attention_layernorm.weight'
        checkpoint[post_attn_ln_base_name] = param_to_weights(post_attn_ln_weight)

        # print(f"done layer {l}")

    final_ln_weight = model_state_dict[f'module.decoder.final_layernorm.weight']
    final_ln_base_name = f'norm.weight'
    checkpoint[final_ln_base_name] = param_to_weights(final_ln_weight)

    # output_layer_weight = model_state_dict[f'module.output_layer.weight']
    # output_layer_base_name = f'module.lm_head.weight'
    # checkpoint[output_layer_base_name] = param_to_weights(output_layer_weight)
    return checkpoint

def save_state_dict(state_dict, save_directory):
    state_dict_split = split_torch_state_dict_into_shards(state_dict)
    for filename, tensors in state_dict_split.filename_to_tensors.items():
        shard = {tensor: state_dict[tensor] for tensor in tensors}
        safe_save_file(
            shard,
            os.path.join(save_directory, filename),
            metadata={"format": "pt"},
        )
    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
            f.write(json.dumps(index, indent=2))

def save_llama_config(model_config, output_path):
    llama_config = LlamaConfig(
        attention_bias=False,
        attention_dropout=0.0,
        # bos_token_id=tokenizer.bos_token_id,
        # eos_token_id=tokenizer.eos_token_id,
        head_dim=model_config.hidden_size // model_config.num_attention_heads,
        hidden_act='silu',
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.ffn_hidden_size,
        model_type='llama',
        num_attention_heads=model_config.num_attention_heads,
        num_hidden_layers=model_config.num_layers,
        num_key_value_heads=model_config.num_query_groups,
        rope_scaling= {
            "factor": float(model_config.scale_factor),
            "high_freq_factor": float(model_config.high_freq_factor),
            "low_freq_factor": float(model_config.low_freq_factor),
            "original_max_position_embeddings": model_config.old_context_len,
            "rope_type": "llama3"
        },
        max_position_embeddings=131072, # model_config.seq_length,
        rope_theta=model_config.rotary_base,
        tie_word_embeddings=True,
        use_cache=True,
        vocab_size=65024,
        torch_dtype='bfloat16',
        initializer_range=model_config.init_method_std,
    )
    output_config = llama_config.to_diff_dict()
    output_config["architectures"] = ["LlamaForCausalLM"]
    output_config["model_type"] = "llama"
    output_config_file = os.path.join(output_path, "config.json")

    print(f'Saving config to "{output_config_file}"')
    with open(output_config_file, "w") as f:
        json.dump(output_config, f, indent=2)

if "__main__" == "__main__":
    parser = argparse.ArgumentParser(description="Convert NeMo checkpoint to Hugging Face format")    
    parser.add_argument("--local-rank")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for the converted checkpoint")
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    torch.distributed.init_process_group()

    model_config = Llama32Config1B()
    model_state_dict = dist_checkpointing.load_plain_tensors(os.path.join(input_path, 'weights'))

    # Print some info
    print("\n🥳 model_state_dict")
    for k, v in model_state_dict.items():
        if k.startswith("module."):
            print(f"{k}: {v.shape}")
        else:
            print(k)

    # Model config
    save_llama_config(model_config, output_path)

    # Model weights
    converted_checkpoint = convert(model_config, model_state_dict)
    save_state_dict(converted_checkpoint, output_path)