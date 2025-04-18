# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch

# from omegaconf import open_dict
from nemo import lightning as nl
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaTokenizerFast,
    convert_slow_tokenizer,
)
from typing import Optional

# from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
# from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from nemo.utils import logging
from nemo.utils.exp_manager import TimingCallback

from nemo.lightning import io

# from nemo.lightning.io.pl import ckpt_to_weights_subdir
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.collections import llm
from lightning.pytorch.callbacks.callback import Callback
from megatron.core.distributed import DistributedDataParallelConfig
from nemo.lightning.pytorch.plugins.mixed_precision import MegatronMixedPrecision

"""
Script to convert a llama checkpoint in nemo (mcore path) into a HuggingFace checkpoint.
This script can be used to 1) generate only the HF weights, or 2) generate an entire HF model folder.

1) Generate only HF weights from a nemo file:

    python convert_llama_nemo_to_hf.py \
    --input_name_or_path /path/to/file.nemo or /path/to/extracted_folder \
    --output_path /path/to/pytorch_model.bin
    
2) Generate the full HF model folder

    python convert_llama_nemo_to_hf.py \
    --input_name_or_path /path/to/file.nemo or /path/to/extracted_folder \
    --output_path /path/to/pytorch_model.bin \
    --hf_input_path /path/to/input_hf_folder \
    --hf_output_path /path/to/output_hf_folder

3) Generate the full HF model folder with a custom tokenizer

    python convert_llama_nemo_to_hf.py \
    --input_name_or_path /path/to/file.nemo or /path/to/extracted_folder \
    --output_path /path/to/pytorch_model.bin \
    --hf_input_path /path/to/input_hf_folder \
    --hf_output_path /path/to/output_hf_folder \
    --input_tokenizer /path/to/custom_nemo_tokenizer.model \
    --hf_output_tokenizer /path/to/output_tokenizer

    Use the --cpu-only flag if the model cannot fit in the GPU (e.g. Llama2 70b). 
    However this option makes the conversion script significantly slower.
"""


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to a checkpoint folder",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help="Path to HF .bin file",
    )
    parser.add_argument(
        "--hf_input_path",
        type=str,
        default=None,
        help="A HF model path, "
        "e.g. a folder containing https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main",
    )
    parser.add_argument("--local-rank")
    parser.add_argument(
        "--hf_output_path",
        type=str,
        default=None,
        help="Output HF model path, "
        "with the same format as above but user's own weights",
    )
    parser.add_argument(
        "--input_tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer used for the input nemo model. (need to extract the .nemo file first)",
    )
    parser.add_argument(
        "--hf_output_tokenizer",
        type=str,
        default=None,
        help="Path to save the tokenizer used for the output HF model.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Precision of output weights."
        "Defaults to precision of the input nemo weights (model.config.trainer.precision)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Load model in cpu only. Useful if the model cannot fit in GPU memory, "
        "but this option makes the conversion script significantly slower.",
    )
    args = parser.parse_args()
    return args


def create_trainer(
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    num_nodes: int = 1,
    num_gpus_per_node: int = 1,
    max_steps: int = 1168251,
    val_check_interval: int = 1000,
    limit_val_batches: int = 0,
    callbacks: Optional[list[Callback]] = [TimingCallback()],
):
    """
    Configure the NeMo Lightning Trainer for Llama3.2 1B model.

    Args:
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_type (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        max_steps (int): Maximum number of training steps.
        callbacks (Optional[list[run.Config[Callback]]]): List of callback configurations.

    Returns:
        run.Config[nl.Trainer]: Configuration for the NeMo Lightning Trainer.

    Examples:
        CLI usage:
            $ nemo llm pretrain trainer=llama32_1b ...

        Python API usage:
            >>> trainer_config = trainer(num_nodes=1, num_gpus_per_node=1)
            >>> print(trainer_config)

    Note:
        This configuration uses extensive parallelism to handle the large model size efficiently.
    """
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_type,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        gradient_as_bucket_view=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
        ),
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        accumulate_grad_batches=1,
        callbacks=callbacks,
        devices=num_gpus_per_node,
        limit_test_batches=50,
        log_every_n_steps=10,
        max_steps=max_steps,
        num_nodes=num_nodes,
        plugins=MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=True,
        ),
        strategy=strategy,
        use_distributed_sampler=False,
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=2,  # not sure it works
    )

    return trainer


def convert(
    input_checkpoint_path, output_hf_file, precision=None, cpu_only=False
) -> None:
    """
    Convert NeMo weights to HF weights
    """
    # dummy_trainer = Trainer(devices=1, accelerator='cpu', callbacks=[TimingCallback()], strategy=nl.MegatronStrategy())
    dummy_trainer = create_trainer()
    model: io.TrainerContext = io.load_context(
        path=ckpt_to_context_subdir(input_checkpoint_path), subpath="model"
    )
    llm.inference.base._setup_trainer_and_restore_model(
        path=input_checkpoint_path, trainer=dummy_trainer, model=model
    )
    # model = input_checkpoint_path
    # model_config = MegatronGPTModel.restore_from(input_nemo_file, trainer=dummy_trainer, return_config=True)
    # model_config.tensor_model_parallel_size = 1
    # model_config.pipeline_model_parallel_size = 1
    # model_config.name = "te_gpt"
    # if cpu_only:
    #     map_location = torch.device('cpu')
    #     model_config.use_cpu_initialization = True
    # else:
    #     map_location = None

    # if cpu_only:
    #     logging.info("******** Loading model on CPU. This will take a significant amount of time.")
    # model = MegatronGPTModel.restore_from(
    #     input_nemo_file, trainer=dummy_trainer, override_config_path=model_config, map_location=map_location
    # )
    precision = "bf16"
    if precision is None:
        precision = model.config.precision
    if precision in [32, "32"]:
        dtype = torch.float32
    elif precision in [16, "16", "16-mixed"]:
        dtype = torch.float16
    elif precision in ["bf16", "bf16-mixed"]:
        dtype = torch.bfloat16
    else:
        logging.warning(
            f"Precision string {precision} is not recognized, falling back to fp32"
        )
        dtype = torch.float32  # fallback
    logging.info(f"Using precision {dtype}")

    def param_to_weights(param):
        return param.to(dtype)

    # param_to_weights = lambda param: param.to(dtype)

    checkpoint = OrderedDict()

    hidden_size = model.config.hidden_size
    head_num = model.config.num_attention_heads
    num_layers = model.config.num_layers
    ffn_hidden_size = model.config.ffn_hidden_size
    num_query_groups = (
        model.config.num_query_groups
    )  # different num_query_groups for 70B

    head_size = hidden_size // head_num  # equivalent to hf's head_dim
    heads_per_group = head_num // num_query_groups
    qkv_total_dim = head_num + 2 * num_query_groups

    # Embedding
    embed_weight = model.state_dict()["module.embedding.word_embeddings.weight"]
    embed_weights_base_name = "module.embed_tokens.weight"
    checkpoint[embed_weights_base_name] = param_to_weights(embed_weight)

    for layer in range(int(num_layers)):
        print(f"converting layer {layer}")

        qkv_weights = model.state_dict()[
            f"module.decoder.layers.{layer}.self_attention.linear_qkv.weight"
        ]
        qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

        q_slice = torch.cat(
            [
                torch.arange(
                    (heads_per_group + 2) * i,
                    (heads_per_group + 2) * i + heads_per_group,
                )
                for i in range(num_query_groups)
            ]
        )
        k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
        v_slice = torch.arange(
            heads_per_group + 1, qkv_total_dim, (heads_per_group + 2)
        )
        ## Example of slices
        ## 7b: num_query_groups = head_num = 32,
        ## q_slice = [0, 3, 6, 9 , ... 90, 93]
        ## k_slice = [1, 4, 7, 10, ... 91, 94]
        ## v_slice = [2, 5, 8, 11, ... 92, 95]
        ## 70b (with GQA): num_query_groups = 8, head_num = 64
        ## q_slice = [0, 1, .. 6, 7, 10, 11, .. 16, 17, 20, 21, .. 67, 70, ... 76, 77]
        ## k_slice = [8, 18, 28, ... 68, 78]
        ## v_slice = [9, 19, 29, ... 69, 79]

        q_weights_base_name = f"module.layers.{layer}.self_attn.q_proj.weight"
        k_weights_base_name = f"module.layers.{layer}.self_attn.k_proj.weight"
        v_weights_base_name = f"module.layers.{layer}.self_attn.v_proj.weight"

        checkpoint[q_weights_base_name] = param_to_weights(
            qkv_weights[q_slice].reshape(-1, hidden_size)
        )
        checkpoint[k_weights_base_name] = param_to_weights(
            qkv_weights[k_slice].reshape(-1, hidden_size)
        )
        checkpoint[v_weights_base_name] = param_to_weights(
            qkv_weights[v_slice].reshape(-1, hidden_size)
        )

        # attention dense
        o_weight = model.state_dict()[
            f"module.decoder.layers.{layer}.self_attention.linear_proj.weight"
        ]
        o_weight_base_name = f"module.layers.{layer}.self_attn.o_proj.weight"
        checkpoint[o_weight_base_name] = param_to_weights(o_weight)

        # mlp
        mlp_weights = model.state_dict()[
            f"module.decoder.layers.{layer}.mlp.linear_fc1.weight"
        ]
        mlp_down_proj_weight = mlp_weights[:ffn_hidden_size, :]
        mlp_gate_proj_weight = mlp_weights[ffn_hidden_size:, :]

        mlp_down_proj_base_name = f"module.layers.{layer}.mlp.gate_proj.weight"
        mlp_gate_proj_base_name = f"module.layers.{layer}.mlp.up_proj.weight"

        checkpoint[mlp_down_proj_base_name] = param_to_weights(mlp_down_proj_weight)
        checkpoint[mlp_gate_proj_base_name] = param_to_weights(mlp_gate_proj_weight)

        mlp_up_proj_weight = model.state_dict()[
            f"module.decoder.layers.{layer}.mlp.linear_fc2.weight"
        ]
        mlp_up_proj_base_name = f"module.layers.{layer}.mlp.down_proj.weight"
        checkpoint[mlp_up_proj_base_name] = param_to_weights(mlp_up_proj_weight)

        # layernorm
        input_ln_weight = model.state_dict()[
            f"module.decoder.layers.{layer}.self_attention.linear_qkv.layer_norm_weight"
        ]
        input_ln_base_name = f"module.layers.{layer}.input_layernorm.weight"
        checkpoint[input_ln_base_name] = param_to_weights(input_ln_weight)

        post_attn_ln_weight = model.state_dict()[
            f"module.decoder.layers.{layer}.mlp.linear_fc1.layer_norm_weight"
        ]
        post_attn_ln_base_name = (
            f"module.layers.{layer}.post_attention_layernorm.weight"
        )
        checkpoint[post_attn_ln_base_name] = param_to_weights(post_attn_ln_weight)

        print(f"done layer {layer}")

    final_ln_weight = model.state_dict()["module.decoder.final_layernorm.weight"]
    final_ln_base_name = "module.norm.weight"
    checkpoint[final_ln_base_name] = param_to_weights(final_ln_weight)

    output_layer_weight = model.state_dict()["module.embedding.word_embeddings.weight"]
    output_layer_base_name = "lm_head.weight"
    checkpoint[output_layer_base_name] = param_to_weights(output_layer_weight)

    os.makedirs(os.path.dirname(output_hf_file), exist_ok=True)
    torch.save(checkpoint, output_hf_file)
    logging.info(f"Weights saved to {output_hf_file}")

    return dtype


def replace_hf_weights_and_tokenizer(
    weights_file,
    dtype,
    input_hf_path,
    output_hf_path,
    tokenizer_path,
    output_hf_tokenizer,
):
    model = AutoModelForCausalLM.from_pretrained(
        input_hf_path,
        local_files_only=True,
        torch_dtype=dtype,
    )
    nemo_exported = torch.load(weights_file)

    if tokenizer_path:
        try:
            tokenizer = LlamaTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True,
                legacy=False,
            )
            tmp_tokenizer = convert_slow_tokenizer.convert_slow_tokenizer(tokenizer)
            fast_tokenizer = LlamaTokenizerFast(tokenizer_object=tmp_tokenizer)
            tokenizer_length = len(fast_tokenizer)
            model.resize_token_embeddings(tokenizer_length)
        except Exception:
            tokenizer = None
            logging.warning(
                "Could not load custom tokenizer, proceeding with default tokenizer"
            )

    model.load_state_dict(nemo_exported)
    model.save_pretrained(output_hf_path)
    logging.info(f"Full HF model saved to {output_hf_path}")

    if tokenizer_path and (tokenizer is not None):
        fast_tokenizer.save_pretrained(output_hf_tokenizer)
        tokenizer.save_pretrained(output_hf_tokenizer)
        logging.info(f"Tokenizer saved to {output_hf_tokenizer}")


if __name__ == "__main__":
    args = get_args()
    if not args.hf_output_tokenizer and args.hf_output_path:
        args.hf_output_tokenizer = args.hf_output_path
    dtype = convert(
        args.input_name_or_path,
        args.output_path,
        precision=args.precision,
        cpu_only=args.cpu_only,
    )
    if args.hf_input_path and args.hf_output_path:
        replace_hf_weights_and_tokenizer(
            args.output_path,
            dtype,
            args.hf_input_path,
            args.hf_output_path,
            args.input_tokenizer,
            args.hf_output_tokenizer,
        )
    else:
        logging.info(
            "`hf_input_path` and/or `hf_output_path` not provided, not generating full HF model."
        )
        logging.info(f".bin file is saved to {args.output_path}")
