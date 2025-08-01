from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional

import torch
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTModel, torch_dtype_from_mcore_config
from nemo.collections.llm.gpt.model.qwen2 import Qwen2Config
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.io.state import TransformFns
from nemo.lightning.pytorch.utils import dtype_from_hf


@dataclass
class Qwen3Config(Qwen2Config):
    """
    Base config for Qwen 3 Models
    """

    add_qkv_bias: bool = False
    qk_layernorm: bool = True
    kv_channels: Optional[int] = 128
    num_query_groups: int = 8
    max_position_embeddings: int = 40960
    vocab_size: int = 151936
    

@dataclass
class Qwen3MoEConfig(Qwen3Config):
    """
    Base config for Qwen 3 MoE Models
    """

    num_moe_experts: int = 128
    moe_router_load_balancing_type: str = "aux_loss"
    moe_aux_loss_coeff: float = 1e-3
    moe_router_topk: int = 8
    moe_router_pre_softmax: bool = False
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True

@dataclass
class Qwen3Config30B_A3B(Qwen3MoEConfig):
    """
    Config for Qwen 3 30B-A3B: https://huggingface.co/Qwen/Qwen3-30B-A3B
    """

    num_layers: int = 48
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_query_groups: int = 4
    ffn_hidden_size: int = 6144
    moe_ffn_hidden_size: int = 768



class Qwen3Model(GPTModel):
    """
    Base model for Qwen 3
    """

    def __init__(
        self,
        config: Annotated[Optional[Qwen3Config], Config[Qwen3Config]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or Qwen3Config(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)

from typing import Optional

import lightning.pytorch as pl
import nemo_run as run
import torch
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.peft import PEFT_STR2CLS
from nemo.collections.llm.recipes.finetune_default import default_finetune_recipe
from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.utils.exp_manager import TimingCallback

NAME = "qwen3_30b_a3b"

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional

import lightning.pytorch as pl
import nemo_run as run
import torch
from lightning.pytorch.callbacks.callback import Callback
from megatron.core.distributed import DistributedDataParallelConfig

from nemo import lightning as nl
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed, fp16_mixed


def qwen3_model() -> run.Config[pl.LightningModule]:
    """
    A function to create a qwen3 models.

    Args:
        version (str): The version of the qwen3 model to create. One of ["qwen3_600m", "qwen3_1p7b",
            "qwen3_4b", "qwen3_8b", "qwen3_14b", "qwen3_32b", "qwen3_30b_a3b", "qwen3_235b_a22b"].

    Returns:
        run.Config[pl.LightningModule]: Configuration for the qwen3 model.
    """
    config = run.Config(Qwen3Config30B_A3B)
    return run.Config(Qwen3Model, config=config)


def qwen3_trainer(
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    expert_parallelism: int = 1,
    account_for_embedding_in_pipeline_split: bool = False,
    account_for_loss_in_pipeline_split: bool = False,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    precision: str = "bf16-mixed",
    accumulate_grad_batches: int = 1,
    limit_test_batches: int = 32,
    limit_val_batches: int = 32,
    log_every_n_steps: int = 10,
    val_check_interval: int = 2000,
    callbacks: Optional[list[run.Config[Callback]]] = None,
) -> run.Config[nl.Trainer]:
    """
    Configure the NeMo Lightning Trainer for qwen3 models.

    This function sets up the distributed training strategy and other training parameters.

    Args:
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_type (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        expert_parallelism (Optional[int]): Degree of expert parallelism.
        account_for_embedding_in_pipeline_split (bool): Whether to treat input embedding layer as a standard
            transformer layer in the context of partition and placement for pipeline parallelism.
        account_for_loss_in_pipeline_split (bool): Whether to treat loss layer as a standard transformer
            layer in the context of partition and placement for pipeline parallelism.
        account_for_loss_in_pipeline_split (bool): = False,
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        max_steps (int): Maximum number of training steps.
        precision (str): Precision configuration, one of fp32, 16-mixed or bf16-mixed.
        accumulate_grad_batches (int): Number of steps per gradient accumulation.
        limit_test_batches (int): Limit the number of test batches.
        limit_val_batches (int): Limit the number of validation batches.
        log_every_n_steps (int): Log every n steps.
        val_check_interval (int): Run validation every N steps.
        callbacks (Optional[list[run.Config[Callback]]]): List of callback configurations.

    Returns:
        run.Config[nl.Trainer]: Configuration for the NeMo Lightning Trainer.
    """
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_type,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        expert_model_parallel_size=expert_parallelism,
        expert_tensor_parallel_size=1,
        account_for_embedding_in_pipeline_split=account_for_embedding_in_pipeline_split,
        account_for_loss_in_pipeline_split=account_for_loss_in_pipeline_split,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,  # Not supported for custom FSDP for now, need to be set to False if using FSDP
            data_parallel_sharding_strategy="optim_grads_params",  # For custom FSDP only
        ),
    )

    precision_plugin = None
    if precision == "16-mixed":
        precision_plugin = fp16_mixed()
    elif precision == "bf16-mixed":
        precision_plugin = bf16_mixed()

    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        callbacks=callbacks,
        devices=num_gpus_per_node,
        accumulate_grad_batches=accumulate_grad_batches,
        limit_test_batches=limit_test_batches,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=log_every_n_steps,
        max_steps=max_steps,
        num_nodes=num_nodes,
        plugins=precision_plugin,
        strategy=strategy,
        use_distributed_sampler=False,
        val_check_interval=val_check_interval,
    )

    return trainer


@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    # General
    dir: Optional[str] = None,
    name: str = "default",
    # Trainer
    tensor_parallelism: int = 4,  # Default for 30B-A3B model
    pipeline_parallelism: int = 2,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    expert_parallelism: Optional[int] = 4,
    sequence_parallelism: bool = True,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    max_steps: int = 300000,
    precision: str = "bf16-mixed",
    accumulate_grad_batches: int = 1,
    gradient_clip_val: float = 1.0,
    limit_test_batches: int = 32,
    limit_val_batches: int = 32,
    log_every_n_steps: int = 10,
    val_check_interval: int = 500,
    # Data
    global_batch_size=32,
    micro_batch_size=2,
    seq_length=4096,
    # Optimizer
    warmup_steps=500,
    constant_steps=0,
    min_lr=3e-5,
    max_lr=3e-4,
    # Training function
    fn=pretrain,
) -> run.Partial:
    """
    Create a pre-training recipe for Qwen3 30B-A3B model.

    This function sets up a complete configuration for pre-training, including
    model, trainer, data, logging, optimization, and resumption settings.
    This model uses Mixture of Experts (MoE) architecture with 128 experts.

    Args:
        dir (Optional[str]): Directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_type (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        num_nodes (int): Number of compute nodes to use.
        num_gpus_per_node (int): Number of GPUs per node.
        max_steps (int): Maximum number of training steps.
        precision (str): Precision configuration, one of fp32, 16-mixed or bf16-mixed.
        accumulate_grad_batches (int): Number of steps per gradient accumulation.
        gradient_clip_val (float): Value for gradient clipping.
        limit_test_batches (int): Limit the number of test batches.
        limit_val_batches (int): Limit the number of validation batches.
        log_every_n_steps (int): Log every n steps.
        val_check_interval (int): Run validation every N steps.
        global_batch_size (int): Global batch size.
        micro_batch_size (int): Micro batch size.
        seq_length (int): Sequence length.
        warmup_steps (int): Number of warmup steps.
        constant_steps (int): Number of constant steps.
        min_lr (float): Minimum learning rate.
        max_lr (float): Maximum learning rate.
        fn (Callable): The pre-training function to use.

    Returns:
        run.Partial: Partial configuration for pre-training.

    Examples:
        CLI usage:
            $ nemo llm pretrain --factory qwen3_30b_a3b
            $ nemo llm pretrain --factory "qwen3_30b_a3b(num_nodes=1, name='my_qwen3_pretrain')"

        Python API usage:
            >>> recipe = pretrain_recipe(name="qwen3_pretrain", num_nodes=1)
            >>> print(recipe)

    Note:
        This recipe uses a mock dataset, look for the finetune examples to see how to change the dataset.
    """
    recipe = run.Partial(
        fn,
        model=qwen3_model(),
        trainer=qwen3_trainer(
            tensor_parallelism=tensor_parallelism,
            pipeline_parallelism=pipeline_parallelism,
            pipeline_parallelism_type=pipeline_parallelism_type,
            virtual_pipeline_parallelism=virtual_pipeline_parallelism,
            context_parallelism=context_parallelism,
            sequence_parallelism=sequence_parallelism,
            expert_parallelism=expert_parallelism,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            max_steps=max_steps,
            precision=precision,
            accumulate_grad_batches=accumulate_grad_batches,
            limit_test_batches=limit_test_batches,
            limit_val_batches=limit_val_batches,
            log_every_n_steps=log_every_n_steps,
            val_check_interval=val_check_interval,
            callbacks=[run.Config(TimingCallback)],
        ),
        data=run.Config(
            MockDataModule,
            seq_length=seq_length,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            tokenizer=run.Config(AutoTokenizer, "Qwen/Qwen3-30B-A3B"),
        ),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(
            precision=precision,
            warmup_steps=warmup_steps,
            constant_steps=constant_steps,
            min_lr=min_lr,
            max_lr=max_lr,
            clip_grad=gradient_clip_val,
        ),
        resume=default_resume(),
    )
    recipe.model.config.recompute_granularity = "full"
    recipe.model.config.recompute_method = "uniform"
    recipe.model.config.recompute_num_layers = 1
    return recipe