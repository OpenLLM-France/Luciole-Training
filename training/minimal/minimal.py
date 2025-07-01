import torch
import os
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.data.mock import MockDataModule
from megatron.core.distributed import DistributedDataParallelConfig
from nemo.lightning.pytorch.plugins.mixed_precision import MegatronMixedPrecision
from nemo.lightning.pytorch.optim import (
    MegatronOptimizerModule,
)
from megatron.core.optimizer import OptimizerConfig


def bf16_mixed():
    """
    BF16 mixed precision configuration for Megatron models.
    """
    return MegatronMixedPrecision(
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=True,
    )


def bf16_with_fp8_mixed():
    """BROKEN"""
    cfg = MegatronMixedPrecision(
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        fp8="hybrid",
        fp8_recipe="delayed",
        fp8_margin=0,
        fp8_amax_history_len=1024,
        fp8_amax_compute_algo="max",
        fp8_param_gather=True,
        grad_reduce_in_fp32=False,  # NVIDIA recommends False for FP8
    )
    return cfg


if __name__ == "__main__":
    data = MockDataModule(seq_length=4096, global_batch_size=512, micro_batch_size=1)

    llama_strategy_args = dict(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        context_parallel_size=1,
        sequence_parallel=False,
        gradient_as_bucket_view=True,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=False,
            average_in_collective=True,
        ),
    )

    strategy = nl.MegatronStrategy(**llama_strategy_args)

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=4,
        max_steps=1,
        num_nodes=1,
        plugins=bf16_with_fp8_mixed(),
        strategy=strategy,
    )

    from nemo.collections.llm.gpt.model.llama import Llama32Config1B

    model = llm.LlamaModel(Llama32Config1B(), tokenizer=data.tokenizer)

    nemo_logger = nl.NeMoLogger(
        name="test_fp8", log_dir=os.path.join(os.getenv("SCRATCH"), "test_fp8")
    )

    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer="data",
        optim=MegatronOptimizerModule(
            config=OptimizerConfig(optimizer="adam", bf16=True, fp16=False, lr=0.1)
        ),
        resume=None,
    )
