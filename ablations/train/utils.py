import torch
import os
import logging
from typing import Optional

from nemo import lightning as nl
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.lightning.pytorch.plugins.mixed_precision import MegatronMixedPrecision
from nemo.lightning.pytorch.optim import (
    CosineAnnealingScheduler,
    MegatronOptimizerModule,
)
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def tensorboard_logger(name: str, save_dir: str = "tb_logs"):
    """Factory function to configure TensorBoard Logger."""
    return TensorBoardLogger(save_dir=save_dir, name=name)


def wandb_logger(project: str, name: str, entity: Optional[str] = None):
    """Factory function to configure W&B Logger."""
    return WandbLogger(project=project, name=name, config={})


def create_autoresume(resume_if_exists=True, resume_ignore_no_checkpoint=True):
    """Factory function to configure AutoResume."""
    return nl.AutoResume(
        resume_if_exists=resume_if_exists,
        resume_ignore_no_checkpoint=resume_ignore_no_checkpoint,
    )


def distributed_fused_adam_with_cosine_annealing(
    precision: str = "bf16-mixed",  # or "16-mixed"
    warmup_steps: int = 2000,
    constant_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.95,
    max_lr: float = 1e-4,
    min_lr: Optional[float] = None,
    clip_grad: float = 1.0,
):
    """
    Creates a distributed fused Adam optimizer with cosine annealing scheduler.
    """
    opt_cfg = OptimizerConfig(
        optimizer="adam",
        lr=max_lr,
        weight_decay=0.1,
        bf16=precision == "bf16-mixed",
        fp16=precision == "16-mixed",
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_eps=1e-5,
        use_distributed_optimizer=True,
        clip_grad=clip_grad,
    )

    min_lr = min_lr if min_lr is not None else (0.1 * max_lr)
    sched = CosineAnnealingScheduler(
        warmup_steps=warmup_steps,
        constant_steps=constant_steps,
        min_lr=min_lr,
    )

    return MegatronOptimizerModule(
        config=opt_cfg,
        lr_scheduler=sched,
    )


def create_logger(
    dir: Optional[str] = None,
    name: str = "default",
    every_n_train_steps=1000,
    tensorboard_logger: Optional[TensorBoardLogger] = None,
    wandb_logger: Optional[WandbLogger] = None,
):
    """Factory function to configure NemoLogger."""
    ckpt = nl.ModelCheckpoint(
        save_last=True,
        save_top_k=-1,
        every_n_train_steps=every_n_train_steps,
        monitor="step",
        mode="max",
        every_n_epochs=None,
        # filename="{step}--{consumed_samples:.0f}-{train_loss:.2f}",
    )

    return nl.NeMoLogger(
        ckpt=ckpt,
        name=name,
        tensorboard=tensorboard_logger,
        wandb=wandb_logger,
        log_dir=dir,
    )


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


def create_trainer(
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_type: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    callbacks: Optional[list[Callback]] = None,
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
        plugins=bf16_mixed(),
        strategy=strategy,
        use_distributed_sampler=False,
        val_check_interval=100,  # default 2000
        limit_val_batches=0.0,  # 32
        num_sanity_val_steps=2,  # dont work
    )

    return trainer


def create_data(
    data_path, tokenizer_name="OpenLLM-France/Lucie-7B", batch_size=512, seq_length=2048
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, use_fast=True)
    data = PreTrainingDataModule(
        paths=data_path,
        global_batch_size=batch_size,
        micro_batch_size=1,
        num_workers=8,
        pin_memory=True,
        seq_length=seq_length,  # 8192 for llama 32 1b
        tokenizer=tokenizer,
        split="90,5,5",
    )
    return data


def read_datamix_file(file):
    loaded_data = None
    if file.endswith(".json"):
        import json

        with open(file, "r") as f:
            loaded_data = json.load(f)
    elif file.endswith(".yaml"):
        import yaml

        with open(file, "r") as f:
            loaded_data = yaml.safe_load(f)
    else:
        raise RuntimeError(f"Config should be a json or a yaml, got {file}")

    def make_data_flattened_list(split="train"):
        data_paths = []
        for dataset in loaded_data.get(split, []):
            data_paths.append(str(dataset["weight"]))
            data_paths.append(os.path.join(loaded_data["data_path"], dataset["name"]))
        return data_paths

    data_paths = {
        "train": make_data_flattened_list("train"),
        "validation": make_data_flattened_list("valid"),
        "test": make_data_flattened_list("test"),
    }
    return data_paths
