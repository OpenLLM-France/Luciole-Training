import torch
import logging
from typing import Optional

from nemo import lightning as nl
from nemo.lightning.pytorch.plugins.mixed_precision import MegatronMixedPrecision

# from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed, bf16_mixed
from nemo.lightning.pytorch.optim import (
    CosineAnnealingScheduler,
    MegatronOptimizerModule,
)
from megatron.core.optimizer import OptimizerConfig
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def bf16_with_fp8_mixed():
    """FP8 recipes are experimental and have not been tested for training convergence."""
    cfg = MegatronMixedPrecision(
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        fp8="hybrid",
        fp8_margin=0,
        fp8_amax_history_len=1024,
        fp8_amax_compute_algo="max",
        fp8_params=True,
        grad_reduce_in_fp32=False,  # NVIDIA recommends False for FP8
    )
    logger.info(f"bf16_with_fp8_mixed:\n{vars(cfg)}")
    return cfg


def create_trainer(
    strategy_args: dict = None,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    val_check_interval: int = 1000,
    limit_val_batches: int = 0,
    callbacks: Optional[list[Callback]] = None,
    fp8: bool = False,
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

    strategy = nl.MegatronStrategy(**strategy_args)

    trainer = nl.Trainer(
        accelerator="gpu",
        accumulate_grad_batches=1,
        callbacks=callbacks,
        devices=num_gpus_per_node,
        limit_test_batches=50,
        log_every_n_steps=10,
        max_steps=max_steps,
        num_nodes=num_nodes,
        plugins=bf16_with_fp8_mixed() if fp8 else bf16_mixed(),
        strategy=strategy,
        use_distributed_sampler=False,
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=2,  # not sure it works
    )

    return trainer
