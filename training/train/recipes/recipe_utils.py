import nemo_run as run
import torch
import logging
import warnings
from typing import Optional

from nemo.lightning.pytorch.optim.lr_scheduler import WarmupHoldPolicyScheduler
from nemo.core.optim.lr_scheduler import WarmupAnnealHoldPolicy
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
from nemo.utils.exp_manager import TimingCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_recipe_trainer(recipe, args):
    recipe.trainer.limit_val_batches = 0.0
    recipe.trainer.log_every_n_steps = (
        1 if args.mode in ["debug", "benchmark", "benchmark100"] else 1
    )
    recipe.trainer.callbacks = [run.Config(TimingCallback)]
    if args.fp8:
        if args.arch == "nemotronh47b":
            logger.info("FP8 is always activated on nemotronh47b")
        else:
            recipe.trainer.plugins = bf16_with_fp8_mixed()
            recipe.trainer.plugins.grad_reduce_in_fp32 = False
            recipe.trainer.strategy.ddp.grad_reduce_in_fp32 = False

    # STRATEGY
    if args.tensor_parallelism:
        recipe.trainer.strategy.tensor_model_parallel_size = args.tensor_parallelism
    if args.pipeline_parallelism:
        recipe.trainer.strategy.pipeline_model_parallel_size = args.pipeline_parallelism
    recipe.trainer.strategy.pipeline_dtype = torch.bfloat16
    if args.virtual_pipeline_parallelism:
        if args.virtual_pipeline_parallelism==-1:
            recipe.trainer.strategy.virtual_pipeline_model_parallel_size = None
        else:
            recipe.trainer.strategy.virtual_pipeline_model_parallel_size = (
                args.virtual_pipeline_parallelism
            )
    if args.context_parallelism:
        recipe.trainer.strategy.context_parallel_size = args.context_parallelism
    if recipe.trainer.strategy.tensor_model_parallel_size==1:
        recipe.trainer.strategy.sequence_parallel = False
    # if args.sequence_parallelism is not None:
    #     recipe.trainer.strategy.sequence_parallel = args.sequence_parallelism
    num_gpus = recipe.trainer.devices * recipe.trainer.num_nodes
    if recipe.data.micro_batch_size > 1 and recipe.data.global_batch_size >= num_gpus:
        logger.warning(
            f"Micro batch size is set to {recipe.data.micro_batch_size} which is greater than 1 and global batch size is greater than number of GPUs. This is not supported for Megat. Setting micro batch size to 1."
        )
        recipe.data.micro_batch_size = 1

    if (
        recipe.trainer.strategy.tensor_model_parallel_size > 4
        and args.tensor_parallelism is None
    ):
        logger.warning(
            f"Tensor parallelism is set to {recipe.trainer.strategy.tensor_model_parallel_size} which is greater than 4. We only have 4 GPUs per node. Setting tensor parallelism to 4."
        )
        # recipe.trainer.strategy.tensor_model_parallel_size = 4
        # # if recipe.data.micro_batch_size > 1:
        # #     recipe.data.micro_batch_size = recipe.data.micro_batch_size // 2
        # # else:
        # recipe.trainer.strategy.pipeline_model_parallel_size = (
        #     recipe.trainer.strategy.pipeline_model_parallel_size * 2
        # )
    return recipe

def set_custom_scheduler(recipe, hold_steps=5, constant_lr=False):
    if constant_lr:
        min_lr = recipe.optim.config.lr
        warmup_steps = 0
        hold_steps = 0
    else:
        min_lr = 1e-5
        warmup_steps = 5
        # min_lr = recipe.optim.lr_scheduler.min_lr
        # warmup_steps = recipe.optim.lr_scheduler.warmup_steps
    recipe.optim.lr_scheduler = run.Config(
        WarmupHoldAnnealingLinearScheduler,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
        hold_steps=hold_steps,
    )
    return recipe

class WarmupHoldAnnealingLinear(WarmupAnnealHoldPolicy):
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )

        step = self.last_epoch

        # Reset learning rate
        if 'reset_lr' in self.optimizer.param_groups[0].keys():
            reset_lr = self.optimizer.param_groups[0]['reset_lr']
            num_steps = reset_lr['num_steps']
            step -= num_steps
            if reset_lr['if_init_step'] and reset_lr['reset_lr_steps']:
                self.decay_steps -= num_steps
                self.max_steps -= num_steps
                self.optimizer.param_groups[0]['reset_lr']['if_init_step'] = False

        # Warmup steps
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self._get_warmup_lr(step)

        if self.constant_steps > 0 and (self.warmup_steps) < step < (self.warmup_steps + self.constant_steps):
            return self.base_lrs

        # Min lr after max steps of updates
        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)
    
    def _get_lr(self, step):
        delta_lr = self.base_lrs[0] - self.min_lr
        mult = (step+1 - (self.warmup_steps+self.constant_steps)) / (self.max_steps - (self.warmup_steps+self.constant_steps))
        out_lr = [self.min_lr + (1 - mult) * delta_lr for _ in self.base_lrs]
        return out_lr

class WarmupHoldAnnealingLinearScheduler(WarmupHoldPolicyScheduler):
    """Warmup Annealing Learning Rate Scheduler."""


    def scheduler(self, model, optimizer):
        lr_scheduler = WarmupHoldAnnealingLinear(
            optimizer,
            warmup_steps=self.warmup_steps,
            warmup_ratio=self.warmup_ratio,
            constant_steps=self.hold_steps,
            constant_ratio=self.hold_ratio,
            max_steps=self.max_steps,
            min_lr=self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": self.interval,
                "frequency": self.frequency,
            },
            "monitor": self.monitor,
        }
