import nemo_run as run
import torch
import os
import logging
import datetime
from lightning.pytorch.callbacks.timer import Timer
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed, bf16_with_fp8_current_scaling_mixed, bf16_with_mxfp8_mixed
from nemo.utils.exp_manager import TimingCallback
from .callbacks import PytorchProfilerCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_time_limit(time_limit, buffer_minutes: int = 30) -> str:
    logging.info(time_limit)
    h, m, s = map(int, time_limit.split(":"))
    slurm_time_limit = datetime.timedelta(hours=h, minutes=m, seconds=s)
    td = slurm_time_limit - datetime.timedelta(minutes=buffer_minutes)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    return f"{td.days:02}:{hours:02}:{minutes:02}:00"
    
class StatelessTimer(Timer):
    """Extension of PTL timers to be per run."""

    # Override PTL Timer's state dict to not store elapsed time information so that we can
    # restore and continue training.
    def state_dict(self):
        """state_dict"""
        return {}

    def load_state_dict(self, state_dict) -> None:
        """load_state_dict"""
        return


def set_recipe_trainer(recipe, args):
    recipe.trainer.limit_val_batches = 0.0
    recipe.trainer.log_every_n_steps = (
        1 if args.mode in ["debug", "benchmark", "benchmark100"] else 5
    )
    time_limit = get_time_limit(args.max_time_per_run, 5 if args.mode in ["debug", "benchmark", "benchmark100"] else 30)
    # os.makedirs(f"{args.output_dir}/traces", exist_ok=True)
    recipe.trainer.callbacks = [
        run.Config(TimingCallback), 
        run.Config(StatelessTimer, duration=time_limit), 
        # run.Config(PytorchProfilerCallback, start_step=15, end_step=20, warmup_steps=1, active_steps=5, trace_dir=f"{args.output_dir}/traces")
    ]
    if args.fp8:
        if args.arch == "nemotronh47b":
            logger.info("FP8 is always activated on nemotronh47b")
        else:
            fp8_plugin = bf16_with_fp8_mixed()
            # fp8_plugin.first_last_layers_bf16 = True
            # fp8_plugin.num_layers_at_start_in_bf16 = 1
            # fp8_plugin.num_layers_at_end_in_bf16 = 1
            recipe.trainer.plugins = fp8_plugin
            recipe.trainer.plugins.grad_reduce_in_fp32 = False
            recipe.trainer.strategy.ddp.grad_reduce_in_fp32 = False

    # STRATEGY
    if args.tensor_parallelism:
        recipe.trainer.strategy.tensor_model_parallel_size = args.tensor_parallelism
    if args.pipeline_parallelism:
        recipe.trainer.strategy.pipeline_model_parallel_size = args.pipeline_parallelism
    recipe.trainer.strategy.pipeline_dtype = torch.bfloat16
    if args.virtual_pipeline_parallelism:
        if args.virtual_pipeline_parallelism == -1:
            recipe.trainer.strategy.virtual_pipeline_model_parallel_size = None
        else:
            recipe.trainer.strategy.virtual_pipeline_model_parallel_size = (
                args.virtual_pipeline_parallelism
            )
    if args.context_parallelism:
        recipe.trainer.strategy.context_parallel_size = args.context_parallelism
    if recipe.trainer.strategy.tensor_model_parallel_size == 1:
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
    recipe.trainer.strategy.ckpt_async_save = True
    return recipe
