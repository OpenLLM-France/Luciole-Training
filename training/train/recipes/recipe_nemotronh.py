import logging


from megatron.core.distributed import DistributedDataParallelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_config(strategy_args=None, args=None, arch=None, tokenizer_name=None):
    pretrain_strategy_args = dict(
        sequence_parallel=True
        if strategy_args["tensor_model_parallel_size"] > 1
        else False,
        ckpt_load_optimizer=True,
        ckpt_save_optimizer=True,
        ckpt_async_save=False,
        save_ckpt_format="torch_dist",
        ckpt_load_strictness="log_all",
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            overlap_grad_reduce=True,
            overlap_param_gather=False,  # Verify that this works
            grad_reduce_in_fp32=True,
        ),
    )
    strategy_args = dict(**strategy_args, **pretrain_strategy_args)

    if arch == "nemotronh8b":
        # strategy_args["tensor_model_parallel_size"] = 2 # 2
        from nemo.collections.llm.gpt.model.ssm import NemotronHConfig8B as config
    elif arch == "nemotronh56b":
        strategy_args["tensor_model_parallel_size"] = 8
        from nemo.collections.llm.gpt.model.ssm import NemotronHConfig56B as config

        if strategy_args["fp8"]:
            from .recipe_utils import nemotron_h_bf16_with_fp8_current_scaling_mixed

            logger.info(
                f"Using custom fp8 plugin: {nemotron_h_bf16_with_fp8_current_scaling_mixed}"
            )
            strategy_args[
                "precision_plugin"
            ] = nemotron_h_bf16_with_fp8_current_scaling_mixed()
    model_config = config(
        tokenizer_library="huggingface",
        tokenizer_name=tokenizer_name,
    )
    return model_config, strategy_args
