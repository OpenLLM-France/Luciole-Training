import logging

from megatron.core.distributed import DistributedDataParallelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_config(strategy_args=None, args=None, arch=None, tokenizer_name=None):
    pretrain_strategy_args = dict(
        sequence_parallel=False,
        ckpt_load_optimizer=True,
        ckpt_save_optimizer=True,
        gradient_as_bucket_view=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=False,
            average_in_collective=True,
        ),
    )
    strategy_args = dict(**strategy_args, **pretrain_strategy_args)
    if arch == "llama1b":
        from nemo.collections.llm.gpt.model.llama import (
            Llama32Config1B as LlamaConfig,
        )

        pretrain_strategy_args["optimizer_warmup_steps"] = 500
    elif arch == "llama3b":
        from nemo.collections.llm.gpt.model.llama import (
            Llama32Config3B as LlamaConfig,
        )
    elif arch == "llama8b":
        from nemo.collections.llm.gpt.model.llama import (
            Llama31Config8B as LlamaConfig,
        )
    elif arch == "llama70b":
        from nemo.collections.llm.gpt.model.llama import (
            Llama31Config70B as LlamaConfig,
        )

        strategy_args["sequence_parallel"] = True
        strategy_args["tensor_model_parallel_size"] = 4
        strategy_args["pipeline_model_parallel_size"] = 4
        strategy_args["virtual_pipeline_model_parallel_size"] = 5
        strategy_args["context_parallel_size"] = 2
    else:
        raise ValueError(f"Unsupported llama model : {arch}")

    return LlamaConfig(), strategy_args
