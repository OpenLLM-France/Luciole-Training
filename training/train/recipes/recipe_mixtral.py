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

    from nemo.collections.llm.gpt.model.mixtral import MixtralConfig8x7B as config

    strategy_args["tensor_model_parallel_size"] = 1
    strategy_args["context_parallel_size"] = 2
    strategy_args["virtual_pipeline_model_parallel_size"] = 8
    strategy_args["pipeline_model_parallel_size"] = 4
    strategy_args["expert_model_parallel_size"] = 8

    # strategy_args["tensor_model_parallel_size"] = 8
    # strategy_args["context_parallel_size"] = 2
    # strategy_args["virtual_pipeline_model_parallel_size"] = 8
    # strategy_args["pipeline_model_parallel_size"] = 4
    # strategy_args["expert_model_parallel_size"] = 1
    # strategy_args["sequence_parallel"] = True

    return config(), strategy_args
