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

    if arch == "mamba1b":
        from nemo.collections.llm.gpt.model.ssm import (
            BaseMambaConfig1_3B as MambaConfig,
        )

        model_config = MambaConfig(
            tokenizer_library="huggingface",
            tokenizer_name=tokenizer_name,
            share_embeddings_and_output_weights=True,
        )
    elif arch == "mambahybrid8b":
        from nemo.collections.llm.gpt.model.ssm import (
            NVIDIAMambaHybridConfig8B as MambaConfig,
        )

        model_config = MambaConfig(
            tokenizer_library="huggingface",
            tokenizer_name=tokenizer_name,
            hybrid_override_pattern="*-".join(["M-" * 5] * 5),
            num_layers=58,
        )
        strategy_args["tensor_model_parallel_size"] = (
            args.tensor_parallelism if args.tensor_parallelism else 4
        )

    return model_config, strategy_args
