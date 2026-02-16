import argparse
import logging
import os
from nemo.collections.llm.recipes.qwen25_7b import (
    finetune_recipe as finetune_base_recipe,
)
import fiddle
import torch
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.llm.gpt.data import FineTuningDataModule
from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs


def finetune_recipe(**kwargs):
    recipe = finetune_base_recipe(**kwargs)
    recipe.model.config.num_layers = 28
    recipe.model.config.num_attention_heads = 28
    recipe.model.config.num_query_groups = 4
    recipe.model.config.hidden_size = 3584
    recipe.model.config.ffn_hidden_size = 18944
    # recipe.model.config.kv_channels = None
    # recipe.model.config.share_embeddings_and_output_weights = True
    # Parallelism
    recipe.trainer.strategy.context_parallel_size = 1
    recipe.trainer.strategy.tensor_model_parallel_size = 4
    recipe.trainer.strategy.sequence_parallel = True
    recipe.optim.config.lr = 5e-5
    recipe.optim.lr_scheduler.min_lr = 5e-6
    recipe.optim.config.adam_beta1 = 0.9
    recipe.optim.config.adam_beta2 = 0.999
    recipe.optim.config.adam_eps = 1e-8
    recipe.optim.config.weight_decay = 0.0
    recipe.optim.lr_scheduler.warmup_steps = 25
    return recipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_path", type=str, default=os.path.join(os.path.expanduser("~"), ".cache/nemo/models/Qwen2.5-7B-Instruct"))
    parser.add_argument("--data_path", type=str, default=f"{os.environ['SCRATCH']}/Datasets/Train-Math-en-fr-NEMO-2")
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.environ.get("SCRATCH", "/tmp"), "Models/Qwen2.5-7B-Instruct-en-fr-2"),
    )
    parser.add_argument("--name", default="nemo_test", type=str)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--num_gpus_per_node", default=4, type=int)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--packed_sequence", default=True, action="store_true")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--seq_length", default=18000, type=int)
    parser.add_argument(
        "--tokenizer_name",
        default=f"{os.environ['SCRATCH']}/Models/Qwen2.5-7B-Instruct",
        type=str,
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    torch.set_float32_matmul_precision("high")

    recipe = finetune_recipe(
        dir=args.output_dir,
        name=args.name,
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus_per_node,
        packed_sequence=args.packed_sequence,
        peft_scheme="none",
    )

    recipe.resume.restore_config.path = args.resume_path


    recipe.trainer.max_steps = 434
    recipe.trainer.val_check_interval = 30
    recipe.log.log_dir = args.output_dir
    recipe.log.ckpt.every_n_train_steps = 50

    # DATA
    tokenizer = get_tokenizer(tokenizer_name=args.tokenizer_name, use_fast=True)
    recipe.data = FineTuningDataModule(
        dataset_root=args.data_path,
        global_batch_size=args.batch_size,
        micro_batch_size=1,
        num_workers=8,
        pin_memory=True,
        seq_length=args.seq_length,
        tokenizer=tokenizer,
        packed_sequence_specs=PackedSequenceSpecs(
            packed_sequence_size=args.seq_length,
            tokenizer_model_name=args.tokenizer_name,
        ),
    )

    # Finetune
    recipe_obj = fiddle.build(recipe)
    recipe_obj()
    logger.info("Finished training.")
