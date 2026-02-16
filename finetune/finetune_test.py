import argparse
import logging
import os
import fiddle
import torch
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs
from nemo import lightning as nl
import nemo_run as run
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.utils.exp_manager import TimingCallback
import nemo
from packaging.version import Version
from utils import serialize_fdl, deep_debug

nemo_version = Version(nemo.__version__)


def get_recipe(arch):
    if arch == "nemotron1b":
        from nemo.collections.llm.recipes.nemotron3_4b import (
            finetune_recipe as finetune_base_recipe,
        )

        def finetune_recipe(**kwargs):
            recipe = finetune_base_recipe(**kwargs)
            recipe.model.config.num_layers = 24
            recipe.model.config.num_attention_heads = 32
            recipe.model.config.num_query_groups = 8
            recipe.model.config.hidden_size = 2048
            recipe.model.config.ffn_hidden_size = 8192
            recipe.model.config.kv_channels = None
            recipe.model.config.share_embeddings_and_output_weights = True
            # Parallelism
            recipe.trainer.strategy.context_parallel_size = 2
            recipe.trainer.strategy.tensor_model_parallel_size = 1
            return recipe

        return finetune_recipe
    elif arch == "nemotronh8b":
        from nemo.collections.llm.recipes.nemotronh_8b import (
            finetune_recipe as finetune_base_recipe,
        )

        def finetune_recipe(**kwargs):
            recipe = finetune_base_recipe(**kwargs)
            # Parallelism
            recipe.trainer.strategy.context_parallel_size = 2
            recipe.trainer.strategy.tensor_model_parallel_size = 1
            return recipe

        return finetune_recipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="nemotron1b")
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.environ.get("OpenLLM_OUTPUT", ""), "finetune/olivier"),
    )
    parser.add_argument("--name", default="nemo_test", type=str)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--num_gpus_per_node", default=4, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--seq_length", default=1024, type=int)
    parser.add_argument(
        "--tokenizer_name",
        default="OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct",
        type=str,
    )
    parser.add_argument("--chat", default=False, action="store_true")
    parser.add_argument("--verbose", default=False, action="store_true")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    torch.set_float32_matmul_precision("high")

    if args.base_model == "nemotron1b":
        finetune_recipe = get_recipe("nemotron1b")
        resume_path = os.path.join(os.environ.get("OpenLLM_OUTPUT", ""), "pretrain/luciole_serie/luciole_nemotron1b_phase2/luciole_nemotron1b_phase2/checkpoints/luciole_nemotron1b_phase2-step=0382455")
    elif args.base_model == "nemotronh8b":
        finetune_recipe = get_recipe("nemotronh8b")
        resume_path = os.path.join(os.environ.get("OpenLLM_OUTPUT", ""), "pretrain/luciole_serie/luciole_nemotronh8b_phase2/luciole_nemotronh8b_phase2/checkpoints/luciole_nemotronh8b_phase2-step=0358929-last")
    else:
        raise ValueError(f"Unknown base model: {args.base_model}")

    recipe = finetune_recipe(
        dir=args.output_dir,
        name=args.name,
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.num_gpus_per_node,
        peft_scheme="none",
    )

    max_steps = 25

    # DATA
    tokenizer = get_tokenizer(tokenizer_name=args.tokenizer_name, use_fast=True)
    if args.chat:
        from nemo_patch.data.fine_tuning import FineTuningDataModule

        data_path = os.path.join(os.environ.get("OpenLLM_OUTPUT", ""), "data/instruct_data/sft_mix_test3")
        recipe.data = run.Config(
            FineTuningDataModule,
            dataset_root=data_path,
            global_batch_size=args.batch_size,
            micro_batch_size=1,
            num_workers=0,
            pin_memory=True,
            seq_length=args.seq_length,
            tokenizer=tokenizer,
            packed_sequence_specs=PackedSequenceSpecs(
                packed_sequence_size=args.seq_length,
                tokenizer_model_name=args.tokenizer_name.split("/")[-1],
            ),
            dataset_kwargs={"chat": True, "use_hf_tokenizer_chat_template": True},
        )

    else:
        from nemo_patch.data.fine_tuning import FineTuningDataModule

        data_path = "databricks"
        recipe.data = run.Config(
            FineTuningDataModule,
            dataset_root=data_path,
            global_batch_size=args.batch_size,
            micro_batch_size=1,
            num_workers=0,
            pin_memory=True,
            seq_length=args.seq_length,
            tokenizer=tokenizer,
            packed_sequence_specs=PackedSequenceSpecs(
                packed_sequence_size=args.seq_length,
                tokenizer_model_name=args.tokenizer_name.split("/")[-1],
            ),
            force=True,
        )

    if nemo_version == Version("2.3.1"):
        recipe.model.tokenizer = recipe.data.tokenizer
    else:
        recipe.tokenizer = "data"
    recipe.model.config.seq_length = recipe.data.seq_length
    recipe.trainer.max_steps = max_steps
    recipe.trainer.val_check_interval = max_steps
    recipe.trainer.limit_val_batches = 0.0

    restore_config = run.Config(
        nl.RestoreConfig, path=resume_path, load_optim_state=False
    )
    recipe.resume = run.Config(
        nl.AutoResume,
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        # resume_past_end=True,  # set to True if you want to continue training even if max_steps was reached
        restore_config=restore_config,
    )

    recipe.trainer.callbacks = [
        run.Config(TimingCallback),
        run.Config(
            ModelCheckpoint,
            every_n_train_steps=10,
            dirpath=args.output_dir,
            save_top_k=-1,
            always_save_context=True,
            save_optim_on_train_end=True,
            save_context_on_train_end=True,
        ),
    ]

    if args.verbose:
        # Print config
        recipe_dict = {
            "data": serialize_fdl(recipe.data),
            "trainer": serialize_fdl(recipe.trainer),
            "model": serialize_fdl(recipe.model),
            "optim": serialize_fdl(recipe.optim),
            "resume": serialize_fdl(recipe.resume),
            "log": serialize_fdl(recipe.log),
        }
        print(recipe_dict)

    # Finetune
    recipe_fn = fiddle.build(recipe)
    if args.verbose:
        deep_debug(recipe_fn, "recipe_fn")

    recipe_fn()
    logger.info("Finished training.")
