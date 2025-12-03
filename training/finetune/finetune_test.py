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
            recipe.trainer.strategy.context_parallel_size = 1
            recipe.trainer.strategy.tensor_model_parallel_size = 2
            return recipe

        return finetune_recipe
    elif arch == "nemotronh8b":
        from nemo.collections.llm.recipes.nemotronh_8b import (
            finetune_recipe as finetune_base_recipe,
        )

        def finetune_recipe(**kwargs):
            recipe = finetune_base_recipe(**kwargs)
            # Parallelism
            recipe.trainer.strategy.context_parallel_size = 1
            recipe.trainer.strategy.tensor_model_parallel_size = 4
            return recipe

        return finetune_recipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--resume_path", type=str, default="/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/pretrain/luciole_serie/luciole_nemotron1b_phase2/luciole_nemotron1b_phase2/checkpoints/luciole_nemotron1b_phase2-step=0382455")
    parser.add_argument(
        "--resume_path",
        type=str,
        default="/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/pretrain/luciole_serie/luciole_nemotronh8b_phase2/luciole_nemotronh8b_phase2/checkpoints/luciole_nemotronh8b_phase2-step=0358929-last",
    )
    # parser.add_argument("--data_path", type=str, default="databricks") # /Datasets/Train-Math-en-fr-NEMO-2
    parser.add_argument(
        "--data_path",
        type=str,
        default="/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/instruct_data/sft_mix",
    )  # /Datasets/Train-Math-en-fr-NEMO-2
    parser.add_argument(
        "--output_dir",
        default=f"{os.environ['qgz_ALL_CCFRSCRATCH']}/OpenLLM-BPI-output/finetune/olivier",
    )
    parser.add_argument("--name", default="nemo_test", type=str)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--num_gpus_per_node", default=4, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--seq_length", default=1024, type=int)
    parser.add_argument(
        "--tokenizer_name",
        default="OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct",
        type=str,
    )
    parser.add_argument("--chat", default=False, action="store_true")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    torch.set_float32_matmul_precision("high")

    finetune_recipe = get_recipe("nemotronh8b")
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
        from nemo.collections.llm.gpt.data import FineTuningDataModule

        recipe.data = FineTuningDataModule(
            dataset_root=args.data_path,
            global_batch_size=args.batch_size,
            micro_batch_size=1,
            num_workers=0,
            pin_memory=True,
            seq_length=args.seq_length,
            tokenizer=tokenizer,
            # packed_sequence_specs=PackedSequenceSpecs(
            #     packed_sequence_size=args.seq_length,
            #     tokenizer_model_name=args.tokenizer_name.split("/")[-1],
            # ),
            dataset_kwargs={"chat": True, "use_hf_tokenizer_chat_template": True},
        )
    else:
        from nemo.collections.llm.gpt.data import FineTuningDataModule

        recipe.data = FineTuningDataModule(
            dataset_root=args.data_path,
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
        )
    recipe.tokenizer = "data"
    recipe.model.config.seq_length = recipe.data.seq_length
    recipe.trainer.max_steps = max_steps
    recipe.trainer.val_check_interval = 10
    recipe.trainer.limit_val_batches = 0.0

    restore_config = nl.RestoreConfig(path=args.resume_path, load_optim_state=False)
    recipe.resume = run.Config(
        nl.AutoResume,
        restore_config=restore_config,
    )
    recipe.trainer.callbacks = [
        run.Config(TimingCallback),
        run.Config(
            ModelCheckpoint,
            every_n_train_steps=10,
            dirpath=dir,
            save_top_k=-1,
            always_save_context=True,
            save_optim_on_train_end=True,
            save_context_on_train_end=True,
        ),
    ]

    # Finetune
    recipe_obj = fiddle.build(recipe)
    recipe_obj()
    logger.info("Finished training.")
