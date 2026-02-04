import argparse
import logging
import os
from nemo.collections.llm.recipes.nemotronh_8b import (
    finetune_recipe as finetune_base_recipe,
)
import fiddle
import torch
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
#from nemo.collections.llm.gpt.data.core import GPTSFTChatDataset
from nemo.collections.llm.gpt.data import FineTuningDataModule
from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs
import json

def finetune_recipe(**kwargs):
    recipe = finetune_base_recipe(**kwargs)
    # Parallelism
    recipe.trainer.strategy.context_parallel_size = 1
    recipe.trainer.strategy.tensor_model_parallel_size = 1
    return recipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_path", type=str, default="/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/pretrain/luciole_serie/luciole_nemotronh8b_phase2/luciole_nemotronh8b_phase2/checkpoints/luciole_nemotronh8b_phase2-step=0358929-last")    
    parser.add_argument("--data_path", type=str, default=f"{os.environ['SCRATCH']}/sft_mix")
    parser.add_argument(
        "--output_dir",
        default=f"{os.environ['qgz_ALL_CCFRSCRATCH']}/OpenLLM-BPI-output/finetune",
    )
    parser.add_argument("--name", default="sftmix_test_8b", type=str)
    parser.add_argument("--num_nodes", default=4, type=int)
    parser.add_argument("--num_gpus_per_node", default=4, type=int)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--packed_sequence", default=False, action="store_true")
    parser.add_argument("--batch_size", default=32, type=int) # 1024
    parser.add_argument("--seq_length", default=4096, type=int)
    parser.add_argument(
        "--tokenizer_name",
        default="/lustre/fsn1/projects/rech/qgz/ufb61ek/luciole_tokenizer_instruct_new/",
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
        peft_scheme="none",
    )
    recipe.resume.restore_config.path = args.resume_path
    if args.debug:
        recipe.trainer.max_steps = 25
    else:
        recipe.trainer.max_steps = 2000
        #recipe.trainer.max_steps = 25000

    # DATA
    #chat_tmp = json.load(open("luciole_tokenizer_instruct/tokenizer_config.json"))["chat_template"]
    tokenizer = get_tokenizer(tokenizer_name=args.tokenizer_name, use_fast=True)
    recipe.data = FineTuningDataModule(
        dataset_root=args.data_path,
        global_batch_size=args.batch_size,
        micro_batch_size=1,
        num_workers=8, #changed from 8 to 0
        pin_memory=True,
        seq_length=args.seq_length,
        tokenizer=tokenizer,
        #packed_sequence_specs=PackedSequenceSpecs(
        #    packed_sequence_size=args.seq_length,
        #    tokenizer_model_name=args.tokenizer_name.split("/")[-2],
        #),
        dataset_kwargs={"chat": True,"use_hf_tokenizer_chat_template": True},
    )
    recipe.tokenizer = "data"
    #recipe.model.tokenizer = recipe.data.tokenizer
    recipe.trainer.val_check_interval = 1000
    #recipe.data._create_dataset(path=args.data_path+"/training.jsonl",kwargs={"chat": True,"use_hf_tokenizer_chat_template": True},)
    #recipe.data._create_dataset(path=args.data_path+"/training.jsonl",chat=True,use_hf_tokenizer_chat_template=True)
    #print("="*50)
    #print("chat template:",recipe.data.tokenizer.tokenizer.chat_template)
    #print("Type of tokenizer:",type(recipe.data.tokenizer))
    #print("Packed Sequence Size:", recipe.data.packed_sequence_size)
    #print("Train path packed:", recipe.data.train_path_packed)
    #print("Default pack path:", recipe.data.default_pack_path)
    #print("Dataset root:", recipe.data.dataset_root)
    #print("Tokenizer model name:", recipe.data._extract_tokenizer_model_name())
    #print("="*50)
    # Finetune
    recipe_obj = fiddle.build(recipe)
    recipe_obj()
    logger.info("Finished training.")
