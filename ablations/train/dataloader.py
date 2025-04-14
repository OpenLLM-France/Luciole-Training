import argparse
import os
from nemo.lightning.data import WrappedDataLoader
from torch.utils.data._utils.collate import default_collate
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections import llm
import fiddle as fdl

def custom_collate_with_positional_offset(batch, offset=100, eos_token_id=1):
    batch = default_collate(batch)  # collate normally
    if 'position_ids' in batch:  
        batch['position_ids'] += offset*(batch['tokens'] == eos_token_id).cumsum(dim=-1)
    return batch

class WrappedPreTrainingDataModule(PreTrainingDataModule):

    def __init__(self, offset_collate=False, **dataloader_kwargs):
        super().__init__(**dataloader_kwargs)
        self.offset_collate = offset_collate

    def _create_dataloader(self, dataset, mode, **kwargs) -> WrappedDataLoader:
        self.init_global_step = self.trainer.global_step
        self.data_sampler.init_global_step = self.init_global_step

        if self.offset_collate:
            collate_fn=getattr(dataset, "collate_fn", custom_collate_with_positional_offset)
        else:
            collate_fn=getattr(dataset, "collate_fn", default_collate)
        
        dataloader = WrappedDataLoader(
            mode=mode,
            dataset=dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
            **kwargs,
        )
        return dataloader

def create_data(
    data_path, tokenizer_name="OpenLLM-France/Lucie-7B", batch_size=512, seq_length=2048
):
    tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, use_fast=True)
    data = WrappedPreTrainingDataModule(
        offset_collate=False,
        paths=data_path,
        global_batch_size=batch_size,
        micro_batch_size=1,
        num_workers=8,
        pin_memory=True,
        seq_length=seq_length,  # 8192 for llama 32 1b
        tokenizer=tokenizer,
        split="1,0,0",
    )
    return data

def save_sample_texts(data, output=None, number_of_data=5):
    dataloader = data.train_dataloader()
    
    for i, batch in enumerate(dataloader):
        print("\n" + f" START TEXT {i} ".center(80, "-"))
        # Extract and decode token IDs
        token_ids = batch["tokens"][0]
        text = data.tokenizer.ids_to_text(token_ids, remove_special_tokens=False)
        print(text)
        print(f" END TEXT {i} ".center(80, "-") + "\n")

        print("\n" + f" START BATCH {i} ".center(80, "-"))
        print(batch)
        print(f" END BATCH {i} ".center(80, "-") + "\n")
        # Save to file if output directory is specified
        if output:
            os.makedirs(output, exist_ok=True)
            file_path = os.path.join(output, f"{i}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

        # Stop after desired number of samples
        if i + 1 >= number_of_data:
            break


def configure_recipe(nodes: int = 1, gpus_per_node: int = 1):
    recipe = llm.llama3_8b.pretrain_recipe(
        name="iterating_dataloader",
        dir="iterating_dataloader",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )
    recipe.model.config.num_layers = 2
    recipe.trainer.max_steps = 5
    return recipe

def run_dataloader(paths, output, number_of_data=1, seq_length=2048):
    recipe = configure_recipe(nodes=1, gpus_per_node=1)
    recipe.data = create_data(paths, batch_size=1, seq_length=seq_length)
    recipe.data.build(5, 1, 1, 1)
    recipe.data.trainer = fdl.build(recipe.trainer)
    save_sample_texts(
        recipe.data,
        output=output,
        number_of_data=int(number_of_data),
    )

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name",
        help="",
        default="wikipedia_fr_text_document",
        type=str,
    )
    parser.add_argument(
        "--number_of_data", help="Number of iteration", default=10, type=str
    )
    parser.add_argument("--seq_length", help="", default=4096, type=str)
    args = parser.parse_args()

    main_path = os.path.join(os.getenv("OpenLLM_OUTPUT"), "data/tokens_ablation")
    data_path = os.path.join(main_path, args.dataset_name)
    output_path = os.path.join(main_path, "batch_examples", args.dataset_name)

    run_dataloader(data_path, output_path, args.number_of_data, args.seq_length)
