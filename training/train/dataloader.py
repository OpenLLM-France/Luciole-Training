import argparse
import os
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections import llm
import fiddle as fdl


def create_data(data_args: dict):
    tokenizer = get_tokenizer(
        tokenizer_name=data_args.pop("tokenizer_name"), use_fast=True
    )
    data = PreTrainingDataModule(
        micro_batch_size=1,
        num_workers=8,
        pin_memory=True,
        tokenizer=tokenizer,
        split="1,0,0",
        **data_args,
    )
    return data


def save_sample_texts(data, output, number_of_data):
    dataloader = data.train_dataloader()

    samples_written = 0
    with open(output + ".txt", "w", encoding="utf-8") as token_file:
        for batch in dataloader:
            tokens_batch = batch["tokens"]

            for token_ids in tokens_batch:
                token_ids = token_ids.tolist()  # Convert tensor to list
                tokens = data.tokenizer.ids_to_tokens(token_ids)
                # text = data.tokenizer.ids_to_text(token_ids, remove_special_tokens=False)

                token_file.write("\n\n>>>>>>>>>>>> NEW SAMPLE <<<<<<<<<<<<\n\n")
                token_file.write(repr(tokens))  # Debug-friendly format

                samples_written += 1
                if samples_written >= number_of_data:
                    return


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


def run_dataloader(paths, tokenizer_name, output, number_of_data, seq_length):
    recipe = configure_recipe(nodes=1, gpus_per_node=1)
    recipe.data = create_data(
        {
            "paths": paths,
            "tokenizer_name": tokenizer_name,
            "global_batch_size": 1,
            "seq_length": seq_length,
        }
    )
    recipe.data.build(5, 1, 1, 1)
    recipe.data.trainer = fdl.build(recipe.trainer)
    save_sample_texts(
        recipe.data,
        output=output,
        number_of_data=int(number_of_data),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "folder_path",
        help="",
        type=str,
    )
    parser.add_argument(
        "--number_of_data", help="Number of iteration", default=10, type=str
    )
    parser.add_argument("--seq_length", help="", default=4096, type=int)
    args = parser.parse_args()

    with open(os.path.join(args.folder_path, "tokenizer_name.txt"), "r") as f:
        tokenizer_name = f.read().strip()

    # Ensure the output directory exists
    os.makedirs(
        os.path.join(args.folder_path, f"batch_examples_seq{args.seq_length}"),
        exist_ok=True,
    )

    for file in os.listdir(args.folder_path):
        dataset_name = file.split(".")[0]

        if file.endswith("text_document.idx"):
            data_path = os.path.join(args.folder_path, dataset_name)
            output_path = os.path.join(
                args.folder_path, f"batch_examples_seq{args.seq_length}", dataset_name
            )

            if not os.path.exists(output_path):
                print(f"Processing dataset: {dataset_name}...")
                run_dataloader(
                    data_path,
                    tokenizer_name,
                    output_path,
                    args.number_of_data,
                    args.seq_length,
                )
