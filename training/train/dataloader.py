import argparse
import os
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections import llm
import fiddle as fdl
import json
import nemo_run as run

def create_data(data_args: dict):
    tokenizer = get_tokenizer(
        tokenizer_name=data_args.pop("tokenizer_name"), use_fast=True
    )
    data = run.Config(PreTrainingDataModule,
        num_workers=8,
        pin_memory=True,
        tokenizer=tokenizer,
        split="1,0,0",
        **data_args,
    )
    return data


def split_before_item(lst, item):
    result = []
    current = []
    for elem in lst:
        if elem == item and current:
            result.append(current)
            current = []
        current.append(elem)
    if current:
        result.append(current)
    return result


def save_sample_texts(data, number_of_texts, number_of_distributions):
    dataloader = data.train_dataloader()

    samples_for_text = 0
    samples_for_dist = 0
    distribution = []
    output_text = ""

    for batch in dataloader:
        tokens_batch = batch["tokens"]

        for token_ids in tokens_batch:
            token_ids = token_ids.tolist()
            tokens = data.tokenizer.ids_to_tokens(token_ids)

            # Collect distribution from up to number_of_distributions samples
            if samples_for_dist < number_of_distributions:
                distribution.extend(
                    [len(chunk) for chunk in split_before_item(tokens, "</s>")]
                )
                samples_for_dist += 1

            # Collect text from up to number_of_texts samples
            if samples_for_text < number_of_texts:
                output_text += "\n\n>>>>>>>>>>>> NEW SAMPLE <<<<<<<<<<<<\n\n"
                output_text += repr(tokens)
                samples_for_text += 1

            # Stop if both limits are reached
            if (
                samples_for_text >= number_of_texts
                and samples_for_dist >= number_of_distributions
            ):
                return output_text, distribution

    return output_text, distribution


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


def run_dataloader(
    folder_path, dataset_name, tokenizer_name, number_of_data, seq_length, force=False
):
    token_file_path = os.path.join(
        folder_path, f"batch_examples_seq{seq_length}", f"{dataset_name}.txt"
    )
    dist_file_dir = os.path.join(folder_path, f"batch_distribution_seq{seq_length}")

    if os.path.exists(token_file_path) and not force:
        print(f"File {token_file_path} already exists. Skipping...")
        return

    print(f"Processing dataset: {dataset_name} with seq_length: {seq_length}")
    # Ensure output directories exist
    os.makedirs(os.path.dirname(token_file_path), exist_ok=True)
    os.makedirs(dist_file_dir, exist_ok=True)

    # Build and load the data
    recipe = configure_recipe(nodes=1, gpus_per_node=1)
    recipe.data = create_data(
        {
            "paths": os.path.join(folder_path, dataset_name),
            "tokenizer_name": tokenizer_name,
            "global_batch_size": 1,
            "seq_length": seq_length,
        }
    )
    recipe.data.build(5, 1, 1, 1)
    recipe.data.trainer = fdl.build(recipe.trainer)

    # Extract samples and distribution
    output_text, distribution = save_sample_texts(
        recipe.data,
        number_of_texts=int(number_of_data),
        number_of_distributions=int(number_of_data) * 10,
    )

    # Save token text
    with open(token_file_path, "w", encoding="utf-8") as token_file:
        token_file.write(output_text)

    # Save distribution
    with open(
        os.path.join(dist_file_dir, f"{dataset_name}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(distribution, f)


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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-generation of data even if it already exists.",
    )
    args = parser.parse_args()

    with open(os.path.join(args.folder_path, "tokenizer_name.txt"), "r") as f:
        tokenizer_name = f.read().strip()

    for file in os.listdir(args.folder_path):
        dataset_name = file.split(".")[0]

        if file.endswith("text_document.idx"):
            run_dataloader(
                args.folder_path,
                dataset_name,
                tokenizer_name,
                number_of_data=args.number_of_data,
                seq_length=args.seq_length,
                force=args.force,
            )
