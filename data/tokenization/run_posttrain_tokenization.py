import os
import numpy as np
import yaml
import argparse
from datasets import load_dataset 
from transformers import AutoTokenizer

"""
For posttraining datasets, tokenizes datasets and saves as numpy arrays for visualization (see visualize_datamix.py).
"""

def get_token_counts(tokenizer, dataset_path, columns=["messages"], split="train"):
    
    if isinstance(columns, str):
        columns = [columns]

    # Stream the dataset
    dataset = load_dataset(dataset_path, streaming=True, split=split)
    
    # Process batches 
    def process_batch(batch):

        num_rows = len(next(iter(batch.values())))
        batch_totals = np.zeros(num_rows, dtype=np.int32)

        for col in columns:
            templated_texts = [
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                for messages in batch[col]
            ]
        
            encoded = tokenizer(
                templated_texts, 
                add_special_tokens=False, 
                padding=False, 
                truncation=False
            )
            
            lengths = [len(ids) for ids in encoded["input_ids"]]
            batch_totals += lengths

        return {"total_tokens": batch_totals.tolist()}

    # Batch map over the stream (releases GIL for parallel CPU counting)
    processed_stream = dataset.map(
        process_batch, 
        batched=True, 
        batch_size=4096, 
        remove_columns=dataset.column_names
    )
    
    # 4. Collect counts into a lightweight Python list
    counts_list = []
    for row in processed_stream:
        counts_list.append(row["total_tokens"])
        
    # 5. Convert to a highly efficient NumPy array
    # We specify int32 to keep the array compact (unless your sequences are >2B tokens)
    return np.array(counts_list, dtype=np.int32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "yaml_file",
        type=str,
        help=".yaml file that contains the information about the datasets you want to tokenize. See for example configs/luciole_instruct_1B_olmomix_2.yaml.",
    )
    parser.add_argument(
        "--columns", 
        nargs="+",
        default=["messages"],
        help="space separated list of column names for token counts."
    )

    args = parser.parse_args()
    yaml_file = args.yaml_file
    columns = args.columns 
    
    with open(yaml_file, "r") as f:
        yaml_data = yaml.safe_load(f)

    tokens_dataset_path = yaml_data["output_dir"]
    os.makedirs(tokens_dataset_path, exist_ok=True)
    
    TOKENIZER = AutoTokenizer.from_pretrained(yaml_data["tokenizer_path"])
    
    # Iterate through each dataset entry
    for dataset in yaml_data["datasets"]:
        name = dataset["name"]
        data_path = dataset["path"]
        npy_path = os.path.join(tokens_dataset_path, f"{name}.npy")
        
        if os.path.isfile(npy_path):#skip datasets which have already been tokenized
            continue

        if os.path.isdir(data_path):
            print("--------------------------------------")
            print(f"🚀 Processing dataset: {name}")
            print("--------------------------------------")

            counts = get_token_counts(TOKENIZER, data_path, columns)
            
            np.save(npy_path, counts)
            
            print(f"✅ Token arrays saved at {npy_path}.")
        else:
            print("--------------------------------------")
            print(f"❌ Dataset not found for {name} at {data_path}.")
            print("--------------------------------------")

#python run_posttrain_tokenization.py /lustre/fsn1/projects/rech/qgz/uhm96nw/OpenLLM-BPI-Training/data/tokenization/configs/luciole_dpo_mix5.yaml --columns accept reject

