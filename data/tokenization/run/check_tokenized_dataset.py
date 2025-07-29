

import os
# from megatron.data import indexed_dataset
from nemo_patch import indexed_dataset
import transformers
import tqdm



def inspect_tokens(bin_file):
    assert os.path.isfile(bin_file), f"File {bin_file} does not exist"
    tokenizer_name_file = os.path.join(os.path.dirname(bin_file), "tokenizer_name.txt")
    assert os.path.isfile(tokenizer_name_file), f"File {tokenizer_name_file} does not exist"
    with open(tokenizer_name_file, "r") as f:
        tokenizer_name = f.read().strip()
    basename = os.path.splitext(bin_file)[0]
    idx_file = basename + ".idx"
    assert os.path.isfile(idx_file), f"File {idx_file} does not exist"
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = indexed_dataset.MMapIndexedDataset(basename)
    num_tokens = 0
    num_chars = 0
    num_docs = 0
    for i, data in enumerate(tqdm.tqdm(dataset)):
        text = tokenizer.decode(data)
        text = text.replace("\n", "\\n")
        data = list(data)
        num_tokens += len(data)
        num_chars += len(text)
        num_docs += 1
        if i <= 5 or i >= len(dataset) - 5:
            if len(data) > 20:
                print("   ", data[:10] + [f" ... {len(data)-20} ... "] + data[-10:])
            else:
                print(data)
            if len(text) > 110:
                print(text[:50] + f" ... {len(text)-100} ... " + text[-50:])
            else:
                print(text)
        elif i == 6:
            print("...\n" * 3)

    print(f"Number of documents: {num_docs}")
    print(f"Number of tokens: {num_tokens}")
    print(f"Number of characters: {num_chars}")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("file")
args = parser.parse_args()
bin_file = args.file

# bin_file = "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/tokens_ablation/fineweb2_fra_Latn_cluster_1_text_document.bin"

# bin_file = "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/tokenizer_65k_latin/fineweb2_fra_Latn_cluster_1_text_document.bin"

inspect_tokens(bin_file)