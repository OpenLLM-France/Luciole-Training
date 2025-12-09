import os
from nemo_patch import indexed_dataset
from transformers import AutoTokenizer
import orjson
import random


def get_length_suffix(length):
    if length < 4096:
        suffix = "_0-4k"
    elif length < 16384:
        suffix = "_4k-16k"
    elif length < 65536:
        suffix = "_16k-64k"
    else:
        suffix = "_64k+"
    return suffix


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Subsample MMap Indexed datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input", type=str, help="Input indexed dataset filename (without extension)"
    )
    parser.add_argument("output", type=str, help="Output folder")
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=128000,
        help="Vocabulary size (is it larger or not than 65500?)",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        help="Sample rate for subsampling (0.0 - 1.0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode, process only first 10 documents",
    )

    args = parser.parse_args()

    input = args.input
    output = args.output
    os.makedirs(output, exist_ok=True)
    vocab_size = args.vocab_size

    name = os.path.basename(input).replace("_text_document", "")

    assert input
    assert os.path.exists(input + ".bin"), "Input .bin file must exist."
    assert os.path.exists(input + ".idx"), "Input .idx file must exist."

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-BPI/tokenizer_128k-arab-regional_v2"
    )

    dataset = indexed_dataset.MMapIndexedDataset(input)
    for i, doc in enumerate(dataset):
        if random.random() > args.sample_rate:
            continue
        length = len(doc)
        suffix = get_length_suffix(length)
        text = tokenizer.decode(doc, skip_special_tokens=True)
        record = {
            "index": i,
            "dataset": name,
            "input": "",
            "output": text,
            "length": length,
        }
        # write text in jsonl file
        output_text_file = os.path.join(output, name + suffix + ".jsonl")
        with open(output_text_file, "ab") as f:
            f.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
        if args.debug and i > 10:
            break

    completed_text_file = os.path.join(output, "completion", name)
    os.makedirs(os.path.dirname(completed_text_file), exist_ok=True)
    with open(completed_text_file, "w"):
        pass

    print(f"✅ Finished splitting long context documents for dataset {name}.")
