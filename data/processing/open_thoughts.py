from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline
from functools import partial

from transformers import AutoTokenizer

import os
import re

def convert_message(message):
    # Convert from the old format to the new format
    message["role"] = {"human": "user", "gpt": "assistant"}[message.pop("from")]
    message["content"] = message.pop("value")
    return message

def convert_messages(data: DocumentsPipeline, rank: int = 0, world_size: int = 1, tokenizer=None) -> DocumentsPipeline:
    for document in data:
        messages = [convert_message(m) for m in document.text]
        document.text = tokenizer.apply_chat_template(messages, tokenize=False)
        # Remove empty think tags, caused by unfinished thoughts
        if "<think>\n\n</think>\n\n<think>" in document.text:
            document.text = document.text.replace("<think>\n\n</think>\n\n<think>", "<think>\n")
            document.text = document.text[:-11] # remove "<|im_end|>\n" at the end
        # Remove Chinese-heavy lines that can appear in thoughts
        document.text = remove_chinese_heavy_lines(document.text)
        document.metadata["conversation"] = messages
        yield document

def remove_chinese_heavy_lines(text):
    lines = text.splitlines()
    cleaned_lines = []
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    latin_pattern = re.compile(r'[A-Za-z]')

    for line in lines:
        chinese_chars = len(chinese_pattern.findall(line))
        latin_chars = len(latin_pattern.findall(line))

        if chinese_chars and chinese_chars > latin_chars:
            continue  # Skip this line

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)
    DATA_PATH = args.data_path

    dataset_name = "open_thoughts" 
    output_path = os.path.join(DATA_PATH, dataset_name)

    pipeline = [
        ParquetReader(
            "hf://datasets/open-thoughts/OpenThoughts3-1.2M/data", glob_pattern = "train-*.parquet",
            text_key="conversations",
        ),
        partial(convert_messages,
                tokenizer=AutoTokenizer.from_pretrained("OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct"),
        ),
        JsonlWriter(
            f"{output_path}/data",
            output_filename="${domain}/rank${rank}.jsonl.gz"),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{output_path}/logs",
        job_name=dataset_name,
        tasks=50,
    )

    main_processing_executor.run()