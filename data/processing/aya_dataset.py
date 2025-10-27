import os

from utils import create_parser, parse_args, create_executor, add_sampler_filter

from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.filters import LambdaFilter

def append_input_output(data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
    for document in data:
        answer = document.metadata["targets"]
        document.text = [
                {"role": "user", "content": document.text},
                {"role": "assistant", "content": answer},
            ]
        yield document

def apply_chat_template(
    data: DocumentsPipeline, rank: int = 0, world_size: int = 1
) -> DocumentsPipeline:
    from transformers import AutoTokenizer

    tokenizer_name = "OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    for doc in data:
        doc.metadata["conversation"] = doc.text
        doc.text = tokenizer.apply_chat_template(
            doc.text, tokenize=False, enable_thinking=False
        )
        yield doc

if __name__ == "__main__":
    parser = create_parser()
    args = parse_args(parser)

    DATA_PATH = args.data_path
    dataset_name = "aya_dataset"
    output_path = os.path.join(DATA_PATH, dataset_name)

    pipeline = [
        HuggingFaceDatasetReader(
                "CohereLabs/aya_dataset",
                {"name": "default", "split": "train"},
                streaming=True,
                text_key="inputs",
            ),
        LambdaFilter(
            lambda doc: doc.metadata["language_code"] in ["arb", "deu", "eng", "fra", "ita", "nld", "por", "spa", "eus"],
        ),
        append_input_output,
        apply_chat_template,
        JsonlWriter(
            os.path.join(output_path, "data"),
            output_filename="${language_code}/${rank}.jsonl.gz",
        ),
    ]
    add_sampler_filter(pipeline, args.sample_rate)

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        logging_dir=os.path.join(output_path, "logs"),
        job_name=dataset_name,
        tasks=50,
    )

    main_processing_executor.run()
