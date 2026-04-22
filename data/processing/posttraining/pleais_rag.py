from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from functools import partial
from transformers import AutoTokenizer
from utils import instruct_adapter
from datatrove.pipeline.filters import LambdaFilter


def format_data(
    data, rank: int = 0, world_size: int = 1, tokenizer=None, openrag_format=False
):
    SYSTEM_PROMPT = """You are a helpful AI assistant named Luciole, trained by LINAGORA and OpenLLM France.

## Rules:
- Use only the provided sources; no external knowledge.
- Write clear paragraph answers.
- Every factual claim must include an inline citation: <ref name="source_k">exact quoted text</ref>
- The source name must match the provided source IDs (e.g., source_1).
- Quotes inside <ref> must be copied exactly from the sources.
- Do not add unsupported information.
- If the answer is not in the sources, say so explicitly.

## Sources:
{sources}
"""
    for doc in data:
        # urls = [doc.metadata["query_seed_url"]]
        # tool_calls = [url for url in urls]
        # tool_messages = [
        #     {"role": "assistant", "content": "", "tool_calls": tool_calls},
        #     {"role": "tool", "content": "TOOL RESPONSE"},
        # ]
        if args.openrag_format:
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(
                        sources=doc.metadata.pop("constraints")
                    ),
                },
                {"role": "user", "content": doc.metadata.pop("query")},
                {"role": "assistant", "content": doc.metadata.pop("synthetic_answer")},
            ]
        else:
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": doc.metadata.pop("query")},
                {"role": "assistant", "content": "TOOL CALL"},
                {"role": "tool", "content": doc.metadata.pop("constraints")},
                {"role": "assistant", "content": doc.metadata.pop("synthetic_answer")},
            ]
        doc.metadata["messages"] = messages
        doc.text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--openrag_format",
        action="store_true",
        help="Use OpenRAG format (documents in the system prompt)",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    format = "openrag" if args.openrag_format else "tool"

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    # Add adapter that add empty text if the text field is None, to avoid skipping data
    pipeline = [
        HuggingFaceDatasetReader(
            "PleIAs/SYNTH",
            {"split": "train"},
            streaming=True,
            adapter=instruct_adapter,
        ),
        LambdaFilter(
            lambda doc: doc.metadata["model"].strip() == "qwen-3-8b-rag",
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/pleais_norag/data",
                output_filename="${language}/rank${rank}.jsonl.gz",
            ),
        ),
        partial(format_data, tokenizer=tokenizer, openrag_format=args.openrag_format),
        JsonlWriter(
            f"{DATA_PATH}/pleais_rag/{format}/data",
            output_filename="${language}/rank${rank}.jsonl.gz",
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/pleais_rag/{format}/logs",
        job_name="pleais_rag_{format}",
        tasks=10,
        time="00:30:00",
        # partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )
    main_processing_executor.run()
