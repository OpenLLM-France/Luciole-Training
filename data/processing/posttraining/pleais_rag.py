from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from functools import partial
from transformers import AutoTokenizer
from datatrove.pipeline.filters import LambdaFilter
from utils import instruct_adapter, _custom_adapter_for_hf, HF_SCHEMA


def format_data(
    data,
    rank: int = 0,
    world_size: int = 1,
    tokenizer=None,
    format="openrag",
):
    RAG_PROMPT = """You are an AI conversational assistant specialized in information retrieval and synthesis.
Your goal is to provide precise, reliable, and well-structured answers using only the retrieved documents.
Prioritize clarity, accuracy, and completeness in your responses.
"""

    CITATION_PROMPT = """
## Citation:

- Use inline <ref name="source_N">...</ref> tags immediately after the claim they support, with no space before the tag.
- Inside the tag, copy or closely paraphrase the relevant excerpt from the source.
- Name each reference after its source number (e.g. source_1, source_4).
- The same name attribute can be reused for multiple citations from the same source, each time with the excerpt relevant to that specific claim.
"""

    SOURCES_PROMPT = """
## Sources:

{sources}
"""
    import random

    for doc in data:
        remove_citation = random.random() < 0.1  # 10% of the time, remove citations
        doc.metadata["remove_citation"] = remove_citation

        SYSTEM_PROMPT = (
            RAG_PROMPT
            + (CITATION_PROMPT if not remove_citation else "")
            + SOURCES_PROMPT.format(sources=doc.metadata.pop("constraints"))
        )

        answer = doc.metadata.pop("synthetic_answer")
        if remove_citation:
            # Remove all <ref name="source_N">...</ref> tags from the answer
            import re

            answer = re.sub(r'<ref name="source_\d+">.*?</ref>', "", answer)

        if "openrag" in format:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": doc.metadata.pop("query")},
                {
                    "role": "assistant",
                    "content": "<think>\n\n"
                    + doc.metadata.get("synthetic_reasoning").strip()
                    + "\n\n</think>\n\n"
                    + answer,
                },
            ]
        else:
            return NotImplementedError(f"Format {format} not implemented yet.")
        doc.metadata["messages"] = messages
        doc.text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
        yield doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--format",
        choices=["tool", "openrag"],
        default="openrag",
        help="Use OpenRAG format (documents in the system prompt)",
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    ##########
    # Process the data
    ##########

    pipeline = [
        HuggingFaceDatasetReader(
            "PleIAs/SYNTH",
            {"split": "train"},
            streaming=True,
            adapter=instruct_adapter,
        ),
        LambdaFilter(
            lambda doc: doc.metadata["model"].strip() == "qwen-3-8b-rag",
        ),
        partial(format_data, tokenizer=tokenizer, format=args.format),
        JsonlWriter(
            f"{DATA_PATH}/pleais_rag/{args.format}/data",
            output_filename="${language}/${rank}.jsonl.gz",
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/pleais_rag/{args.format}/logs",
        job_name=f"pleais_rag_{args.format}",
        tasks=10,
        time="00:30:00",
        # partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )

    if args.debug:
        main_processing_executor.run()

    else:
        ##########
        # Push to hub
        ##########
        pipeline = [
            JsonlReader(
                f"{DATA_PATH}/pleais_rag/{args.format}/data",
            ),
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-France/PleAIs_RAG",
                private=True,
                local_working_dir=f"{DATA_PATH}/pleais_rag/{args.format}/data_hf",
                output_filename="data/" + args.format + "/${language}/${rank}.parquet",
                adapter=partial(
                    _custom_adapter_for_hf,
                    source="PleIAs/SYNTH",
                    id_key="synth_id",
                    language=None,
                    language_key="language",
                    conversation_key="messages",
                ),
                cleanup=True,
                expand_metadata=False,
                schema=HF_SCHEMA,
            ),
        ]

        hf_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/pleais_rag/{args.format}/logs_hf",
            job_name="pleais_rag",
            tasks=1,
            skip_completed=not args.force,
            depends=main_processing_executor,
        )

        hf_executor.run()
