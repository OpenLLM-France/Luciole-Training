from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from functools import partial
from transformers import AutoTokenizer
from datatrove.pipeline.filters import LambdaFilter
from utils import instruct_adapter, _custom_adapter_for_hf, HF_SCHEMA
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


class CitationFilter(BaseFilter):
    name = "✳️  Citation Filtering"

    def __init__(self, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)

    @staticmethod
    def extract_sources(text: str) -> dict:
        import re

        matches = re.findall(r"<(source_\d+)>(.*?)</\1>", text, re.DOTALL)
        return {key: value.strip() for key, value in matches}

    def replace_citation(self, doc: Document, sources: dict, synthetic_answer: str):
        import re

        def drop_unsupported_citation(match):
            source_id, content = match.group(1), match.group(2).strip()
            # Keep the citation only if its excerpt is found in the cited source.
            if content in sources.get(source_id, ""):
                return match.group(0)
            return ""

        cleaned_answer = re.sub(
            r'<ref name="(.*?)">(.*?)</ref>',
            drop_unsupported_citation,
            synthetic_answer,
            flags=re.DOTALL,
        )

        return cleaned_answer

    @staticmethod
    def has_citation(answer: str) -> bool:
        import re

        # A document still has citations if any <ref> tag survived the cleaning.
        return bool(
            re.search(r'<ref name="(.*?)">(.*?)</ref>', answer, flags=re.DOTALL)
        )

    def filter(self, doc: Document) -> bool:
        sources = self.extract_sources(doc.metadata["constraints"])

        cleaned_answer = self.replace_citation(
            doc, sources, doc.metadata["synthetic_answer"]
        )
        doc.metadata["synthetic_answer"] = cleaned_answer
        doc.metadata["has_citation"] = self.has_citation(cleaned_answer)

        return True


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

- Use inline <ref name="source_N">...</ref> tags immediately after the claim they support.
- Inside the tag, copy the relevant excerpt from the source.
- Name each reference after its source number (e.g. source_1, source_4).
- The same name attribute can be reused for multiple citations from the same source, each time with the excerpt relevant to that specific claim.
"""

    SOURCES_PROMPT = """
## Sources:

{sources}
"""
    for doc in data:
        # Unsupported citations were already stripped by CitationFilter, so the
        # answer keeps its valid <ref> tags. Only include the citation prompt
        # when at least one citation survived.
        has_citation = doc.metadata["has_citation"]

        SYSTEM_PROMPT = (
            RAG_PROMPT
            + (CITATION_PROMPT if has_citation else "")
            + SOURCES_PROMPT.format(sources=doc.metadata.pop("constraints"))
        )

        answer = doc.metadata.pop("synthetic_answer")

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
        "OpenLLM-France/tokenizer_128k-arab-regional_v2_instruct_train"
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
        CitationFilter(),
        partial(format_data, tokenizer=tokenizer, format=args.format),
        JsonlWriter(
            f"{DATA_PATH}/pleais_rag_citation/{args.format}/data",
            output_filename="citation_${has_citation}/${rank}.jsonl.gz",
            expand_metadata=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/pleais_rag_citation/{args.format}/logs",
        job_name=f"pleais_rag_citation_{args.format}",
        tasks=10,
        time="00:30:00",
        # partition="cpu_p1",
        qos="qos_cpu-dev",
        skip_completed=not args.force,
    )

    if args.debug:
        main_processing_executor.run()

    # else:
    #     main_processing_executor.run()
    else:
        ##########
        # Push to hub
        ##########
        pipeline = [
            JsonlReader(
                f"{DATA_PATH}/pleais_rag_citation/{args.format}/data",
            ),
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-France/PleAIs_RAG_v2",
                private=True,
                local_working_dir=f"{DATA_PATH}/pleais_rag_citation/{args.format}/data_hf",
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
