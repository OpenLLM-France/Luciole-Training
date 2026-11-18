from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from datatrove.pipeline.filters import LambdaFilter
from utils import instruct_adapter, ChatTokenCount
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter


# <ref name="source_N">excerpt</ref>
REF_RE = r'<ref name="(.*?)">(.*?)</ref>'
# Quote with its source, written as source_N, source N, Source N or sources N, M:
#   "excerpt" (source_1) / [source 2] / (cf. source 3) / (source_1, source_4) / (sources 1 and 2)
#   "excerpt" (Source 1, Socrates via Xenophon)     <- extra text after the source
#   "excerpt" ● (Source 5) / "excerpt"** (Source 1) <- confidence marker or bold
_SOURCES = r"(sources?[ _]?\d+(?:\s*(?:,|and|&)\s*(?:sources?[ _]?)?\d+)*)"
QUOTE_THEN_SOURCE_RE = (
    r'"([^"\n]+)"\s*(?:\*\*|[●○◐◑])?\s*[\(\[]\s*(?:cf\.\s*)?' + _SOURCES
    + r"(?:\s*[,;:]\s*[^\)\]\d\n]*)?\s*[\)\]]"
)
#   Source 1: "excerpt" / (source_1: "excerpt")
SOURCE_THEN_QUOTE_RE = r"\b" + _SOURCES + r'\s*:\s*"([^"\n]+)"'


class CitationFilter(BaseFilter):
    """Drop rows with a citation not found verbatim in the source it names, checked on
    both ``synthetic_answer`` and ``synthetic_reasoning`` (strict: only leading/trailing
    whitespace is ignored, so a quote shortened with "..." or with changed quote marks
    fails):

    - <ref name="source_N">excerpt</ref>: the excerpt must be non-empty and in source_N;
    - quotes with their source (QUOTE_THEN_SOURCE_RE, SOURCE_THEN_QUOTE_RE): the quote
      must be in one of the named sources. Quotes with no source next to them (e.g.
      quoting the query) are not checked.

    Sets ``metadata["has_citation"]`` (a <ref> in the answer).
    """

    name = "✳️  Citation Filtering"

    def __init__(self, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)

    @staticmethod
    def extract_sources(text: str) -> dict:
        import re

        matches = re.findall(r"<(source_\d+)>(.*?)</\1>", text, re.DOTALL)
        return {key: value.strip() for key, value in matches}

    def replace_citation(self, doc: Document, sources: dict, synthetic_answer: str):
        """Remove the unsupported <ref> (used by pleais_rag_dpo_filter.DPOCitationFilter)."""
        import re

        def drop_unsupported_citation(match):
            source_id, content = match.group(1), match.group(2).strip()
            # Keep the citation only if its excerpt is found in the cited source.
            if content in sources.get(source_id, ""):
                return match.group(0)
            return ""

        return re.sub(REF_RE, drop_unsupported_citation, synthetic_answer, flags=re.DOTALL)

    @staticmethod
    def has_citation(answer: str) -> bool:
        import re

        return bool(re.search(REF_RE, answer, flags=re.DOTALL))

    @staticmethod
    def all_citations_supported(sources: dict, text: str) -> bool:
        """Every <ref> points to an existing source and quotes a non-empty excerpt of it."""
        import re

        for source_id, content in re.findall(REF_RE, text, flags=re.DOTALL):
            content = content.strip()
            if not content or content not in sources.get(source_id, ""):
                return False
        return True

    @staticmethod
    def all_quotes_supported(sources: dict, text: str) -> bool:
        """Every quote with its source is found in one of the named sources."""
        import re

        pairs = re.findall(QUOTE_THEN_SOURCE_RE, text, flags=re.I)
        pairs += [(q, c) for c, q in re.findall(SOURCE_THEN_QUOTE_RE, text, flags=re.I)]
        for quote, cited in pairs:
            quote = quote.strip()
            cited_sources = ["source_" + n for n in re.findall(r"\d+", cited)]
            if not quote or not any(quote in sources.get(s, "") for s in cited_sources):
                return False
        return True

    def filter(self, doc: Document) -> bool:
        sources = self.extract_sources(doc.metadata["constraints"])
        answer = doc.metadata["synthetic_answer"]
        reasoning = doc.metadata["synthetic_reasoning"]

        # Texts are kept untouched; the whole row is dropped on any wrong citation.
        for field, text in (("answer", answer), ("reasoning", reasoning)):
            if not self.all_citations_supported(sources, text):
                return False, f"unsupported_ref_{field}"
            if not self.all_quotes_supported(sources, text):
                return False, f"unsupported_quote_{field}"

        doc.metadata["has_citation"] = self.has_citation(answer)
        return True


class DocumentsFilter(BaseFilter):
    """Keep rows with at least `min_documents` sources and at least `min_words` words
    in total across the sources (whitespace-split, section title included). Removes
    the rows built on one or a few truncated snippets, where SYNTH mostly answers
    that it cannot answer."""

    name = "📚 Documents Filtering"

    def __init__(self, min_documents: int = 2, min_words: int = 300, exclusion_writer: DiskWriter = None):
        super().__init__(exclusion_writer)
        self.min_documents = min_documents
        self.min_words = min_words

    def filter(self, doc: Document) -> bool:
        sources = CitationFilter.extract_sources(doc.metadata["constraints"])
        if len(sources) < self.min_documents:
            return False, "too_few_documents"
        if sum(len(text.split()) for text in sources.values()) < self.min_words:
            return False, "too_few_words"
        return True


class InternalKnowledgeFilter(BaseFilter):
    """Drop rows whose reasoning or answer says it relies on the model's own knowledge.

    Reasoning: SYNTH labels injected outside facts with "internal knowledge"
    ("**Historical context from internal knowledge:**", "Internal knowledge: ...",
    "(internal knowledge)") or "domain knowledge" ("### 4. Domain Knowledge Integration",
    "**Context & domain knowledge:**", "(from domain knowledge)"). Every "internal
    knowledge" is a hit; "domain knowledge" is a hit unless it is what the query
    needs ("requires (technical) domain knowledge", "multi-domain knowledge").
    Answer (<ref> excerpts removed): only first-person phrasing ("my knowledge",
    "from internal knowledge"); "our knowledge of Egyptian history" or "NASA's internal
    knowledge" are about the topic and are not matched.
    """

    name = "🧠 Internal Knowledge Filtering"

    REASONING_INTERNAL = r"\binternal knowledge\b"
    REASONING_DOMAIN = r"(?<!multi-)(?<!cross-)\bdomain knowledge\b"
    NEEDED_BEFORE = r"\b(?:requir\w*|need\w*|demand\w*|involv\w*|lack\w*)\s+(?:\w+[- ]){0,3}$"
    ANSWER_SELF = (
        r"\bmy\s+(?:own\s+|internal\s+|general\s+|background\s+)?knowledge\b"
        r"|\b(?:from|based on|using)\s+(?:my\s+)?internal knowledge\b"
    )

    @classmethod
    def reasoning_uses_own_knowledge(cls, reasoning: str) -> bool:
        import re

        if re.search(cls.REASONING_INTERNAL, reasoning, flags=re.I):
            return True
        for m in re.finditer(cls.REASONING_DOMAIN, reasoning, flags=re.I):
            before = reasoning[max(0, m.start() - 40) : m.start()]
            if not re.search(cls.NEEDED_BEFORE, before, flags=re.I):
                return True
        return False

    @classmethod
    def answer_uses_own_knowledge(cls, answer: str) -> bool:
        import re

        answer = re.sub(r'<ref name="[^"]*">.*?</ref>', "", answer, flags=re.S)
        return bool(re.search(cls.ANSWER_SELF, answer, flags=re.I))

    def filter(self, doc: Document) -> bool:
        if self.answer_uses_own_knowledge(doc.metadata["synthetic_answer"]):
            return False, "own_knowledge_answer"
        if self.reasoning_uses_own_knowledge(doc.metadata["synthetic_reasoning"]):
            return False, "own_knowledge_reasoning"
        return True


# System prompt of the chat template (PR #4 of OpenLLM-France/tokenizer_128k_dual)
# reduced to the rules the SYNTH answers follow: only the retrieved documents, and
# <ref id="N"> citations (exact excerpt, existing document, no nesting: enforced by the
# filters). The intro line replaces the template's "You are an AI conversational assistant
# ..." / "Your goal is ..." / "Prioritize ..." lines, and "exclusively" is dropped (SYNTH
# answers do infer). Left out: "Never infer,
# assume, or rely on any external knowledge" (SYNTH answers do infer), "if insufficient, invite the user to rephrase or add
# documents" (not done in SYNTH), "one <ref> per source", the table example, language
# consistency and "concise yet complete".
RAG_SYSTEM_PROMPT = """You answer the user's question from the retrieved documents provided below.

## Rules

- Base your answer on the information contained in the retrieved documents.
- When quoting from the retrieved documents, wrap each excerpt inline with `<ref id="N">excerpt</ref>`, where `N` is the `id` of the cited `<document>` and `excerpt` is copied verbatim from that document. Never nest `<ref>` tags."""



def render_documents(texts: list) -> str:
    """<documents> block exactly as the chat template renders `documents` (no title)."""
    block = "Here are the retrieved documents:\n\n<documents>"
    for i, text in enumerate(texts, start=1):
        block += f'\n<document id="{i}">\n' + text.strip() + "\n</document>"
    return block + "\n</documents>"


HUB_DATASET = "OpenLLM-France/pleias_rag"
RAG_TOKENIZER = "OpenLLM-France/tokenizer_128k_dual"
RAG_TOKENIZER_REVISION = "refs/pr/4"


class ThinkingSplit(BaseFilter):
    """Keep the reasoning for a fixed fraction of the rows only:
    ``metadata["thinking"]`` = "with_thinking" for ``thinking_ratio`` of the rows,
    "without_thinking" (reasoning not used) otherwise. The choice hashes ``seed:id``, so it is reproducible and
    does not depend on the number of tasks."""

    name = "🎲 Thinking Split"

    def __init__(self, thinking_ratio: float = 0.1, seed: int = 0):
        super().__init__()
        self.thinking_ratio = thinking_ratio
        self.seed = seed

    def filter(self, doc: Document) -> bool:
        import hashlib

        h = hashlib.md5(f"{self.seed}:{doc.id}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2**64
        doc.metadata["thinking"] = "with_thinking" if u < self.thinking_ratio else "without_thinking"
        self.stat_update(doc.metadata["thinking"])
        return True


class RagFormat(BaseFilter):
    """SYNTH row -> messages: system (RAG_SYSTEM_PROMPT + <documents> block, formatted
    like the chat template), user, assistant. No `documents` column.

    Only the citation tags change: <ref name="source_N"> -> <ref id="N">, where
    <document id="N"> holds <source_N> (sources are always source_1..source_k in order;
    other rows are dropped). Rows with a broken <ref> tag left in the assistant turn are
    dropped. The SYNTH wording (answer and reasoning) is kept as is.
    Thinking rows get "<think>\n{reasoning}\n</think>\n\n{answer}", non-thinking rows
    (ThinkingSplit) the answer only.
    """

    name = "📝 RAG Format"

    def filter(self, doc: Document) -> bool:
        import re

        sources = CitationFilter.extract_sources(doc.metadata["constraints"])
        if list(sources) != [f"source_{i}" for i in range(1, len(sources) + 1)]:
            return False, "non_contiguous_sources"
        answer = re.sub(
            r'<ref name="source_(\d+)">', r'<ref id="\1">', doc.metadata["synthetic_answer"]
        ).strip()
        reasoning = doc.metadata["synthetic_reasoning"].strip()
        if doc.metadata["thinking"] == "with_thinking":
            answer = "<think>\n" + reasoning + "\n</think>\n\n" + answer
        # Broken tags the citation filter cannot see (answer cut inside a tag,
        # '<ref name="source_2", "source_4">' never closed, ...): every remaining <ref
        # must be a complete <ref id="N">...</ref>.
        well_formed = re.findall(r'<ref id="\d+">.*?</ref>', answer, flags=re.S)
        if answer.count("<ref") != len(well_formed) or answer.count("</ref>") != len(well_formed):
            return False, "malformed_ref"
        doc.metadata["messages"] = [
            {
                "role": "system",
                "content": RAG_SYSTEM_PROMPT + "\n\n" + render_documents(list(sources.values())),
            },
            {"role": "user", "content": doc.metadata["query"].strip()},
            {"role": "assistant", "content": answer},
        ]
        doc.metadata["dataset"] = "PleIAs/SYNTH"
        doc.text = ""
        return True


class RagChatTokenCount(ChatTokenCount):
    """ChatTokenCount with enable_thinking set per row (thinking / non-thinking)."""

    def filter(self, doc: Document) -> bool:
        self.chat_template_kwargs = {
            "training_mode": True,
            "enable_thinking": doc.metadata["thinking"] == "with_thinking",
        }
        return super().filter(doc)


def rag_hub_adapter(self, document):
    """Pinned Hub columns of the "formatted_chat" subset, like tool_calling/utils.py:hub_adapter
    (no text column).
    Render with ``tokenizer.apply_chat_template(messages)``."""
    return {
        "id": str(document.metadata.get("synth_id") or document.id),
        "dataset": str(document.metadata.get("dataset") or ""),
        "messages": [
            {"role": str(m.get("role", "")), "content": str(m.get("content") or "")}
            for m in document.metadata.get("messages") or []
        ],
        "language": str(document.metadata.get("language") or ""),
        "has_citation": bool(document.metadata.get("has_citation")),
        "token_count": int(document.metadata["token_count"]),
        "assistant_token_count": int(document.metadata["assistant_token_count"]),
    }


SYNTH_COLUMNS = [
    "language", "exercise", "model", "query", "query_seed_url", "query_seed_text",
    "additional_seed_url", "seed_license", "constraints", "script",
    "synthetic_reasoning", "synthetic_answer",
]


def synth_raw_hub_adapter(self, document):
    """Raw SYNTH row as on PleIAs/SYNTH (synth_id + its columns), for data/raw/."""
    row = {"synth_id": str(document.id)}
    row.update({k: str(document.metadata.get(k) or "") for k in SYNTH_COLUMNS})
    row["words"] = int(document.metadata.get("words") or 0)
    return row


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--format",
        choices=["tool", "openrag"],
        default="openrag",
        help="Use OpenRAG format (documents in the system prompt)",
    )
    parser.add_argument(
        "--step",
        choices=["extract", "process", "all", "push_raw", "push_citation_filter", "format", "push_formatted_chat"],
        default="all",
        help="extract: SYNTH -> RAG rows only; process: answer + reasoning citation "
        "filters (saved locally); all: extract + process; "
        "push_raw: push the extracted rows to data/raw/ of the Hub dataset; "
        "push_citation_filter: push the process output to data/citation_filter/; "
        "format: filters + messages (saved locally); "
        "push_formatted_chat: push the formatted rows to data/formatted_chat/ of the Hub dataset",
    )
    parser.add_argument(
        "--workers", type=int, default=32, help="Parallel local workers"
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path
    RAW_DIR = f"{DATA_PATH}/pleias_rag_citation/synth_rag"

    ##########
    # Step 1: keep only the RAG rows of PleIAs/SYNTH (500 parquet files, ~236 GB;
    # every file mixes all models, so everything is read). Dropped rows are not saved.
    ##########

    extract_pipeline = [
        ParquetReader(
            "hf://datasets/PleIAs/SYNTH",
            glob_pattern="*.parquet",
            id_key="synth_id",
            adapter=instruct_adapter,
        ),
        LambdaFilter(
            lambda doc: doc.metadata["model"].strip() == "qwen-3-8b-rag",
        ),
        JsonlWriter(RAW_DIR, expand_metadata=True),
    ]

    extract_executor = create_executor(
        extract_pipeline,
        local=args.local,
        debug=args.debug,
        limit_debug=args.limit_debug,
        logging_dir=f"{DATA_PATH}/pleias_rag_citation/logs_synth_rag",
        job_name="pleias_synth_rag",
        tasks=500,  # one parquet file per task
        workers=args.workers,
        time="02:00:00",
        partition="prepost",  # reads from the Hub: needs network
        skip_completed=not args.force,
    )

    ##########
    # Step 2: citation filter, saved locally (same fields as the raw rows): on both the
    # answer and the reasoning, drop rows with any <ref> or quoted "excerpt" (source_N) /
    # Source N: "excerpt" not verbatim in the cited source.
    # --step push_citation_filter pushes it to data/citation_filter/ (subset citation_filter).
    ##########

    CITATION_DIR = f"{DATA_PATH}/pleias_rag_citation/citation_filter"
    main_processing_executor = create_executor(
        [
            JsonlReader(RAW_DIR, adapter=instruct_adapter),
            CitationFilter(),
            JsonlWriter(f"{CITATION_DIR}/data", expand_metadata=True),
        ],
        local=args.local,
        debug=args.debug,
        limit_debug=args.limit_debug,
        logging_dir=f"{CITATION_DIR}/logs",
        job_name="pleias_rag_citation_filter",
        tasks=32,
        workers=args.workers,
        time="00:30:00",
        partition="cpu_p1",
        skip_completed=not args.force,
        depends=extract_executor if args.step == "all" else None,
    )

    ##########
    # Step 3: from the step 2 output: documents filter (>= 2 sources, >= 300 words),
    # own-knowledge filter, reasoning kept for 45% of the rows (thinking) and removed for
    # the other 55% (non-thinking), which gives ~1:1 total tokens between the two splits,
    # messages (system: rules + <documents>; user; assistant), broken-<ref> filter,
    # token count; saved locally per split (with_thinking / without_thinking).
    ##########

    FORMAT_DIR = f"{DATA_PATH}/pleias_rag_citation/formatted_chat"
    format_executor = create_executor(
        [
            JsonlReader(f"{CITATION_DIR}/data", adapter=instruct_adapter),
            DocumentsFilter(min_documents=2, min_words=300),
            InternalKnowledgeFilter(),
            ThinkingSplit(thinking_ratio=0.45, seed=0),  # ~1:1 total tokens thinking vs non-thinking
            RagFormat(),
            RagChatTokenCount(
                RAG_TOKENIZER,
                tokenizer_revision=RAG_TOKENIZER_REVISION,
                max_tokens=2**16,
                exclusion_writer=JsonlWriter(f"{FORMAT_DIR}/too_long"),
            ),
            JsonlWriter(
                f"{FORMAT_DIR}/data",
                output_filename="${thinking}/${rank}.jsonl.gz",
                expand_metadata=True,
            ),
        ],
        local=args.local,
        debug=args.debug,
        limit_debug=args.limit_debug,
        logging_dir=f"{FORMAT_DIR}/logs",
        job_name="pleias_rag_format",
        tasks=32,
        workers=args.workers,
        time="01:00:00",
        partition="prepost",  # the tokenizer revision is downloaded from the Hub
        skip_completed=not args.force,
    )

    if args.step == "format":
        format_executor.run()
    elif args.step == "push_formatted_chat":
        # Subset "formatted_chat" of OpenLLM-France/pleias_rag (splits thinking / non_thinking).
        create_executor(
            [
                JsonlReader(f"{FORMAT_DIR}/data", adapter=instruct_adapter),
                HuggingFaceDatasetWriter(
                    dataset=HUB_DATASET + "_debug" * args.debug,
                    private=True,
                    local_working_dir=f"{FORMAT_DIR}/data_hf",
                    output_filename="data/formatted_chat/${thinking}/${language}/${rank}.parquet",
                    adapter=rag_hub_adapter,
                    cleanup=True,
                    expand_metadata=False,
                    commit_message=args.commit,
                ),
            ],
            local=args.local,
            debug=args.debug,
            limit_debug=args.limit_debug,
            logging_dir=f"{FORMAT_DIR}/logs_hf",
            job_name="pleias_rag_formatted_chat_hf",
            tasks=8,
            workers=args.workers,
            time="02:00:00",
            partition="prepost",
            skip_completed=not args.force,
        ).run()
    elif args.step == "push_citation_filter":
        # Subset "citation_filter": rows passing both citation filters, unchanged.
        create_executor(
            [
                JsonlReader(f"{CITATION_DIR}/data", adapter=instruct_adapter),
                HuggingFaceDatasetWriter(
                    dataset=HUB_DATASET + "_debug" * args.debug,
                    private=True,
                    local_working_dir=f"{CITATION_DIR}/data_hf",
                    output_filename="data/citation_filter/${language}/${rank}.parquet",
                    adapter=synth_raw_hub_adapter,
                    cleanup=True,
                    expand_metadata=False,
                    commit_message=args.commit,
                ),
            ],
            local=args.local,
            debug=args.debug,
            limit_debug=args.limit_debug,
            logging_dir=f"{CITATION_DIR}/logs_hf",
            job_name="pleias_rag_citation_filter_hf",
            tasks=8,
            workers=args.workers,
            time="02:00:00",
            partition="prepost",
            skip_completed=not args.force,
        ).run()
    elif args.step == "extract":
        extract_executor.run()
    elif args.step == "push_raw":
        # Subset "raw" of OpenLLM-France/pleias_rag: the RAG rows of SYNTH, unchanged.
        push_raw_executor = create_executor(
            [
                JsonlReader(RAW_DIR, adapter=instruct_adapter),
                HuggingFaceDatasetWriter(
                    dataset=HUB_DATASET + "_debug" * args.debug,
                    private=True,
                    local_working_dir=f"{DATA_PATH}/pleias_rag_citation/synth_rag_hf",
                    output_filename="data/raw/${language}/${rank}.parquet",
                    adapter=synth_raw_hub_adapter,
                    cleanup=True,
                    expand_metadata=False,
                    commit_message=args.commit,
                ),
            ],
            local=args.local,
            debug=args.debug,
            limit_debug=args.limit_debug,
            logging_dir=f"{DATA_PATH}/pleias_rag_citation/logs_synth_rag_hf",
            job_name="pleias_synth_rag_hf",
            tasks=8,
            workers=args.workers,
            time="02:00:00",
            partition="prepost",
            skip_completed=not args.force,
        )
        push_raw_executor.run()
    else:
        main_processing_executor.run()
