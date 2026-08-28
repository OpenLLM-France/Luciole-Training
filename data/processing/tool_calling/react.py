# Script that input instruct data and output preference data for posttraining

import os

# ~500 concurrent retriever encodes otherwise exhaust OpenBLAS's buffer pool:
# "BLAS : Program is Terminated. Because you tried to allocate too many memory
# regions." Must be set before numpy/torch are imported below.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

# Imports below are intentionally placed after the env-var setup above.
# ruff: noqa: E402
import json
import pathlib
import random
import pyarrow as pa
import re
from utils import create_parser, parse_args, create_executor
from datatrove.data import Document
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from datatrove.pipeline.inference.run_inference import InferenceConfig
from datatrove.pipeline.inference.tool_calling import ToolCallingInferenceRunner
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.filters import SamplerFilter
from functools import partial
from transformers import AutoTokenizer
from utils import (
    apply_chat_template,
    instruct_adapter,
    add_system_prompt,
    NemoRLFormat,
)
from react_wiki_env import WikiEnv, SCORERS
from react_tools import TOOL_SETS, build_tool_sets
from react_retriever_env import new_retriever_env
from react_wiki_structured_env import new_structured_env

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",  # For testing only, not recommended for generation
    "qwen3-32b": "Qwen/Qwen3-32B",
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",  # MoE, ~3B active; hybrid thinking toggle like 32B
    "qwen3-30b-a3b-instruct-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507",  # non-thinking only
    "qwen3-30b-a3b-thinking-2507": "Qwen/Qwen3-30B-A3B-Thinking-2507",  # thinking only
    "gpt-oss-20b": "openai/gpt-oss-20b",
}

# Offline Wikipedia ZIM mirrors (Kiwix), keyed by language.
_WIKI_ZIM_DIR = os.path.expandvars("$OpenLLM_OUTPUT/data/react_assets/wikipedia")
WIKI_ZIM_PATHS = {
    "en": os.path.join(_WIKI_ZIM_DIR, "wikipedia_en_all_nopic.zim"),
    "fr": os.path.join(_WIKI_ZIM_DIR, "wikipedia_fr_all_nopic_2026-05.zim"),
}

# Dense-retrieval indices built by build_wiki_index.py; used by the retriever set.
_WIKI_INDEX_DIR = os.path.expandvars("$OpenLLM_OUTPUT/data/react_assets")
WIKI_INDEX_PATHS = {
    "hotpotqa_en": os.path.join(
        _WIKI_INDEX_DIR, "beir_index"
    ),  # HotpotQA/BEIR corpus (English)
    "finewiki_en": os.path.join(
        _WIKI_INDEX_DIR, "finewiki_index", "en"
    ),  # FineWiki full-article chunks (English)
}

# Base policy, used as-is when thinking is enabled.
SYSTEM_PROMPT = (
    "You answer the user's question by interacting with Wikipedia through the "
    "provided tools. You MUST NOT rely on your own prior knowledge: treat it as "
    "unreliable. Every fact in your answer must come from evidence you "
    "retrieved with the tools in this conversation. If you have not seen it in "
    "a tool result, you do not know it -- search for it first.\n"
    "If a tool call returns an error (for example a message starting with "
    "'Error'), do not give up: read the error, correct your arguments or pick a "
    "different action, and retry."
)


# Appended when thinking is disabled: the <think> block otherwise covers this.
# Owns the reasoning-note rules, since it is the only prompt that defines the note.
REASONING_PROMPT = (
    "\nReason before you act. Before every tool call, write a brief note in "
    "your message that:\n"
    "- says what you learned from the previous result (if any), and\n"
    "- states what you will do next and why.\n"
    "Your reasoning note is never the final output: it only precedes the tool "
    "call.\n"
    "On your first turn, also break the question into the facts you need to "
    "find."
)

FRENCH_PROMPT = (
    "\nYou MUST write every message, and "
    "the answer passed to `submit_answer`, in French."
)

# Appended (retriever set only) under --inline_citations. Prompt-only: it leaves
# submit_answer alone, so it applies whether or not the dataset has a finish tool.
# ID is a <source> id from the results, e.g. A3.
INLINE_CITATION_PROMPT = (
    '\nCite your sources inline in your answer. Place a self-closing tag <ref name="ID"/> '
    "immediately after each statement a retrieved source supports, where ID is "
    "the id of the <source> it came from. Cite every claim that comes from a "
    "source; when several sources support one statement, add a <ref .../> for each."
)

# Drawn one per document so the model does not overfit a single phrasing.
FEVER_QUESTION_TEMPLATES = [
    'Is the claim "{claim}" supported by the evidence in Wikipedia?',
    "According to Wikipedia, is the following claim true? {claim}",
    "Using Wikipedia as evidence, verify the claim: {claim}",
    "Fact-check this claim against Wikipedia:\n{claim}",
    "Does the evidence in Wikipedia support or refute the claim: {claim}?",
    "Determine whether Wikipedia supports the claim:\n{claim}",
]

# Auto-open a disambiguation page's top option instead of listing the options.
# Must be passed to both preproc and the env factory: it drives the `search` tool
# description as well as the behaviour.
AUTO_DISAMBIGUATION = True


def call_tool(env, name, arguments):
    """Dispatch one tool call to its env, returning any failure as an observation
    string so the model can react to it instead of the episode aborting."""
    tool_names = getattr(env, "allowed_tool_names", None)
    if tool_names is not None and name not in tool_names:
        return (
            f"Error: unknown tool '{name}'. Available tools: {', '.join(tool_names)}."
        )
    if not isinstance(arguments, dict):
        return f"Error: arguments for '{name}' must be a JSON object, got {type(arguments).__name__}."
    try:
        return getattr(env, name)(**arguments)
    except TypeError as e:
        # Wrong or missing arguments for the tool signature. Strip the internal
        # "WikiEnv.search() " qualname prefix Python prepends, so the model only
        # ever sees the tool by its public name.
        msg = re.sub(r"^[\w.]+\(\) ", "", str(e))
        return f"Error calling '{name}': {msg}"
    except Exception as e:
        return f"Error calling '{name}': {type(e).__name__}: {e}"


def new_env(
    doc,
    backend="online",
    zim_path=None,
    accept_threshold=None,
    auto_disambiguation=False,
    scorer=None,
):
    """Create a fresh Wikipedia browser for one conversation. Search loads the page
    a later lookup reads, so env state must not be shared across conversations."""
    env = WikiEnv(
        ground_truth=doc.metadata["answer"],
        backend=backend,
        zim_path=zim_path,
        accept_threshold=accept_threshold,
        auto_disambiguation=auto_disambiguation,
        scorer=scorer,
    )
    # Same list object, so later submit_answer scores land in the output metadata.
    doc.metadata["submit_answer_scores"] = env.scores
    return env


def new_env_for_doc(
    doc,
    backend="online",
    zim_path=None,
    index_dir=None,
    accept_threshold=None,
    auto_disambiguation=False,
    scorer=None,
):
    """Build the env matching this document's drawn tool_type."""
    tool_type = doc.metadata["tool_type"]
    if tool_type == "wiki_api":
        env = new_env(
            doc,
            backend=backend,
            zim_path=zim_path,
            accept_threshold=accept_threshold,
            auto_disambiguation=auto_disambiguation,
            scorer=scorer,
        )
    elif tool_type == "wiki_structured":
        env = new_structured_env(
            doc,
            backend=backend,
            zim_path=zim_path,
            accept_threshold=accept_threshold,
            auto_disambiguation=auto_disambiguation,
            scorer=scorer,
        )
    elif tool_type == "retriever":
        env = new_retriever_env(
            doc, index_dir=index_dir, accept_threshold=accept_threshold, scorer=scorer
        )
    else:
        raise ValueError(f"Unknown tool_type: {tool_type!r}")
    # From the tools actually offered, not TOOL_SETS: build_tool_sets may drop
    # submit_answer for the dataset.
    env.allowed_tool_names = [t["function"]["name"] for t in doc.metadata["tools"]]
    return env


def generation_config(model, temperature=None, enable_thinking=False, max_tokens=None):
    if "gpt-oss" in MODELS[model]:
        # gpt-oss always reasons: depth is reasoning_effort, not enable_thinking.
        if not temperature:
            temperature = 1.0
        if not max_tokens:
            max_tokens = 8192
        return {
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "temperature": temperature,
            "top_p": 1.0,
        }
    # https://huggingface.co/Qwen/Qwen3-1.7B#best-practices
    # https://huggingface.co/Qwen/Qwen3-32B#best-practices
    if enable_thinking:
        if not temperature:
            temperature = 0.6
        if not max_tokens:
            # Caps a SINGLE turn, so it must stay well below model_max_context
            # (32768) -- the conversation grows each turn, and a full-window
            # value leaves no room for the prompt: every request 400s.
            max_tokens = 8192
        return {
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": True},
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
        }
    else:
        if not temperature:
            temperature = 0.7
        if not max_tokens:
            max_tokens = 2048
        return {
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": temperature,
            "top_p": 0.80,
            "top_k": 20,
            "min_p": 0.0,
        }


def agent_query_builder(runner, doc, gen_config):
    # Initial turn only: the runner reuses tools + sampling on every later turn.
    return {
        "messages": doc.metadata["messages"],
        "tools": doc.metadata["tools"],
        **gen_config,
    }


def _build_system_prompt(enable_thinking, lang, cite_inline=False):
    system_prompt = SYSTEM_PROMPT
    if not enable_thinking:
        system_prompt += REASONING_PROMPT
    if cite_inline:
        system_prompt += INLINE_CITATION_PROMPT
    # The translated queries do not pin the answer language, so force it.
    if lang == "fr":
        system_prompt += FRENCH_PROMPT
    return system_prompt


def preproc(
    data,
    rank: int = 0,
    world_size: int = 1,
    enable_thinking: bool = False,
    lang: str = "en",
    tool_types: list = None,
    auto_disambiguation: bool = False,
    dataset_name: str = None,
    inline_citations: bool = False,
):
    # tool_type is drawn per document, not per run, so a run mixes the tool sets.
    # Seeded from rank: reproducible per task, different across shards.
    tool_sets = build_tool_sets(auto_disambiguation, dataset_name)
    tool_types = list(tool_types) if tool_types else list(tool_sets.keys())
    # Inline citations are a retriever-set concept (only it emits <source> ids).
    system_prompts = {
        tt: _build_system_prompt(
            enable_thinking, lang, cite_inline=(inline_citations and tt == "retriever")
        )
        for tt in tool_types
    }
    rng = random.Random(rank)
    for doc in data:
        tool_type = rng.choice(tool_types)
        tools = tool_sets[tool_type]
        question = doc.metadata.pop("question")
        answer = doc.metadata.pop("answer", None)
        messages = [
            {"role": "system", "content": system_prompts[tool_type]},
            {"role": "user", "content": question},
        ]
        doc = Document(
            id=doc.id,
            text="<empty>",
            metadata={
                "question": question,
                "answer": answer,
                "messages": messages,
                "tools": tools,
                "tool_type": tool_type,
                "dataset_name": dataset_name,
                "language": lang,
                "enable_thinking": enable_thinking,
            },
        )
        yield doc


def postprocess_fn(self, doc, model, gen_config):
    doc.metadata["generation_config"] = {
        "model_name_or_path": MODELS[model],
        **gen_config,
    }
    # Absent when the agent never submitted, but the parquet schema needs the key.
    doc.metadata.setdefault("submit_answer_scores", [])
    return doc


# Pinned because schema=None locks the file to its FIRST document and rejects any
# later batch with different columns ("Table schema does not match"), which older
# `formatted/data` shards do have. Coercing to this makes absent keys null.
REACT_PARQUET_SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("id", pa.string()),
        ("question", pa.string()),
        ("answer", pa.string()),
        (
            "messages",
            pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())])),
        ),
        ("tools", pa.string()),
        ("tool_type", pa.string()),
        ("dataset_name", pa.string()),
        ("language", pa.string()),
        ("enable_thinking", pa.bool_()),
        ("num_turns", pa.int64()),
        ("inference_results", pa.string()),
        (
            "generation_config",
            pa.struct(
                [
                    ("model_name_or_path", pa.string()),
                    ("max_tokens", pa.int64()),
                    # Mutually exclusive per model family; the unused one is null.
                    (
                        "chat_template_kwargs",
                        pa.struct(
                            [
                                ("enable_thinking", pa.bool_()),
                                ("reasoning_effort", pa.string()),
                            ]
                        ),
                    ),
                    ("temperature", pa.float64()),
                    ("top_p", pa.float64()),
                    ("top_k", pa.int64()),
                    ("min_p", pa.float64()),
                ]
            ),
        ),
        ("submit_answer_scores", pa.list_(pa.float64())),
        ("file_path", pa.string()),
    ]
)


def stabilize_for_parquet(data, rank: int = 0, world_size: int = 1):
    """Fix the two columns whose arrow type drifts across shards, which breaks
    load_dataset()'s schema unification:
      - `id`: int64 for FEVER, string elsewhere -> str everywhere.
      - `inference_results`: message.{content,reasoning} is all-None (null type) in
        one thinking mode, string in the other ("cast string to null") -> JSON.
    """
    for doc in data:
        doc.id = str(doc.id)
        ir = doc.metadata.get("inference_results")
        if ir is not None and not isinstance(ir, str):
            doc.metadata["inference_results"] = json.dumps(ir, default=str)
        yield doc


def filter_function(doc):
    answer_scores = doc.metadata.get("submit_answer_scores", [])
    if doc.metadata["answer"] is not None:
        if not answer_scores:
            return False, "no_answer"
        if answer_scores[-1] == 0.0:
            return False, "answer_incorrect"
        if answer_scores[-1] < 0.5:
            return False, "answer_low_confidence"
    if doc.metadata.get("num_turns", 0) <= 1:
        return False, "too_few_turns"
    return True


def create_reader(
    dataset_name, language, input_path=None, glob_pattern="**/*.jsonl.gz"
):
    if input_path is not None:
        reader = [
            JsonlReader(
                input_path, glob_pattern=glob_pattern, adapter=instruct_adapter
            ),
        ]
        return reader
    if dataset_name == "hotpotqa":
        if language == "en":
            reader = [
                HuggingFaceDatasetReader(
                    "hotpotqa/hotpot_qa",
                    {"split": "train", "name": "fullwiki"},
                    streaming=False,
                    adapter=instruct_adapter,
                )
            ]
        elif language == "fr":

            def get_question(data, rank: int = 0, world_size: int = 1):
                for doc in data:
                    doc.metadata["question"] = doc.metadata.pop("query")
                    yield doc

            reader = [
                HuggingFaceDatasetReader(
                    "Mvanypersele/luciole_RAG",
                    {"name": "hotpotqa_fr", "split": "train"},
                    streaming=False,
                    adapter=instruct_adapter,
                ),
                get_question,
            ]
        else:
            raise ValueError(f"Unsupported language: {language}")
    elif dataset_name == "multihopqa":
        reader = [
            HuggingFaceDatasetReader(
                "xanhho/2WikiMultihopQA",
                {
                    "split": "train",
                    "revision": "e37a4050605363be62f1d02e6eb888fe5f56530e",
                },
                streaming=False,
                adapter=instruct_adapter,
            ),
        ]
    elif dataset_name == "fever":

        def get_question(data, rank: int = 0, world_size: int = 1):
            # Draw a random phrasing per doc (rank-seeded for reproducibility).
            rng = random.Random(rank)
            for doc in data:
                template = rng.choice(FEVER_QUESTION_TEMPLATES)
                doc.metadata["question"] = template.format(
                    claim=doc.metadata.get("claim")
                )
                label = doc.metadata.pop("label")
                if label == "NOT ENOUGH INFO":
                    continue  # Skip unanswerable docs: the model cannot be graded on them.
                doc.metadata["answer"] = label.lower()
                yield doc

        reader = [
            HuggingFaceDatasetReader(
                "fever/fever",
                {
                    "split": "train",
                    "revision": "5f577157472532aa1d9924d2df63aac44f70cf2b",
                },
                streaming=False,
                adapter=instruct_adapter,
            ),
            get_question,
        ]
    elif dataset_name == "pleais_rag":

        def get_question(data, rank: int = 0, world_size: int = 1):
            for doc in data:
                doc.metadata["question"] = doc.metadata["messages"][1]["content"]
                doc.metadata["answer"] = None
                yield doc

        reader = [
            HuggingFaceDatasetReader(
                "OpenLLM-France/PleAIs_RAG",
                {"split": "train"},
                streaming=False,
                adapter=instruct_adapter,
            ),
            get_question,
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return reader


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument(
        "--glob_pattern",
        type=str,
        default="**/*.jsonl.gz",
        help="Glob pattern to match input files under --input_path",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=None,
        help="Sample rate for sampling input data (between 0.0 and 1.0)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Temperature for sampling"
    )
    parser.add_argument(
        "--enable_thinking", action="store_true", help="Enable thinking."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Max tokens to generate per turn (defaults: 8192 thinking and gpt-oss, "
        "2048 non-thinking)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-32b",
        choices=MODELS.keys(),
        help="Model to use for generation",
    )
    parser.add_argument(
        "--tp", type=int, default=1, help="Tensor-parallel size (GPUs per node)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="offline",
        choices=["online", "offline"],
        help="Wikipedia source: 'offline' (local ZIM mirror, default) or 'online' (live HTTP)",
    )
    parser.add_argument(
        "--main_name",
        type=str,
        default="react",
        help="Main output directory name for this run (default: 'react')",
    )
    parser.add_argument("--dataset_name", type=str, default="hotpotqa")
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Path to input data (continuing on the failed shards of a previous run)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=WIKI_ZIM_PATHS.keys(),
        help="Wikipedia language for the offline backend; selects the ZIM mirror "
        "from WIKI_ZIM_PATHS",
    )
    parser.add_argument(
        "--retriever_corpus",
        type=str,
        default="finewiki",
        choices=["hotpotqa", "finewiki"],
        help="Which dense-retrieval corpus the 'retriever' tool set queries; combined "
        "with --lang to key WIKI_INDEX_PATHS (e.g. 'finewiki' + 'en' -> finewiki_en)",
    )
    parser.add_argument(
        "--accept_threshold",
        type=float,
        default=-1,
        help="F1 the answer must strictly exceed for submit_answer to accept it and "
        "end the episode (0 = any token overlap; below it the model is asked to "
        "reformulate; negative = accept any answer on first submit)",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=6,
        help="Maximum assistant turns per question before the agent loop is cut off",
    )
    parser.add_argument(
        "--tool_types",
        nargs="+",
        default=None,
        choices=list(TOOL_SETS.keys()),
        help="Tool sets a run draws from, one per document (default: all). Restrict "
        "to isolate or compare browsing styles, e.g. '--tool_types wiki_api "
        "wiki_structured' or '--tool_types wiki_structured'",
    )
    parser.add_argument(
        "--inline_citations",
        action="store_true",
        help="Retriever set: instruct the model to cite its sources inline with "
        '<ref name="ID"/> markup (ID = a <source> id). Prompt-only; the tool '
        "schemas are unchanged. No effect on the other tool sets.",
    )
    parser.add_argument(
        "--format_only", action="store_true", help="Run only formatting."
    )
    args = parse_args(parser)
    DATA_PATH = args.data_path

    # The 2507 splits dropped the hybrid thinking toggle: Instruct is
    # non-thinking only, Thinking always reasons. Enforce that --enable_thinking
    # matches the chosen model so we don't silently run the wrong mode.
    model_path = MODELS[args.model]
    if model_path.endswith("Instruct-2507"):
        assert (
            not args.enable_thinking
        ), f"{model_path} is non-thinking only; drop --enable_thinking"
    elif model_path.endswith("Thinking-2507"):
        assert (
            args.enable_thinking
        ), f"{model_path} is thinking only; pass --enable_thinking"

    dataset_name = args.dataset_name
    # Only a fresh run needs a known HF source; a continuation reads --input_path
    # and just carries the name through, so anything goes there.
    KNOWN_DATASETS = ("hotpotqa", "multihopqa", "fever", "pleais_rag")
    if args.input_path is None and dataset_name not in KNOWN_DATASETS:
        parser.error(
            f"--dataset_name must be one of {KNOWN_DATASETS} (got {dataset_name!r})"
        )
    thinking_dir = "thinking" if args.enable_thinking else "non_thinking"
    OUT_NAME = f"{args.main_name}/{dataset_name}/{thinking_dir}/{args.lang}"

    # Push the already-formatted data and exit.
    if args.push_only:
        push_pipeline = [
            JsonlReader(
                f"{DATA_PATH}/{OUT_NAME}/formatted/data",
            ),
            stabilize_for_parquet,
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-France/ReAct" + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{DATA_PATH}/{OUT_NAME}/data_hf",
                output_filename=f"data/{dataset_name}/"
                + "${tool_type}/"
                + f"{thinking_dir}/{args.lang}/"
                + "${rank}.parquet",
                cleanup=True,
                expand_metadata=True,
                schema=REACT_PARQUET_SCHEMA,
            ),
        ]
        create_executor(
            push_pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/{OUT_NAME}/logs_hf",
            job_name="react_hf",
            tasks=1,
            skip_completed=not args.force,
        ).run()
        raise SystemExit(0)

    # Fail fast if an asset is missing, rather than deep inside an episode. Only
    # assert what the selected --tool_types require.
    active_tool_types = args.tool_types or list(TOOL_SETS.keys())
    zim_path = WIKI_ZIM_PATHS[args.lang]
    index_key = f"{args.retriever_corpus}_{args.lang}"
    index_dir = WIKI_INDEX_PATHS.get(index_key)
    needs_zim = bool({"wiki_api", "wiki_structured"} & set(active_tool_types))
    needs_index = "retriever" in active_tool_types
    if needs_zim and args.backend == "offline":
        assert os.path.exists(
            zim_path
        ), f"ZIM mirror not found for lang '{args.lang}': {zim_path}"
    if needs_index:
        assert index_dir, f"No retriever index configured for '{index_key}'; known: {list(WIKI_INDEX_PATHS)}"
        assert os.path.isdir(
            index_dir
        ), f"Retriever index '{index_key}' not found on disk: {index_dir}"
    env_factory = partial(
        new_env_for_doc,
        backend=args.backend,
        zim_path=zim_path,
        index_dir=index_dir,
        accept_threshold=args.accept_threshold,
        auto_disambiguation=AUTO_DISAMBIGUATION,
        scorer=SCORERS.get(dataset_name),
    )

    # Structured tool calling: vLLM parses tool calls into message.tool_calls.
    # Parser choice is model-family specific.
    model_kwargs = {"enable-auto-tool-choice": True}
    if "gpt-oss" in MODELS[args.model]:
        # Harmony format. Always reasons, so the reasoning parser is unconditional.
        model_kwargs["tool-call-parser"] = "openai"
        model_kwargs["reasoning-parser"] = "openai_gptoss"
    else:
        model_kwargs["tool-call-parser"] = "hermes"
        if args.enable_thinking:
            # Splits <think> into message.reasoning_content, which the Qwen3
            # template needs to preserve it across the rollout.
            model_kwargs["reasoning-parser"] = "qwen3"

    config: InferenceConfig = InferenceConfig(
        server_type="vllm",
        model_name_or_path=MODELS[args.model],
        tp=args.tp,
        model_max_context=32768,
        max_concurrent_requests=500,
        max_concurrent_tasks=500,
        metric_interval=120,
        model_kwargs=model_kwargs,
    )

    #########
    # Generation
    #########
    reader = create_reader(
        dataset_name,
        args.lang,
        input_path=args.input_path,
        glob_pattern=args.glob_pattern,
    )
    gen_config = generation_config(
        model=args.model,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_tokens,
    )

    pipeline = reader + [
        partial(
            preproc,
            enable_thinking=args.enable_thinking,
            lang=args.lang,
            tool_types=args.tool_types,
            auto_disambiguation=AUTO_DISAMBIGUATION,
            dataset_name=dataset_name,
            inline_citations=args.inline_citations,
        ),
        ToolCallingInferenceRunner(
            query_builder=partial(agent_query_builder, gen_config=gen_config),
            config=config,
            env_factory=env_factory,
            tool_executor=call_tool,
            finish_tool="submit_answer",
            max_turns=args.max_turns,
            records_per_chunk=500,
            checkpoints_local_dir=f"{DATA_PATH}/{OUT_NAME}/checkpoints",
            output_writer=JsonlWriter(
                f"{DATA_PATH}/{OUT_NAME}/data",
                output_filename="${tool_type}/${rank}_chunk_${chunk_index}.jsonl",
            ),
            postprocess_fn=partial(
                postprocess_fn, model=args.model, gen_config=gen_config
            ),
            skip_bad_requests=True,
        ),
    ]

    if args.sample_rate is not None:
        pipeline.insert(
            1,
            SamplerFilter(
                rate=args.sample_rate,
                seed=42,
                exclusion_writer=JsonlWriter(
                    f"{DATA_PATH}/{OUT_NAME}/sampler_filtered",
                ),
            ),
        )

    executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        limit_debug=args.limit_debug,
        logging_dir=f"{DATA_PATH}/{OUT_NAME}/logs",
        job_name="react_gen",
        tasks=1,
        time="20:00:00",
        qos="qos_gpu_h100-t3",
        partition="gpu_p6",
        cpus_per_task=32,
        env_command=f"source {_DATA_DIR}/set_env_inference.sh",
        sbatch_args={
            "account": "wuh@h100",
            "constraint": "h100",
            "gres": f"gpu:{args.tp}",
            "nodes": 1,
            "hint": "nomultithread",
        },
        skip_completed=not args.force,
    )

    #########
    # Format the generated conversations into templated SFT text. Runs on CPU
    # and depends on the inference executor above.
    #########

    tokenizer = AutoTokenizer.from_pretrained(
        "OpenLLM-France/tokenizer_128k-arab-regional_v2_instruct_train"
    )

    format_pipeline = [
        JsonlReader(
            f"{DATA_PATH}/{OUT_NAME}/data",
            adapter=instruct_adapter,
        ),
        partial(add_system_prompt, tokenizer=tokenizer),
        NemoRLFormat(),
        partial(apply_chat_template, tokenizer=tokenizer),
        LambdaFilter(
            filter_function=filter_function,
            exclusion_writer=JsonlWriter(
                f"{DATA_PATH}/{OUT_NAME}/formatted/filtered",
                output_filename="${filter_reason}/${rank}.jsonl",
            ),
        ),
        JsonlWriter(
            f"{DATA_PATH}/{OUT_NAME}/formatted/data",
            output_filename="${tool_type}/${rank}.jsonl",
            expand_metadata=True,
        ),
    ]

    format_executor = create_executor(
        format_pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/{OUT_NAME}/formatted/logs",
        job_name="react_format",
        partition="cpu_p1",
        tasks=1,
        time="01:00:00",
        env_command=f"source {_DATA_DIR}/set_env.sh",
        skip_completed=not args.force,
        depends=executor if not args.format_only else None,
    )

    format_executor.run()
