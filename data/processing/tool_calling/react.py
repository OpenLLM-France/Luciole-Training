# Script that input instruct data and output preference data for posttraining

import os

# The retriever tool set embeds each query with a CPU SentenceTransformer, and up
# to max_concurrent_tasks (~500) of those run at once in datatrove's threadpool.
# Left multi-threaded, each encode asks OpenBLAS/OMP for many threads; the total
# blows past OpenBLAS's per-thread buffer pool and it aborts with
# "BLAS : Program is Terminated. Because you tried to allocate too many memory
# regions." One BLAS thread per encode keeps the buffer count bounded (query
# encoding is a single short string, so per-call speed is irrelevant). Must be set
# before numpy/torch are imported (transitively via utils/datatrove below).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import pathlib
import random
import re
from utils import create_parser, parse_args, create_executor
from datatrove.data import Document
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.writers import JsonlWriter, HuggingFaceDatasetWriter
from datatrove.pipeline.inference.run_inference import InferenceConfig
from datatrove.pipeline.inference.tool_calling import ToolCallingInferenceRunner
from datatrove.pipeline.filters import LambdaFilter
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

#_DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

# Offline Wikipedia ZIM mirrors (Kiwix), keyed by language.
_WIKI_ZIM_DIR = os.path.expandvars("$OpenLLM_OUTPUT/data/react_assets/wikipedia")
WIKI_ZIM_PATHS = {
    "en": os.path.join(_WIKI_ZIM_DIR, "wikipedia_en_all_nopic.zim"),
    "fr": os.path.join(_WIKI_ZIM_DIR, "wikipedia_fr_all_nopic_2026-05.zim"),
}

# Dense-retrieval index directories (built by build_wiki_index.py), keyed by
# language. Each holds passages.jsonl / embeddings.f16.npy / meta.json and an
# optional index.faiss. Used by the 'retriever' tool set.
_WIKI_INDEX_DIR = os.path.expandvars("$OpenLLM_OUTPUT/data/react_assets")
WIKI_INDEX_PATHS = {
    "en": os.path.join(_WIKI_INDEX_DIR, "beir_index"),      # HotpotQA corpus (English)
    "fr": os.path.join(_WIKI_INDEX_DIR, "zim_index", "fr"),  # French ZIM lead paragraphs
}

# Base policy, used as-is when thinking is enabled: the model already reasons
# in its <think> block, so no explicit "reason before you act" instruction is
# needed here (see REASONING_PROMPT for the non-thinking addition).
SYSTEM_PROMPT = (
    "You answer the user's question by interacting with Wikipedia through the "
    "provided tools. You MUST NOT rely on your own prior knowledge: treat it as "
    "unreliable. Every fact in your answer must come from evidence you "
    "retrieved with the tools in this conversation. If you have not seen it in "
    "a tool result, you do not know it -- search for it first.\n"
    "You MUST always end your message with a tool call. Your reasoning note is "
    "never the final output: it only precedes the tool call. Every turn must "
    "include at least one tool call ({tools}) -- "
    "never reply with text alone.\n"
    "If a tool call returns an error (for example a message starting with "
    "'Error'), do not give up: read the error, correct your arguments or pick a "
    "different action, and retry.")

# Appended when thinking is disabled, so the model reasons explicitly in its
# message content before each tool call (in thinking mode the <think> block
# already covers this).
REASONING_PROMPT = (
    "\nReason before you act. Before every tool call, write a brief note in "
    "your message that:\n"
    "- says what you learned from the previous result (if any), and\n"
    "- states what you will do next and why.\n"
    "On your first turn, also break the question into the facts you need to "
    "find.")

FRENCH_PROMPT = (
    "\nYou MUST write every message, and "
    "the answer passed to `submit_answer`, in French."
)

# Paraphrases of the FEVER fact-checking question. One is drawn per document so
# the model does not overfit to a single phrasing; each must contain "{claim}".
FEVER_QUESTION_TEMPLATES = [
    'Is the claim "{claim}" supported by the evidence in Wikipedia?',
    'According to Wikipedia, is the following claim true? "{claim}"',
    'Using Wikipedia as evidence, verify the claim: "{claim}".',
    'Fact-check this claim against Wikipedia: "{claim}".',
    'Does the evidence in Wikipedia support or refute the claim "{claim}"?',
    'Determine whether Wikipedia supports the claim "{claim}".',
]

# On a Wikipedia disambiguation page, automatically open its top option (and name
# the alternatives) instead of returning the option list for the model to choose
# from. Applies to the wiki_api and wiki_structured tool sets; it drives both the
# env behaviour and the flag-aware `search` tool description (build_tool_sets).
AUTO_DISAMBIGUATION = True

def call_tool(env, name, arguments, tool_names=None):
    """Dispatch a single tool call to its WikiEnv implementation.

    `env` is a per-conversation WikiEnv instance (search loads the page that a
    subsequent lookup reads, so state must not be shared across conversations).

    Always returns an observation string. Any failure -- unknown tool, bad
    arguments, or an error raised by the tool itself (e.g. a network error while
    searching Wikipedia) -- is caught and returned as an error message, so the
    model can react to it and the whole episode is not aborted.

    `tool_names` is the set of tool names allowed for the active tool set (see
    TOOL_SETS); an unrecognised call is turned into an error observation. When
    None, it falls back to the env's own `allowed_tool_names` (set per document
    by new_env_for_doc, since each doc draws its own tool_type); if the env has
    none either, the name check is skipped (any method on the env may be called).
    """
    if tool_names is None:
        tool_names = getattr(env, "allowed_tool_names", None)
    if tool_names is not None and name not in tool_names:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(tool_names)}."
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


def new_env(doc, backend="online", zim_path=None, accept_threshold=None,
            auto_disambiguation=False, scorer=None):
    """Create a fresh, isolated Wikipedia browser for one conversation.

    The document's ground-truth answer is passed in so that `submit_answer`
    can grade the agent's final answer with exact match. `backend`/`zim_path`
    select live Wikipedia vs. a local ZIM snapshot (see WikiEnv);
    `accept_threshold` is the F1 an answer must strictly exceed to be accepted
    (negative = accept any answer on the first submit); `auto_disambiguation`
    auto-opens the top option of a disambiguation page instead of asking;
    `scorer` overrides the default token-F1 grading (e.g. FEVER exact match).
    """
    env = WikiEnv(ground_truth=doc.metadata["answer"], backend=backend, zim_path=zim_path,
                  accept_threshold=accept_threshold, auto_disambiguation=auto_disambiguation,
                  scorer=scorer)
    # Alias the env's score history into the doc so each submit_answer score
    # lands in the output metadata (same list object -> appends persist).
    doc.metadata["submit_answer_scores"] = env.scores
    return env


def new_env_for_doc(doc, backend="online", zim_path=None, index_dir=None,
                    accept_threshold=None, auto_disambiguation=False, scorer=None):
    """Build the env matching this document's drawn tool_type.

    tool_type is chosen per document in preproc (not a global run setting), so
    the env factory has to dispatch on `doc.metadata["tool_type"]`: 'wiki_api'
    browses the ZIM/HTTP Wikipedia, 'retriever' searches the dense index. The
    allowed tool names for the chosen set are stashed on the env so call_tool
    can validate calls without a globally-fixed tool set. `scorer` (a run-level
    grading function, e.g. FEVER exact match) overrides the default token-F1.
    """
    tool_type = doc.metadata["tool_type"]
    if tool_type == "wiki_api":
        env = new_env(doc, backend=backend, zim_path=zim_path,
                      accept_threshold=accept_threshold,
                      auto_disambiguation=auto_disambiguation, scorer=scorer)
    elif tool_type == "wiki_structured":
        env = new_structured_env(doc, backend=backend, zim_path=zim_path,
                                 accept_threshold=accept_threshold,
                                 auto_disambiguation=auto_disambiguation, scorer=scorer)
    elif tool_type == "retriever":
        env = new_retriever_env(doc, index_dir=index_dir,
                                accept_threshold=accept_threshold, scorer=scorer)
    else:
        raise ValueError(f"Unknown tool_type: {tool_type!r}")
    env.allowed_tool_names = [t["function"]["name"] for t in TOOL_SETS[tool_type]]
    return env


def generation_config(temperature=None, enable_thinking=False, max_tokens=None):
    # https://huggingface.co/Qwen/Qwen3-1.7B#best-practices
    # https://huggingface.co/Qwen/Qwen3-32B#best-practices
    if enable_thinking:
        if not temperature:
            temperature = 0.6
        if not max_tokens:
            # Per-turn output budget. Must stay well below model_max_context
            # (32768): it is the cap on a SINGLE turn's generation, and the
            # conversation grows each turn, so requesting the whole window
            # (Qwen3's 32768 single-response recommendation) leaves no room for
            # the prompt and every request 400s.
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


def agent_query_builder(runner, doc, temperature=None, enable_thinking=False, max_tokens=None):
    # Initial turn for the agent loop: raw messages + structured tools are sent
    # to /v1/chat/completions; the server applies the chat template and parses
    # tool calls. The runner reuses tools + sampling on every subsequent turn.
    return {
        "messages": doc.metadata["messages"],
        "tools": doc.metadata["tools"],
        **generation_config(temperature=temperature, enable_thinking=enable_thinking, max_tokens=max_tokens),
    }


def _build_system_prompt(tools, enable_thinking, lang):
    # Enumerate the active tool names into the prompt so the "every turn must
    # call a tool" rule lists the tools actually available (search/lookup vs.
    # wikipedia_retriever/next_results), not a hard-coded set.
    tools_clause = ", ".join(f"`{t['function']['name']}`" for t in tools)
    # In thinking mode the model reasons in its <think> block, so use the
    # variant without the explicit "reason before you act" instructions.
    system_prompt = SYSTEM_PROMPT.format(tools=tools_clause)
    if not enable_thinking:
        system_prompt += REASONING_PROMPT
    # The French HotpotQA queries are translated but the answer language is not
    # pinned by the data, so force French output explicitly.
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
):
    # tool_type is drawn independently per document (not a global run setting):
    # a single run interleaves the active tool sets so the model sees them mixed.
    # The system prompt and tool schemas depend on the tool set, so precompute one
    # prompt per tool_type and select per doc. Seed the RNG from rank so each
    # task is reproducible and the draws differ across shards. `tool_types`
    # restricts which sets a run draws from (default: all of TOOL_SETS), so a run
    # can isolate or compare browsing styles (e.g. wiki_api vs wiki_structured).
    # The `search` descriptions depend on auto_disambiguation, so resolve the tool
    # schemas for the run's mode (build_tool_sets); it must match the env factory's
    # auto_disambiguation, which drives the actual behaviour.
    tool_sets = build_tool_sets(auto_disambiguation, dataset_name)
    tool_types = list(tool_types) if tool_types else list(tool_sets.keys())
    system_prompts = {
        tt: _build_system_prompt(tool_sets[tt], enable_thinking, lang)
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


def postprocess_fn(self, doc, model_size, gen_config):
    doc.metadata[f"inference_results"] = doc.metadata.pop("inference_results")
    doc.metadata[f"generation_config"] = {"model_name_or_path": MODEL_SIZES[model_size], **gen_config}
    # Scores of every submit_answer call this run (last element = final score).
    doc.metadata[f"submit_answer_scores"] = doc.metadata.pop("submit_answer_scores", [])
    return doc

def stabilize_for_parquet(data, rank: int = 0, world_size: int = 1):
    """Coerce the two columns whose type is unstable across shards to a fixed type.

    load_dataset() unifies the parquet schema of every shard, so a column must
    have the same arrow type in all of them. Two do not, and break the load:
      - `id`: int64 for FEVER (integer source ids), string for the QA datasets
        -> force str everywhere.
      - `inference_results[].message.{content,reasoning}`: string in one thinking
        mode and all-None (-> arrow null type) in the other, so unifying a
        thinking and a non_thinking shard raises "cast string to null" -> serialize
        the whole structure to a JSON string (the same treatment `tools` gets).
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
        if not answer_scores:  # Check if the last score is 0.0 or if the list is empty
            return False, "no_answer"
        if answer_scores[-1] == 0.0:
            return False, "answer_incorrect"
        if answer_scores[-1] < 0.5:
            return False, "answer_low_confidence"
    if doc.metadata.get("num_turns", 0) <= 1:
        return False, "too_few_turns"
    return True

def create_reader(dataset_name, language):
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
                {"split": "train", "revision": "e37a4050605363be62f1d02e6eb888fe5f56530e"},
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
                doc.metadata["question"] = template.format(claim=doc.metadata.get("claim"))
                label = doc.metadata.pop("label")
                if label == "NOT ENOUGH INFO":
                    continue  # Skip unanswerable docs: the model cannot be graded on them.
                doc.metadata["answer"] = label.lower()
                yield doc
        reader = [
            HuggingFaceDatasetReader(
                "fever/fever",
                {"split": "train", "revision": "5f577157472532aa1d9924d2df63aac44f70cf2b"},
                streaming=False,
                adapter=instruct_adapter,
            ),
            get_question,
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return reader
    
MODEL_SIZES = {
    "0.6b": "Qwen/Qwen3-0.6B", # For testing only, not recommended for generation
    "32b": "Qwen/Qwen3-32B",
}

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--glob_pattern", type=str, default=None, help="Glob pattern to match input files")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for sampling")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking.")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max tokens to generate (defaults: 32768 thinking, 2048 non-thinking)")
    parser.add_argument("--size", type=str, default="32b", choices=MODEL_SIZES.keys(), help="Size of the chosen model")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--backend", type=str, default="offline", choices=["online", "offline"],
                        help="Wikipedia source: 'offline' (local ZIM mirror, default) or 'online' (live HTTP)")
    parser.add_argument("--main_name", type=str, default="react", help="Main output directory name for this run (default: 'react')")
    parser.add_argument("--dataset_name", type=str, default="hotpotqa", choices=["hotpotqa", "multihopqa", "fever"])
    parser.add_argument("--lang", type=str, default="en", choices=WIKI_ZIM_PATHS.keys(),
                        help="Wikipedia language for the offline backend; selects the ZIM mirror "
                             "from WIKI_ZIM_PATHS")
    parser.add_argument("--accept_threshold", type=float, default=-1,
                        help="F1 the answer must strictly exceed for submit_answer to accept it and "
                             "end the episode (0 = any token overlap; below it the model is asked to "
                             "reformulate; negative = accept any answer on first submit)")
    parser.add_argument("--max_turns", type=int, default=6,
                        help="Maximum assistant turns per question before the agent loop is cut off")
    parser.add_argument("--tool_types", nargs="+", default=None, choices=list(TOOL_SETS.keys()),
                        help="Tool sets a run draws from, one per document (default: all). Restrict "
                             "to isolate or compare browsing styles, e.g. '--tool_types wiki_api "
                             "wiki_structured' or '--tool_types wiki_structured'")
    parser.add_argument("--format_only", action="store_true", help="Run only formatting.")
    args = parse_args(parser)
    DATA_PATH = args.data_path

    # Output tree: dataset / thinking mode / language. tool_type is no longer a
    # path axis here because a single run mixes both (drawn per doc in preproc);
    # it is recorded per document in metadata and used to partition only the
    # final HF parquet layout (see ${tool_type} in the HF writer below).
    dataset_name = args.dataset_name
    thinking_dir = "thinking" if args.enable_thinking else "non_thinking"
    OUT_NAME = f"{args.main_name}/{dataset_name}/{thinking_dir}/{args.lang}"

    # --push_only: push the already-formatted data to the hub and exit. Nothing
    # else is needed -- no reader, inference config, tokenizer, or format
    # pipeline -- so build just the push executor here and stop.
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
                output_filename=f"data/{dataset_name}/" + "${tool_type}/" + f"{thinking_dir}/{args.lang}/" + "${rank}.parquet",
                cleanup=True,
                expand_metadata=True,
                schema=None,
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

    # The active tool sets are drawn per doc in preproc, so the env factory
    # dispatches per document; they share submit_answer and the F1 grading and
    # differ only in how the agent reaches Wikipedia (title browse / section
    # browse / dense retrieval). Fail fast here if a backend asset needed by an
    # active tool set is missing for this language, rather than deep inside an
    # episode. wiki_api and wiki_structured both browse the ZIM; retriever needs
    # the dense index -- so only assert what the selected --tool_types require.
    active_tool_types = args.tool_types or list(TOOL_SETS.keys())
    zim_path = WIKI_ZIM_PATHS[args.lang]
    index_dir = WIKI_INDEX_PATHS[args.lang]
    needs_zim = bool({"wiki_api", "wiki_structured"} & set(active_tool_types))
    needs_index = "retriever" in active_tool_types
    if needs_zim and args.backend == "offline":
        assert os.path.exists(zim_path), f"ZIM mirror not found for lang '{args.lang}': {zim_path}"
    if needs_index:
        assert os.path.isdir(index_dir), f"Retriever index not found for lang '{args.lang}': {index_dir}"
    env_factory = partial(new_env_for_doc, backend=args.backend, zim_path=zim_path,
                          index_dir=index_dir, accept_threshold=args.accept_threshold,
                          auto_disambiguation=AUTO_DISAMBIGUATION,
                          scorer=SCORERS.get(dataset_name))

    # Structured tool calling: vLLM parses tool calls into message.tool_calls.
    model_kwargs = {
        "enable-auto-tool-choice": True,
        "tool-call-parser": "hermes",  # Qwen3 uses the hermes-style parser
    }
    if args.enable_thinking:
        # Split the <think> block into message.reasoning_content so the Qwen3
        # chat template preserves it across the tool-calling rollout instead of
        # leaving it inline in content.
        model_kwargs["reasoning-parser"] = "qwen3"

    config: InferenceConfig = InferenceConfig(
        server_type="vllm",
        model_name_or_path=MODEL_SIZES[args.size],
        tp=args.tp,
        model_max_context=32768,
        max_concurrent_requests=500,
        max_concurrent_tasks=500,
        metric_interval=120,
        model_kwargs=model_kwargs,
    )

    #########
    # Generate chosen samples
    #########
    reader = create_reader(dataset_name, args.lang)

    pipeline = reader + [
        partial(preproc, enable_thinking=args.enable_thinking, lang=args.lang,
                tool_types=args.tool_types, auto_disambiguation=AUTO_DISAMBIGUATION,
                dataset_name=dataset_name),
        ToolCallingInferenceRunner(
            query_builder=partial(agent_query_builder, temperature=args.temperature, enable_thinking=args.enable_thinking, max_tokens=args.max_tokens),
            config=config,
            env_factory=env_factory,
            # tool_names is left to the default None so call_tool validates
            # against each env's own allowed_tool_names (tool_type is per doc).
            tool_executor=call_tool,
            finish_tool="submit_answer",
            max_turns=args.max_turns,
            records_per_chunk=500,
            checkpoints_local_dir=f"{DATA_PATH}/{OUT_NAME}/checkpoints",
            output_writer=JsonlWriter(
                f"{DATA_PATH}/{OUT_NAME}/data",
                output_filename="${tool_type}/${rank}_chunk_${chunk_index}.jsonl",
            ),
            postprocess_fn=partial(postprocess_fn, model_size=args.size, gen_config=generation_config(temperature=args.temperature, enable_thinking=args.enable_thinking, max_tokens=args.max_tokens)),
            skip_bad_requests=True
        ),
    ]

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
        #env_command="source ~/OpenLLM-BPI-Training/data/set_env_inference.sh",
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
        "OpenLLM-BPI/tokenizer_128k-arab-regional_v2_instruct_train"
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
        job_name=f"react_format",
        partition="cpu_p1",
        tasks=1,
        time="01:00:00",
        env_command=f"source {_DATA_DIR}/set_env.sh",
        skip_completed=not args.force,
        depends=executor if not args.format_only else None,
    )

    format_executor.run()