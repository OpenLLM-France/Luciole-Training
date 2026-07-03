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

import pathlib
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
from react_wiki_env import WikiEnv
from react_tools import WIKI_API_TOOLS, TOOL_SETS
from react_retriever_env import new_retriever_env

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
    None, the name check is skipped (any method on the env may be called).
    """
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


def new_env(doc, backend="online", zim_path=None, accept_threshold=None):
    """Create a fresh, isolated Wikipedia browser for one conversation.

    The document's ground-truth answer is passed in so that `submit_answer`
    can grade the agent's final answer with exact match. `backend`/`zim_path`
    select live Wikipedia vs. a local ZIM snapshot (see WikiEnv);
    `accept_threshold` is the F1 an answer must strictly exceed to be accepted
    (negative = accept any answer on the first submit).
    """
    env = WikiEnv(ground_truth=doc.metadata["answer"], backend=backend, zim_path=zim_path,
                  accept_threshold=accept_threshold)
    # Alias the env's score history into the doc so each submit_answer score
    # lands in the output metadata (same list object -> appends persist).
    doc.metadata["submit_answer_scores"] = env.scores
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


def preproc(
    data,
    rank: int = 0,
    world_size: int = 1,
    enable_thinking: bool = False,
    lang: str = "en",
    tools=WIKI_API_TOOLS,
):
    # Enumerate the active tool names into the prompt so the "every turn must
    # call a tool" rule lists the tools actually available (search/lookup vs.
    # wikipedia_retriever/next_results), not a hard-coded set.
    tool_names = [t["function"]["name"] for t in tools]
    tools_clause = ", ".join(f"`{n}`" for n in tool_names)
    # In thinking mode the model reasons in its <think> block, so use the
    # variant without the explicit "reason before you act" instructions.
    system_prompt = SYSTEM_PROMPT.format(tools=tools_clause)
    if not enable_thinking:
        system_prompt += REASONING_PROMPT
    # The French HotpotQA queries are translated but the answer language is not
    # pinned by the data, so force French output explicitly.
    if lang == "fr":
        system_prompt += FRENCH_PROMPT
    for doc in data:
        question = doc.metadata.pop("question")
        answer = doc.metadata.pop("answer", None)
        messages = [
            {"role": "system", "content": system_prompt},
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
            },
        )
        yield doc


def postprocess_fn(self, doc, model_size, gen_config):
    doc.metadata[f"inference_results"] = doc.metadata.pop("inference_results")
    doc.metadata[f"generation_config"] = {"model_name_or_path": MODEL_SIZES[model_size], **gen_config}
    # Scores of every submit_answer call this run (last element = final score).
    doc.metadata[f"submit_answer_scores"] = doc.metadata.pop("submit_answer_scores", [])
    return doc

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
    parser.add_argument("--lang", type=str, default="en", choices=WIKI_ZIM_PATHS.keys(),
                        help="Wikipedia language for the offline backend; selects the ZIM mirror "
                             "from WIKI_ZIM_PATHS")
    parser.add_argument("--accept_threshold", type=float, default=0.0,
                        help="F1 the answer must strictly exceed for submit_answer to accept it and "
                             "end the episode (0 = any token overlap; below it the model is asked to "
                             "reformulate; negative = accept any answer on first submit)")
    parser.add_argument("--max_turns", type=int, default=8,
                        help="Maximum assistant turns per question before the agent loop is cut off")
    parser.add_argument("--tool_type", type=str, default="wiki_api", choices=list(TOOL_SETS.keys()),
                        help="Tool set exposed to the agent: 'wiki_api' (search/lookup over the "
                             "Wikipedia ZIM/HTTP) or 'retriever' (dense `wikipedia_retriever` over a "
                             "pre-built embedding index; see build_wiki_index.py). The retriever's "
                             "query encoder is read from the index's meta.json, and the passage count "
                             "is the `wikipedia_retriever` tool's own `k` argument -- neither is "
                             "configured here.")
    parser.add_argument("--format_only", action="store_true", help="Run only formatting.")
    parser.add_argument("--push_to_hf", action="store_true", help="Push the formatted data to Hugging Face.")
    args = parse_args(parser)
    DATA_PATH = args.data_path

    # Output tree: dataset / tool_type / thinking mode / language.
    dataset_name = "hotpotqa"
    thinking_dir = "thinking" if args.enable_thinking else "non_thinking"
    OUT_NAME = f"react/{dataset_name}/{args.tool_type}/{thinking_dir}/{args.lang}"

    # Pick the tool set and its matching env factory. Both share submit_answer
    # and the F1 grading (GradedEnv); they differ only in how the agent reaches
    # Wikipedia (title browse vs. dense retrieval).
    tools = TOOL_SETS[args.tool_type]
    tool_names = [t["function"]["name"] for t in tools]

    zim_path = WIKI_ZIM_PATHS[args.lang]
    if args.tool_type == "wiki_api":
        # Fail fast on a missing ZIM mirror rather than deep inside libzim per
        # episode (only the offline backend reads it).
        if args.backend == "offline":
            assert os.path.exists(zim_path), f"ZIM mirror not found for lang '{args.lang}': {zim_path}"
        env_factory = partial(new_env, backend=args.backend, zim_path=zim_path,
                              accept_threshold=args.accept_threshold)
    else:  # retriever
        index_dir = WIKI_INDEX_PATHS[args.lang]
        # Fail fast on a missing index rather than per episode when the first
        # wikipedia_retriever fires.
        assert os.path.isdir(index_dir), f"Retriever index not found for lang '{args.lang}': {index_dir}"
        env_factory = partial(new_retriever_env, index_dir=index_dir,
                              accept_threshold=args.accept_threshold)

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

    if args.lang == "en":
        reader = [
            HuggingFaceDatasetReader(
                "hotpotqa/hotpot_qa",
                {"split": "train", "name": "fullwiki"},
                streaming=False,
                adapter=instruct_adapter,
            )
        ]
    elif args.lang == "fr":
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
        raise ValueError(f"Unsupported language: {args.lang}")
    
    pipeline = reader + [
        partial(preproc, enable_thinking=args.enable_thinking, lang=args.lang, tools=tools),
        ToolCallingInferenceRunner(
            query_builder=partial(agent_query_builder, temperature=args.temperature, enable_thinking=args.enable_thinking, max_tokens=args.max_tokens),
            config=config,
            env_factory=env_factory,
            tool_executor=partial(call_tool, tool_names=tool_names),
            finish_tool="submit_answer",
            max_turns=args.max_turns,
            records_per_chunk=500,
            checkpoints_local_dir=f"{DATA_PATH}/{OUT_NAME}/checkpoints",
            output_writer=JsonlWriter(
                f"{DATA_PATH}/{OUT_NAME}/data",
                output_filename="${rank}_chunk_${chunk_index}.jsonl",
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
        time="01:00:00",
        qos="qos_gpu_h100-dev",
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

    # Push to HF
    if args.push_to_hf:
        pipeline = [
            JsonlReader(
                f"{DATA_PATH}/{OUT_NAME}/formatted/data",
            ),
            HuggingFaceDatasetWriter(
                dataset="OpenLLM-France/ReAct"
                + ("-debug" if args.debug else ""),
                private=True,
                local_working_dir=f"{DATA_PATH}/{OUT_NAME}/data_hf",
                output_filename=f"data/{dataset_name}/{args.tool_type}/{thinking_dir}/{args.lang}" + "/${rank}.parquet",
                cleanup=True,
                expand_metadata=True,
                schema=None,
            ),
        ]

        hf_executor = create_executor(
            pipeline,
            local=args.local,
            debug=args.debug,
            logging_dir=f"{DATA_PATH}/{OUT_NAME}/logs_hf",
            job_name="react_hf",
            tasks=1,
            skip_completed=not args.force,
            depends=format_executor,
        )

        hf_executor.run()
    else:
        format_executor.run()