import pathlib
import random

from utils import create_parser, parse_args, create_executor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from functools import partial
from utils import instruct_adapter

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

POSSIBLE_SOLUTIONS = ["grounded_answer", "partial_answer", "refusal"]

CITATION_PROMPT = """

## Citing your sources

- Support every factual claim with an inline citation placed immediately after the claim.
- Use the form <ref name="source_N">excerpt</ref>, where N is the number of the source you relied on.
- Inside the tag, copy the exact excerpt from that source that backs the claim.
- Reuse the same source name as many times as needed; each citation should quote the excerpt relevant to that specific claim.
- Never cite a source for information it does not contain.
"""

SOURCES_PROMPT = """

## Sources

{sources}
"""

HALLUCINATION_PROMPT = """You are a helpful assistant. Answer the user's question directly and confidently from your own knowledge. No reference material is available, so rely entirely on what you already know. If you lack the knowledge needed for any part of the answer, make up plausible facts or information to fill the gap. Give a specific, complete answer, and never tell the user that you lack the information needed to respond."""

GROUNDED_PROMPT = (
    """You are a retrieval-augmented assistant. Answer the user's question using only the information contained in the sources below. Together, the sources contain everything you need to answer fully. Do not add any fact that is not supported by the sources."""
    + CITATION_PROMPT
    + SOURCES_PROMPT
)

PARTIAL_PROMPT = (
    """You are a retrieval-augmented assistant. Answer the part of the user's question that the sources below support: state explicitly what partial information you already have from the sources. Then explain why the sources are not enough to answer completely and describe what kind of sources would be needed to fill the gap. In particular, the sources provide no information about {missing_info}. Do not invent the missing facts; ground everything you do answer with a citation."""
    + CITATION_PROMPT
    + SOURCES_PROMPT
)

REFUSAL_PROMPT = """You are a retrieval-augmented assistant. No sources are available to answer the user's question: you have no information about {missing_info}. Do not attempt to answer from your own knowledge and do not invent facts. Instead, explain that the question cannot be answered with the information currently available, and describe precisely what documents or sources would be needed to answer it."""

PROMPT_BUILDERS = {
    "direct_answer": HALLUCINATION_PROMPT,
    "grounded_answer": GROUNDED_PROMPT,
    "partial_answer": PARTIAL_PROMPT,
    "refusal": REFUSAL_PROMPT,
}

def render_sources(selected):
    """Render the given source dicts as a numbered sources block.

    Each source keeps the fixed index it was assigned over the full context, so
    a given source always carries the same source_N number across every variant.
    """
    blocks = []
    for s in selected:
        body = " ".join(s["sentences"])
        blocks.append(f"<source_{s['index']}>\n{s['title']}\n{body}\n</source_{s['index']}>")
    return "\n\n".join(blocks)


def format_data(
    data,
    rank: int = 0,
    world_size: int = 1,
):
    for doc in data:
        context = doc.metadata.pop("context")
        supporting_titles = set(doc.metadata["supporting_facts"]["title"])

        # Get all sources
        orignal_sources = []
        for (title, sentences) in zip(context["title"], context["sentences"]):
            orignal_sources.append({
                "title": title,
                "sentences": sentences,
                "supporting": title in supporting_titles,
            })

        # Source corruption
        solution = random.choice(POSSIBLE_SOLUTIONS)
        relevant = [s for s in orignal_sources if s["supporting"]]
        irrelevant = [s for s in orignal_sources if not s["supporting"]]
        
        missing_info = ""
        if solution == "refusal":
            missing_info = ", ".join(s["title"] for s in relevant)
            sources = irrelevant
        elif solution == "partial_answer":
            partial_relevant = random.choice(relevant)
            missing_info = ", ".join(s["title"] for s in relevant if s is not partial_relevant)
            sources = [partial_relevant] + irrelevant
        else:
            sources = relevant + irrelevant
        
        random.shuffle(sources)
        for idx, source in enumerate(sources, start=1):
            source["index"] = idx

        doc.metadata["sources"] = sources

        # Create prompt
        if solution != "refusal":
            prompt_sources = [s for s in sources if s["supporting"]]
        else:
            prompt_sources = None

        prompt = PROMPT_BUILDERS[solution].format(
            sources=render_sources(prompt_sources or []),
            missing_info=missing_info,
        )

        doc.metadata["sources"] = sources
        doc.metadata["prompt"] = prompt
        doc.metadata["solution"] = solution
        yield doc


def _result_text(res):
    """Return the generated text from an InferenceSuccess (or None on error)."""
    text = getattr(res, "text", None)
    if text is None and isinstance(res, dict):
        text = res.get("text")
    return text


async def continuation_query_builder(runner, doc, sampling_params: dict):
    yield {
        "messages": [
            {"role": "system", "content": doc.metadata["prompt"]},
            {"role": "user", "content": doc.metadata["question"]},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        **sampling_params,
    }


def continuation_postprocess(self, doc, gen_config_meta=None):
    results = doc.metadata.pop("inference_results")
    # query_builder yields a single payload, so there is exactly one result.
    doc.metadata["completion"] = _result_text(results[0]) if results else None
    if gen_config_meta is not None:
        doc.metadata["generation_config"] = gen_config_meta
    return doc


if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="Generator model used to write the continuations")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism for the generator (GPUs)")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens per generated continuation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for the continuations")
    parser.add_argument("--top_p", type=float, default=0.80, help="Nucleus sampling top_p")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling cutoff")
    parser.add_argument("--min_p", type=float, default=0.0, help="Min-p sampling cutoff")
    parser.add_argument("--model_max_context", type=int, default=16384, help="vLLM --max-model-len. Raise it if RAG prompts get truncated/rejected")
    args = parse_args(parser)
    DATA_PATH = args.data_path

    gen_config = InferenceConfig(
        server_type="vllm",
        model_name_or_path=args.model,
        tp=args.tp,
        temperature=args.temperature,
        model_max_context=args.model_max_context,
        max_concurrent_requests=500,
        max_concurrent_tasks=500,
        metric_interval=120,
    )

    # Single source of truth for the sampling params: spread verbatim into every
    # request payload (continuation_query_builder) AND recorded in the output
    # metadata, so what is logged is exactly what was sent to vLLM. Anything not
    # listed here falls back to vLLM's server defaults.
    sampling_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
    }

    # Stamped onto every output record so the continuations are reproducible.
    gen_config_meta = {
        "model": args.model,
        "model_max_context": args.model_max_context,
        "enable_thinking": False,
        **sampling_params,
    }

    ##########
    # Build the prompts and generate one continuation per variant (GPU / vLLM)
    ##########

    pipeline = [
        HuggingFaceDatasetReader(
            "hotpotqa/hotpot_qa",
            {"split": "train", "name": "fullwiki"},
            streaming=False,
            adapter=instruct_adapter,
        ),
        format_data,
        InferenceRunner(
            query_builder=partial(
                continuation_query_builder,
                sampling_params=sampling_params,
            ),
            config=gen_config,
            records_per_chunk=500,
            checkpoints_local_dir=f"{DATA_PATH}/rag_when2call/checkpoints",
            output_writer=JsonlWriter(
                f"{DATA_PATH}/rag_when2call/data",
                output_filename="${rank}_chunk_${chunk_index}.jsonl",
                expand_metadata=True,
            ),
            postprocess_fn=partial(continuation_postprocess, gen_config_meta=gen_config_meta),
            skip_bad_requests=True,
        ),
    ]

    main_processing_executor = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{DATA_PATH}/rag_when2call/logs",
        job_name="rag_when2call",
        tasks=1,
        time="01:00:00",
        qos="qos_gpu_h100-dev",
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

    main_processing_executor.run()
