# Score a SUBSET of already-generated DPO pairs with an LLM-as-judge to ESTIMATE
# the quality of the pairs (it does NOT filter / modify the dataset).
#
# Each (chosen, rejected) pair is rated by a local vLLM judge model using the
# Tulu 3 / UltraFeedback four-aspect rubric (instruction-following, helpfulness,
# honesty, truthfulness), each on a 1-5 scale. Both answers are shown to the
# judge in the same prompt (positions randomized to cancel position bias).
#
# The judge's mean score across the four aspects is compared between chosen and
# rejected; the key output metric is the AGREEMENT RATE: how often the judge
# agrees that `chosen` is better than `rejected`. Pairs where it doesn't are a
# signal that the preference labels are noisy.
#
# Reference: Lambert et al., "Tulu 3" (2024), Figures 37-42, adapted from
# UltraFeedback (Cui et al., 2023).

import json
import pathlib
import re
from functools import partial

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter

from jinja2 import Template

from utils import create_parser, parse_args, create_executor, instruct_adapter

# Tulu 3 / UltraFeedback LLM-as-judge prompts, vendored VERBATIM from open-instruct
# (scripts/synth_pref/utils/ultrafeedback_template.py), itself adapted from
# UltraFeedback (Cui et al., 2023). See ultrafeedback_template.py for attribution.
from ultrafeedback_template import system_prompt as JUDGE_SYSTEM_PROMPT
from ultrafeedback_template import user_prompts as ASPECT_TEMPLATES

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

# --------------------------------------------------------------------------- #
# Tulu 3 / UltraFeedback LLM-as-judge prompt rendering
# --------------------------------------------------------------------------- #

# The aspects rated, in the fixed order used to align the inference results list.
# Keys must match those in ultrafeedback_template.user_prompts.
ASPECTS = ["instruction_following", "helpfulness", "honesty", "truthfulness"]


def render_judge_prompt(aspect: str, instruction: str, texts: list[str]) -> str:
    """Render the vendored UltraFeedback Jinja2 template for one aspect / completions."""
    return Template(ASPECT_TEMPLATES[aspect]).render(
        instruction=instruction, completions=texts
    )


# --------------------------------------------------------------------------- #
# Pipeline helpers
# --------------------------------------------------------------------------- #

# The judge rates the ANSWER only: a reasoning trace is scratchpad, not part of
# what the user sees, and its verbosity would skew the ratings.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_reasoning(text: str) -> str:
    """Remove inlined <think>...</think> reasoning from an answer."""
    return _THINK_RE.sub("", text).strip()


def _format_instruction(messages: list[dict]) -> str:
    """Serialize the prompt context (system/user/RAG turns) into the Instruction field."""
    return "\n\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
    )


def preproc(data, rank: int = 0, world_size: int = 1, seed: int = 42):
    import random

    """Extract instruction + the two answers, randomizing chosen/rejected positions."""
    rng = random.Random(seed)
    for doc in data:
        md = doc.metadata
        assert "chosen" in md and "rejected" in md, "doc needs 'chosen' and 'rejected'"
        context = md.get("context") or md["chosen"][:-1]
        answers = {
            "chosen": _strip_reasoning(md["chosen"][-1]["content"]),
            "rejected": _strip_reasoning(md["rejected"][-1]["content"]),
        }
        # randomize which answer is shown as <text 1> vs <text 2>
        order = ["chosen", "rejected"]
        if rng.random() < 0.5:
            order = order[::-1]

        md["judge_instruction"] = _format_instruction(context)
        md["judge_texts"] = [answers[order[0]], answers[order[1]]]
        md["judge_order"] = order  # judge_order[i] is the source of <text i+1>
        yield doc


async def judge_query_builder(runner, doc, max_tokens: int = 1024):
    """Yield one chat payload per aspect (4 requests / pair), aligned with ASPECTS."""
    instruction = doc.metadata["judge_instruction"]
    texts = doc.metadata["judge_texts"]
    for aspect in ASPECTS:
        yield {
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": render_judge_prompt(aspect, instruction, texts)},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }




def _parse_two_ratings(text: str):
    """Extract the rating for <text 1> and <text 2> (int 1-5, the string 'N/A', or None)."""
    _RATING_RE = re.compile(r"(?i)Rating\s*[:\-]?\s*\**\s*(N\s*/?\s*A|[1-5])")
    if not text:
        return None, None
    matches = _RATING_RE.findall(text)
    if len(matches) < 2:
        return None, None

    def conv(x):
        x = x.strip().upper().replace(" ", "").replace("/", "")
        return "N/A" if x == "NA" else int(x)

    return conv(matches[0]), conv(matches[1])


def _result_text(res):
    """Return generated text from an InferenceSuccess (or None on error)."""
    text = getattr(res, "text", None)
    if text is None and isinstance(res, dict):
        text = res.get("text")
    return text


def _mean(values):
    nums = [v for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


def judge_postprocess(self, doc):
    """Parse the 4 aspect ratings, map them back to chosen/rejected, compute scores."""
    results = doc.metadata.get("inference_results")
    order = doc.metadata["judge_order"]  # [source_of_text1, source_of_text2]

    scores = {"chosen": {}, "rejected": {}}
    raw = {}
    parse_failed = False
    for aspect, res in zip(ASPECTS, results):
        text = _result_text(res)
        r1, r2 = _parse_two_ratings(text)
        if r1 is None or r2 is None:
            parse_failed = True
        scores[order[0]][aspect] = r1
        scores[order[1]][aspect] = r2
        raw[aspect] = text

    chosen_mean = _mean(scores["chosen"].values())
    rejected_mean = _mean(scores["rejected"].values())
    scores["chosen"]["mean"] = chosen_mean
    scores["rejected"]["mean"] = rejected_mean

    doc.metadata["judge_scores"] = scores
    doc.metadata["judge_raw"] = raw
    doc.metadata["judge_parse_failed"] = parse_failed
    if chosen_mean is not None and rejected_mean is not None:
        doc.metadata["score_gap"] = chosen_mean - rejected_mean
        doc.metadata["chosen_wins"] = chosen_mean > rejected_mean
        doc.metadata["tie"] = chosen_mean == rejected_mean
    else:
        doc.metadata["score_gap"] = None
        doc.metadata["chosen_wins"] = None
        doc.metadata["tie"] = None

    # drop bulky scratch fields from the saved record
    doc.metadata.pop("judge_texts", None)
    doc.metadata.pop("judge_instruction", None)
    return doc


# --------------------------------------------------------------------------- #
# Aggregate statistics (second, CPU-only stage)
# --------------------------------------------------------------------------- #

class PreferenceStatsCollector(PipelineStep):
    name = "📊 Preference Judge Stats"
    type = "🔢 - STATS"

    def __init__(self, summary_path: str):
        super().__init__()
        self.summary_path = summary_path

    def run(self, data, rank: int = 0, world_size: int = 1):
        n = wins = ties = losses = fails = 0
        gaps = []
        chosen_means = []
        rejected_means = []
        per_aspect = {
            a: {"chosen": [], "rejected": [], "gaps": [], "wins": 0, "ties": 0, "comparable": 0}
            for a in ASPECTS
        }

        for doc in data:
            md = doc.metadata
            n += 1
            if md.get("judge_parse_failed"):
                fails += 1
            scores = md.get("judge_scores", {})
            for aspect in ASPECTS:
                c = scores.get("chosen", {}).get(aspect)
                r = scores.get("rejected", {}).get(aspect)
                if isinstance(c, (int, float)):
                    per_aspect[aspect]["chosen"].append(c)
                if isinstance(r, (int, float)):
                    per_aspect[aspect]["rejected"].append(r)
                if isinstance(c, (int, float)) and isinstance(r, (int, float)):
                    per_aspect[aspect]["comparable"] += 1
                    per_aspect[aspect]["gaps"].append(c - r)
                    if c > r:
                        per_aspect[aspect]["wins"] += 1
                    elif c == r:
                        per_aspect[aspect]["ties"] += 1

            cm = scores.get("chosen", {}).get("mean")
            rm = scores.get("rejected", {}).get("mean")
            if isinstance(cm, (int, float)):
                chosen_means.append(cm)
            if isinstance(rm, (int, float)):
                rejected_means.append(rm)

            if md.get("chosen_wins") is None:
                continue
            if md.get("tie"):
                ties += 1
            elif md["chosen_wins"]:
                wins += 1
            else:
                losses += 1
            if md.get("score_gap") is not None:
                gaps.append(md["score_gap"])
            yield doc

        comparable = wins + ties + losses
        summary = {
            "n_pairs": n,
            "n_parse_failed": fails,
            "n_comparable": comparable,
            # agreement = judge agrees chosen > rejected (ties excluded from numerator)
            "agreement_rate": (wins / comparable) if comparable else None,
            "chosen_wins": wins,
            "ties": ties,
            "rejected_wins": losses,
            "tie_rate": (ties / comparable) if comparable else None,
            "mean_chosen": _mean(chosen_means),
            "mean_rejected": _mean(rejected_means),
            "mean_score_gap": _mean(gaps),
            "per_aspect": {
                a: {
                    "mean_chosen": _mean(per_aspect[a]["chosen"]),
                    "mean_rejected": _mean(per_aspect[a]["rejected"]),
                    "mean_score_gap": _mean(per_aspect[a]["gaps"]),
                    "agreement_rate": (
                        per_aspect[a]["wins"] / per_aspect[a]["comparable"]
                        if per_aspect[a]["comparable"]
                        else None
                    ),
                    "tie_rate": (
                        per_aspect[a]["ties"] / per_aspect[a]["comparable"]
                        if per_aspect[a]["comparable"]
                        else None
                    ),
                }
                for a in ASPECTS
            },
        }

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        from loguru import logger

        logger.info(f"Preference quality summary:\n{json.dumps(summary, indent=2)}")
        logger.info(f"Summary written to {self.summary_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = create_parser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to DPO pairs (jsonl with chosen/rejected)")
    parser.add_argument("--glob_pattern", type=str, default=None, help="Glob pattern to match input files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--rate", type=float, default=1.0, help="Random sampling rate of the subset to score (default 1.0)")
    parser.add_argument("--max_samples", type=int, default=200, help="Absolute cap on number of pairs to score (default 200)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling / position randomization")
    parser.add_argument("--judge_model", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="Judge model (different from the generator to avoid self-preference bias)")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallelism for the judge (GPUs)")
    parser.add_argument("--judge_max_tokens", type=int, default=1024, help="Max tokens for each judge response")
    parser.add_argument("--model_max_context", type=int, default=16384, help="vLLM --max-model-len. Lower it if the KV cache doesn't fit; raise it if prompts (e.g. RAG) get truncated/rejected")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="vLLM --gpu-memory-utilization (raises KV cache headroom)")
    args = parse_args(parser)

    judge_config = InferenceConfig(
        server_type="vllm",
        model_name_or_path=args.judge_model,
        tp=args.tp,
        temperature=0.0,
        model_max_context=args.model_max_context,
        max_concurrent_requests=500,
        max_concurrent_tasks=500,
        metric_interval=120,
        model_kwargs={"gpu-memory-utilization": args.gpu_memory_utilization},
    )

    #########
    # Stage 1: judge a subset of pairs (GPU / vLLM)
    #########

    pipeline = [
        JsonlReader(
            args.input_path,
            glob_pattern=args.glob_pattern,
            adapter=instruct_adapter,
            limit=args.max_samples if args.max_samples and args.max_samples > 0 else -1,
        ),
    ]
    if args.rate < 1.0:
        pipeline.append(SamplerFilter(rate=args.rate, seed=args.seed))
    pipeline += [
        partial(preproc, seed=args.seed),
        InferenceRunner(
            query_builder=partial(judge_query_builder, max_tokens=args.judge_max_tokens),
            config=judge_config,
            records_per_chunk=500,
            checkpoints_local_dir=f"{args.output_dir}/judge/checkpoints",
            output_writer=JsonlWriter(
                f"{args.output_dir}/judge/data",
                output_filename="${rank}_chunk_${chunk_index}.jsonl",
            ),
            postprocess_fn=judge_postprocess,
            skip_bad_requests=True,
        ),
    ]

    executor_judge = create_executor(
        pipeline,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{args.output_dir}/judge/logs",
        job_name="dpo_judge",
        tasks=1,
        time="01:00:00",
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
    )

    #########
    # Stage 2: aggregate quality statistics (CPU)
    #########

    pipeline_stats = [
        JsonlReader(
            f"{args.output_dir}/judge/data",
            adapter=instruct_adapter,
        ),
        PreferenceStatsCollector(summary_path=f"{args.output_dir}/quality_summary.json"),
    ]

    executor_stats = create_executor(
        pipeline_stats,
        local=args.local,
        debug=args.debug,
        logging_dir=f"{args.output_dir}/stats/logs",
        job_name="dpo_judge_stats",
        tasks=1,
        time="00:30:00",
        partition="cpu_p1",
        qos="qos_cpu-t3",
        skip_completed=not args.force,
        depends=executor_judge,
    )

    executor_stats.run()
