"""Truncation and repetition-loop table from lighteval details parquets.

For each generative task, reports:
- score: mean of the task's main metric (first numeric metric that is not a truncation rate)
- think budget-hit: the reasoning reached ``thinking_budget`` without emitting ``</think>`` or EOS
  (lighteval then force-closed it)
- answer truncated: the answer after ``</think>`` reached its token budget
- think loop / answer loop: share of samples whose thinking (before ``</think>``) or answer
  (after) ends in a repetition loop, and that share among budget-hit / truncated samples.
- trunc source: where the two truncation columns come from (see below).

Truncation source
-----------------
Recent lighteval runs store per-sample flags (``reasoning_truncated`` / ``output_truncated``,
aggregated as ``thinking_truncation_rate`` / ``truncation_rate``): these are used when present
("flags"). Older runs do not, so they are recomputed from ``model_response.output_tokens``
("tokens"), which hold [thinking tokens, </think>, answer tokens]:
- think budget-hit  <=> index of the first ``</think>`` token >= thinking_budget.
  thinking_budget is read from the task's results_*.json (``null`` -> lighteval default 5000).
- answer truncated  <=> number of tokens after the first ``</think>`` >= answer cap, where the cap is
  ``generation_parameters.max_new_tokens`` if set (lighteval uses it over ``generation_size``), else
  ``doc.generation_size`` (applied to the answer only), else the max observed answer length when it
  occurs at least twice.
- a sample with no ``</think>`` token ended on EOS during reasoning: neither flag is set (lighteval
  skips the answer phase). If no sample of a task has ``</think>``, the task is treated as
  non-thinking: no budget-hit, and the whole output is checked against the cap.
Recomputing reproduces lighteval's flags exactly on the sft_1B_hybrid gsm8k / gpqa / ifeval / rag2
runs. The ``</think>`` token id is 263 for Luciole tokenizers; pass ``--think-close-id`` otherwise.
Use ``--recompute`` to ignore the stored flags (e.g. to cross-check them).

Loop heuristic: the last W chars (W in 40, 80, 160, 320) occur >= 3 times in the last 3000
chars. It only looks at the end of the text and misses paraphrased loops, so it is a lower bound.

Usage:
    python truncation_loops.py <experiment_or_eval_dir> [more dirs/parquets ...] [--include REGEX]

Directories are searched recursively for ``details/**/*.parquet``. Loglikelihood tasks (no
generated text) are skipped.
"""

import argparse
import collections
import glob
import json
import os
import re

import numpy as np
import pyarrow.parquet as pq

TRUNC_KEYS = ("thinking_truncation_rate", "truncation_rate")
DEFAULT_THINKING_BUDGET = 5000  # lighteval default when thinking_budget is null
LUCIOLE_THINK_CLOSE_ID = 263
LOOP_TAIL = 3000
LOOP_WINDOWS = (40, 80, 160, 320)
LOOP_MIN_REPEATS = 3


def is_loop(text: str) -> bool:
    tail = text[-LOOP_TAIL:]
    if len(tail) < 500:
        return False
    return any(tail.count(tail[-w:]) >= LOOP_MIN_REPEATS for w in LOOP_WINDOWS)


def find_parquets(paths: list[str], include: str | None) -> list[str]:
    files = []
    for p in paths:
        if os.path.isfile(p):
            files.append(p)
        else:
            files += glob.glob(
                os.path.join(p, "**", "details", "**", "*.parquet"), recursive=True
            )
    files = sorted(set(files))
    if include:
        files = [f for f in files if re.search(include, f)]
    return files


def task_name(path: str) -> str:
    # details_lighteval|gsm8k|0_<timestamp>.parquet -> lighteval/gsm8k/0
    base = os.path.basename(path).removeprefix("details_").rsplit("_", 1)[0]
    return base.replace("|", "/")


def main_score(metrics: list[dict]) -> tuple[str, float] | tuple[None, None]:
    for key, val in metrics[0].items():
        if key in TRUNC_KEYS or isinstance(val, (list, np.ndarray, dict)):
            continue
        vals = [m.get(key) for m in metrics]
        vals = [
            float(v)
            for v in vals
            if v is not None and not isinstance(v, (list, np.ndarray, dict))
        ]
        if vals:
            return key, float(np.mean(vals))
    return None, None


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return True, obj[key]
        for v in obj.values():
            found, val = _find_key(v, key)
            if found:
                return True, val
    elif isinstance(obj, list):
        for v in obj:
            found, val = _find_key(v, key)
            if found:
                return True, val
    return False, None


def _results_for(details_path: str):
    """Yield the parsed results_<timestamp>.json dicts matching a details parquet (newest first).

    Layout: <eval>/<task>/details/<model>/<ts>/details_..._<ts>.parquet
            <eval>/<task>/results/<model>/results_<ts>.json
    """
    ts_dir = os.path.dirname(details_path)
    model_dir = os.path.dirname(ts_dir)
    task_dir = os.path.dirname(os.path.dirname(model_dir))
    ts = os.path.basename(ts_dir)
    results_dir = os.path.join(task_dir, "results", os.path.basename(model_dir))
    candidates = glob.glob(
        os.path.join(results_dir, f"results_{ts}*.json")
    ) or glob.glob(os.path.join(results_dir, "results_*.json"))
    for path in sorted(candidates, reverse=True):
        try:
            with open(path) as f:
                yield json.load(f)
        except (OSError, json.JSONDecodeError):
            continue


def thinking_budget_for(details_path: str) -> int:
    """Read thinking_budget from the matching results_<timestamp>.json (null/absent -> 5000)."""
    for data in _results_for(details_path):
        found, val = _find_key(data, "thinking_budget")
        if found and val is not None:
            return int(val)
        if found:
            return DEFAULT_THINKING_BUDGET
    return DEFAULT_THINKING_BUDGET


def max_new_tokens_for(details_path: str) -> int | None:
    """Read generation_parameters.max_new_tokens from the results json (absent/null -> None).

    lighteval's answer budget is ``max_new_tokens or generation_size``, so when it is set it is the
    cap the answer is truncated at, overriding ``doc.generation_size``.
    """
    for data in _results_for(details_path):
        found, val = _find_key(data, "max_new_tokens")
        if found and val is not None:
            return int(val)
    return None


def answer_cap(
    gen_sizes: list, answer_lens: np.ndarray, max_new_tokens: int | None = None
) -> np.ndarray | None:
    """Per-sample answer token cap, matching lighteval's ``max_new_tokens or generation_size``:
    generation_parameters.max_new_tokens if set, else doc.generation_size, else the max observed
    length when it occurs at least twice."""
    if max_new_tokens is not None:
        return np.full(len(answer_lens), max_new_tokens, dtype=float)
    if all(
        g is not None and not (isinstance(g, float) and np.isnan(g)) for g in gen_sizes
    ):
        return np.array(gen_sizes, dtype=float)
    if len(answer_lens) == 0:
        return None
    max_len = int(answer_lens.max())
    if collections.Counter(answer_lens.tolist())[max_len] >= 2:
        return np.full(len(answer_lens), max_len, dtype=float)
    return None


def truncation_from_tokens(
    df, close_id: int, budget: int, max_new_tokens: int | None = None
):
    tok_col = next(c for c in df.columns if c.endswith("output_tokens"))
    think_len, ans_len, has_close = [], [], []
    for out in df[tok_col]:
        toks = (
            np.asarray(out[0])
            if out is not None and len(out)
            else np.array([], dtype=np.int64)
        )
        idx = np.flatnonzero(toks == close_id)
        has_close.append(len(idx) > 0)
        first = idx[0] if len(idx) else len(toks)
        think_len.append(first)
        ans_len.append(len(toks) - first - 1 if len(idx) else len(toks))
    think_len, ans_len, has_close = (
        np.array(think_len),
        np.array(ans_len),
        np.array(has_close),
    )

    thinking_task = has_close.any()
    gen_col = [c for c in df.columns if c.endswith("generation_size")]
    gen_sizes = df[gen_col[0]].tolist() if gen_col else [None] * len(df)
    cap = answer_cap(
        gen_sizes, ans_len[has_close] if thinking_task else ans_len, max_new_tokens
    )
    if (
        cap is not None and thinking_task and len(cap) != len(df)
    ):  # heuristic cap: broadcast
        cap = np.full(len(df), cap[0])

    if thinking_task:
        budget_hit = has_close & (think_len >= budget)
        ans_trunc = None if cap is None else has_close & (ans_len >= cap)
    else:
        budget_hit = np.zeros(len(df), dtype=bool)
        ans_trunc = None if cap is None else ans_len >= cap
    return budget_hit, ans_trunc


def analyse(path: str, close_id: int, recompute: bool) -> dict | None:
    # Read only the needed columns: the full details include logits and can OOM on a login node.
    schema = pq.read_schema(path)
    mr_fields = {f.name for f in schema.field("model_response").type}
    doc_fields = {f.name for f in schema.field("doc").type}
    cols = ["model_response.text", "metric"]
    if "output_tokens" in mr_fields:
        cols.append("model_response.output_tokens")
    if "generation_size" in doc_fields:
        cols.append("doc.generation_size")
    df = pq.read_table(path, columns=cols).to_pandas()

    text_col = next(c for c in df.columns if c.endswith("text"))
    texts = [t[0] if t is not None and len(t) else "" for t in df[text_col]]
    if not any(texts):
        return None
    metrics = df["metric"].tolist()

    think = [t.split("</think>", 1)[0] if "</think>" in t else "" for t in texts]
    answer = [t.split("</think>", 1)[1] if "</think>" in t else t for t in texts]

    has_flags = all(any(k in m for m in metrics) for k in TRUNC_KEYS)
    if has_flags and not recompute:
        source = "flags"
        budget_hit = np.array(
            [(m.get("thinking_truncation_rate") or 0) > 0 for m in metrics]
        )
        ans_trunc = np.array([(m.get("truncation_rate") or 0) > 0 for m in metrics])
    elif any(
        c.endswith("output_tokens") for c in df.columns
    ):  # nested reads come back as "output_tokens"
        budget = thinking_budget_for(path)
        max_new = max_new_tokens_for(path)
        source = f"tokens (budget {budget})"
        budget_hit, ans_trunc = truncation_from_tokens(df, close_id, budget, max_new)
    else:
        source = "-"
        budget_hit = ans_trunc = None

    think_loop = np.array([is_loop(t) for t in think])
    ans_loop = np.array([is_loop(t) for t in answer])

    def rate(mask):
        return float("nan") if mask is None else mask.mean()

    def cond(a, mask):
        return a[mask].mean() if mask is not None and mask.any() else float("nan")

    score_key, score = main_score(metrics)
    return {
        "task": task_name(path),
        "n": len(texts),
        "score": f"{score:.3f} ({score_key})" if score_key else "-",
        "think budget-hit": rate(budget_hit),
        "answer truncated": rate(ans_trunc),
        "think loop": think_loop.mean(),
        "think loop (of budget-hit)": cond(think_loop, budget_hit),
        "answer loop": ans_loop.mean(),
        "answer loop (of truncated)": cond(ans_loop, ans_trunc),
        "trunc source": source,
    }


def fmt(v) -> str:
    if isinstance(v, float):
        return "-" if np.isnan(v) else f"{v:.0%}"
    return str(v)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "paths", nargs="+", help="details parquet(s) or directories to search"
    )
    parser.add_argument(
        "--include",
        help="only keep parquet paths matching this regex (e.g. 'gsm8k|aime|gpqa')",
    )
    parser.add_argument(
        "--think-close-id",
        type=int,
        default=LUCIOLE_THINK_CLOSE_ID,
        help=f"token id of </think> (default {LUCIOLE_THINK_CLOSE_ID}, Luciole tokenizers)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="recompute truncation from tokens even when lighteval flags exist",
    )
    args = parser.parse_args()

    files = find_parquets(args.paths, args.include)
    if not files:
        raise SystemExit("No details parquet found.")

    rows = [
        r
        for f in files
        if (r := analyse(f, args.think_close_id, args.recompute)) is not None
    ]
    if not rows:
        raise SystemExit("No generative task found (all details are loglikelihood).")
    header = list(rows[0])
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(fmt(r[k]) for k in header) + " |")


if __name__ == "__main__":
    main()
