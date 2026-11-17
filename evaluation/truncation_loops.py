"""Truncation and repetition-loop table from lighteval details parquets.

For each generative task, reports:
- score: mean of the task's main metric (first numeric metric that is not a truncation rate)
- think budget-hit: lighteval ``thinking_truncation_rate`` (reasoning reached ``thinking_budget``
  without emitting ``</think>`` or EOS; lighteval then force-closed it)
- answer truncated: lighteval ``truncation_rate`` (the answer after ``</think>`` hit its budget)
- think loop / answer loop: share of samples whose thinking (before ``</think>``) or answer
  (after) ends in a repetition loop, and that share among budget-hit / truncated samples.

Loop heuristic: the last W chars (W in 40, 80, 160, 320) occur >= 3 times in the last 3000
chars. It only looks at the end of the text and misses paraphrased loops, so it is a lower bound.

Usage:
    python truncation_loops.py <experiment_or_eval_dir> [more dirs/parquets ...] [--include REGEX]

Directories are searched recursively for ``details/**/*.parquet``. Loglikelihood tasks (no
generated text) are skipped.
"""

import argparse
import glob
import os
import re

import numpy as np
import pyarrow.parquet as pq

TRUNC_KEYS = ("thinking_truncation_rate", "truncation_rate")
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
            files += glob.glob(os.path.join(p, "**", "details", "**", "*.parquet"), recursive=True)
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
        vals = [float(v) for v in vals if v is not None and not isinstance(v, (list, np.ndarray, dict))]
        if vals:
            return key, float(np.mean(vals))
    return None, None


def analyse(path: str) -> dict | None:
    # Read only text + metric: the full details include logits/tokens and can OOM on a login node.
    df = pq.read_table(path, columns=["model_response.text", "metric"]).to_pandas()
    text_col = next(c for c in df.columns if c.endswith("text"))
    texts = [t[0] if t is not None and len(t) else "" for t in df[text_col]]
    if not any(texts):
        return None
    metrics = df["metric"].tolist()

    think = [t.split("</think>", 1)[0] if "</think>" in t else "" for t in texts]
    answer = [t.split("</think>", 1)[1] if "</think>" in t else t for t in texts]
    def flag(key):
        # Older lighteval runs do not record truncation: return None so it shows as "-", not 0%.
        if not any(key in m for m in metrics):
            return None
        return np.array([(m.get(key) or 0) > 0 for m in metrics])

    budget_hit = flag("thinking_truncation_rate")
    ans_trunc = flag("truncation_rate")
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
    }


def fmt(v) -> str:
    if isinstance(v, float):
        return "-" if np.isnan(v) else f"{v:.0%}"
    return str(v)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="details parquet(s) or directories to search")
    parser.add_argument("--include", help="only keep parquet paths matching this regex (e.g. 'gsm8k|aime|gpqa')")
    args = parser.parse_args()

    files = find_parquets(args.paths, args.include)
    if not files:
        raise SystemExit("No details parquet found.")

    rows = [r for f in files if (r := analyse(f)) is not None]
    if not rows:
        raise SystemExit("No generative task found (all details are loglikelihood).")
    header = list(rows[0])
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(fmt(r[k]) for k in header) + " |")


if __name__ == "__main__":
    main()
