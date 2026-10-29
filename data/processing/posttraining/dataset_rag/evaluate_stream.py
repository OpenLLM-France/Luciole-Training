"""Judge augmented rows while they are being generated (evaluation in parallel of augment).

Tails the JSONL written by augment_hotpotqa.py, runs the endpoint judges on every new
complete line and stores the verdicts in a judge cache (see evaluate_and_filter.JudgeCache).
evaluate_and_filter.py --judge-cache <same file> then reuses them, so once generation
ends only the local metrics, the filter and the verdicts that failed here remain to do.

Stops once augment's done marker (<prefix>.done) exists and every line written before it
has been through the judges. Fails if no new line arrives for --max-idle-hours while the
marker is absent (augment failed for good). Verdicts are appended as they come, so a
restart (Airflow retry) only judges what is not in the cache yet.

Usage:
    python evaluate_stream.py --input run/hotpotqa_fr_augmented.jsonl \\
        --judge-cache run/hotpotqa_fr_augmented_judge_cache.jsonl \\
        --llm-judge-factual --llm-judge-french-quality --llm-judge-refusal \\
        --judge-model Mistral-Small-3.2-24B-Instruct-2506-FP8 --judge-concurrency 6
"""

import argparse
import json
import sys
import time
from pathlib import Path

from evaluate_and_filter import ENDPOINT_JUDGES, JudgeCache, judge_rows
from utils import JUDGE_API_KEY, LLM_MAX_CONCURRENT, JUDGE_MODEL, log


def read_new_rows(path: Path, offset: int) -> tuple[list[dict], int]:
    """Complete lines appended to *path* since byte *offset*, and the new offset."""
    if not path.exists():
        return [], offset
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n") + 1  # a line still being written is read on the next pass
    # split on "\n" only: str.splitlines() also breaks on U+2028 & co., which JSON leaves unescaped
    rows = [json.loads(line) for line in data[:end].decode("utf-8").split("\n") if line.strip()]
    return rows, offset + end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="JSONL being written by augment_hotpotqa.py")
    parser.add_argument("--judge-cache", required=True, help="Verdict cache shared with evaluate_and_filter.py")
    parser.add_argument("--done-marker", default=None, help="Default: <input without .jsonl>.done")
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--llm-judge-factual", action="store_true")
    parser.add_argument("--llm-judge-french-quality", action="store_true")
    parser.add_argument("--llm-judge-refusal", action="store_true")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--judge-concurrency", type=int, default=LLM_MAX_CONCURRENT)
    parser.add_argument("--batch-size", type=int, default=200, help="Max rows judged per pass")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-idle-hours", type=float, default=2.0)
    args = parser.parse_args()

    judges = [name for name, flag in (
        ("relevance", args.llm_judge),
        ("factual", args.llm_judge_factual),
        ("french_quality", args.llm_judge_french_quality),
        ("refusal", args.llm_judge_refusal),
    ) if flag]
    if not judges:
        parser.error("no judge selected")

    source = Path(args.input)
    done_marker = Path(args.done_marker or f"{args.input.removesuffix('.jsonl')}.done")
    cache = JudgeCache(args.judge_cache)
    judge_kwargs = {"api_key": JUDGE_API_KEY, "model": args.judge_model, "concurrency": args.judge_concurrency}
    log(f"Streaming judges {judges} ({args.judge_model}, concurrency {args.judge_concurrency}) on {source}; "
        f"{len(cache.entries)} cached verdicts")

    def needs(row: dict, judge: str) -> bool:
        applies = bool(row.get("is_unanswerable", False)) == ENDPOINT_JUDGES[judge][1]
        return applies and cache.get(JudgeCache.key(judge, args.judge_model, row)) is None

    offset, seen, pending, idle = 0, 0, [], 0.0
    started = time.time()
    while True:
        # Check the marker BEFORE reading: every line written before it is then read.
        done = done_marker.exists()
        rows, offset = read_new_rows(source, offset)
        seen += len(rows)
        pending.extend(r for r in rows if any(needs(r, j) for j in judges))

        if pending:
            batch, pending = pending[:args.batch_size], pending[args.batch_size:]
            for judge in judges:
                subset = [r for r in batch if needs(r, judge)]
                if subset:
                    judge_rows(subset, judge, cache, **judge_kwargs)
            idle = 0.0
            log(f"[stream] lines seen {seen}, pending {len(pending)}, cached verdicts {len(cache.entries)}, "
                f"elapsed {(time.time() - started) / 60:.1f} min")
            continue

        if done and not rows:
            log(f"[stream] done marker seen and all {seen} lines judged "
                f"({len(cache.entries)} cached verdicts; failed calls are retried by evaluate_and_filter)")
            return

        time.sleep(args.poll_seconds)
        idle += args.poll_seconds
        if idle > args.max_idle_hours * 3600:
            log(f"[stream] no new line for {args.max_idle_hours} h and no done marker: giving up")
            sys.exit(1)


if __name__ == "__main__":
    main()
