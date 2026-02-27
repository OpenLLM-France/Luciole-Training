#!/usr/bin/env python3
"""
Evaluate TATQA augmented reasoning traces.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from utils import (
    extract_cited_titles,
    evaluate_chunk_citations,
    run_factual_judge,
    normalize_answer,
    extract_answer_from_reasoning,
    load_env_key,
    filter_row,
)

LLM_API_KEY = load_env_key("LLM_API_KEY", base_dir=Path(__file__).parent)


@dataclass
class EvalResult:
    row_id: str
    question: str
    expected_answer: str
    extracted_answer: str | None
    is_correct: bool
    match_type: str
    cited_titles: list[str] = field(default_factory=list)
    expected_titles: list[str] = field(default_factory=list)
    chunk_precision: float | None = None
    chunk_recall: float | None = None



def _parse_num_token(token: str) -> float | None:
    t = token.strip().replace(",", "")
    is_percent = t.endswith("%")
    if is_percent:
        t = t[:-1].strip()
    if not t:
        return None
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg = True
        t = t[1:-1].strip()
    if t.startswith("-"):
        neg = True
    t = t.replace("$", "").replace("€", "")
    try:
        v = float(t)
    except ValueError:
        return None
    if neg and v > 0:
        v = -v
    if is_percent:
        # Keep as human percent unit (2.9% => 2.9) to align with most TATQA answers.
        return v
    return v


def extract_numbers(text: str) -> list[float]:
    tokens = re.findall(r"[$€]?\(?-?\d[\d,]*(?:\.\d+)?\)?%?", text)
    out = []
    for tok in tokens:
        v = _parse_num_token(tok)
        if v is not None and not math.isnan(v):
            out.append(v)
    return out


def scale_variants(value: float, scale: str) -> list[float]:
    scale = (scale or "").strip().lower()
    variants = {value}
    if scale == "thousand":
        variants.add(value * 1000.0)
        variants.add(value / 1000.0)
    elif scale == "million":
        variants.add(value * 1_000_000.0)
        variants.add(value / 1_000_000.0)
    elif scale == "billion":
        variants.add(value * 1_000_000_000.0)
        variants.add(value / 1_000_000_000.0)
    elif scale == "percent":
        variants.add(value / 100.0)
        variants.add(value * 100.0)
    return list(variants)


def numeric_close(expected: float, observed: float, rel_tol: float = 0.01) -> bool:
    if expected == observed:
        return True
    abs_tol = 1e-9
    return math.isclose(expected, observed, rel_tol=rel_tol, abs_tol=abs_tol)


def evaluate_span(expected: str, extracted: str | None) -> tuple[bool, str]:
    if extracted is None:
        return False, "none"
    ne = normalize_answer(expected)
    nx = normalize_answer(extracted)
    if ne == nx:
        return True, "exact"
    if ne and (ne in nx or nx in ne):
        return True, "fuzzy"
    ew = set(ne.split())
    xw = set(nx.split())
    if ew and xw:
        overlap = len(ew & xw) / len(ew)
        if overlap >= 0.8:
            return True, "fuzzy"
    return False, "none"


def evaluate_multi_span(row: dict, extracted: str | None) -> tuple[bool, str]:
    if extracted is None:
        return False, "none"
    raw = row.get("answer_spans")
    if not raw:
        if isinstance(row.get("answer_raw"), list):
            raw = [str(x) for x in row["answer_raw"]]
        else:
            raw = [s.strip() for s in str(row.get("answer", "")).split(",") if s.strip()]
    nx = normalize_answer(extracted)
    all_found = True
    for span in raw:
        ns = normalize_answer(str(span))
        if not ns or ns not in nx:
            all_found = False
            break
    return (all_found, "multi_span_all" if all_found else "none")


def evaluate_numeric(row: dict, extracted: str | None) -> tuple[bool, str]:
    if extracted is None:
        return False, "none"

    expected_candidates: list[float] = []
    answer_raw = row.get("answer_raw")
    if isinstance(answer_raw, (int, float)):
        expected_candidates.append(float(answer_raw))
    elif isinstance(answer_raw, list):
        for v in answer_raw:
            if isinstance(v, (int, float)):
                expected_candidates.append(float(v))
            else:
                parsed = _parse_num_token(str(v))
                if parsed is not None:
                    expected_candidates.append(parsed)
    else:
        parsed = _parse_num_token(str(row.get("answer", "")))
        if parsed is not None:
            expected_candidates.append(parsed)

    if not expected_candidates:
        return evaluate_span(str(row.get("answer", "")), extracted)

    scale = row.get("scale", "")
    expected_expanded = []
    for v in expected_candidates:
        expected_expanded.extend(scale_variants(v, scale))

    observed = extract_numbers(extracted)
    if not observed:
        parsed = _parse_num_token(extracted)
        if parsed is not None:
            observed = [parsed]
    if not observed:
        return False, "none"

    for exp in expected_expanded:
        for obs in observed:
            if numeric_close(exp, obs, rel_tol=0.01):
                return True, "numeric_tol_1pct"
    return False, "none"


def evaluate_answer(row: dict, extracted: str | None) -> tuple[bool, str]:
    if row.get("is_unanswerable", False):
        return True, "unanswerable"
    answer_type = row.get("answer_type", "")
    if answer_type == "span":
        return evaluate_span(str(row.get("answer", "")), extracted)
    if answer_type == "multi-span":
        return evaluate_multi_span(row, extracted)
    if answer_type in {"arithmetic", "count"}:
        return evaluate_numeric(row, extracted)
    return evaluate_span(str(row.get("answer", "")), extracted)


def evaluate(
    file_path: str,
    eval_chunks: bool = False,
    llm_judge_factual: bool = False,
) -> dict:
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    results = []
    stats = defaultdict(int)
    stats_by_answer_type = defaultdict(lambda: {"correct": 0, "total": 0})
    stats_by_answer_from = defaultdict(lambda: {"correct": 0, "total": 0})
    chunk_stats = defaultdict(float)
    factual_judge_stats = defaultdict(float)
    enriched_rows = []

    # Factual LLM-as-a-judge evaluation
    factual_judge_results: list[dict | None] = [None] * len(rows)
    if llm_judge_factual:
        answerable_indices = [i for i, r in enumerate(rows) if not r.get("is_unanswerable", False)]
        answerable_rows = [rows[i] for i in answerable_indices]
        raw_factual = run_factual_judge(answerable_rows, {}, api_key=LLM_API_KEY)
        for idx, jr in zip(answerable_indices, raw_factual):
            factual_judge_results[idx] = jr

    for row_idx, row in enumerate(rows):
        expected = str(row.get("answer", ""))
        reasoning = row.get("reasoning_trace", "")
        is_unanswerable = row.get("is_unanswerable", False)
        extracted = None if is_unanswerable else extract_answer_from_reasoning(reasoning)
        is_correct, match_type = evaluate_answer(row, extracted)

        cited_titles = []
        expected_titles = []
        chunk_precision = None
        chunk_recall = None
        chunk_f1 = None
        if eval_chunks and not is_unanswerable:
            cited_titles = extract_cited_titles(reasoning)
            expected_titles = row.get("supporting_facts_titles", [])
            chunk_precision, chunk_recall = evaluate_chunk_citations(cited_titles, expected_titles)
            if chunk_precision is not None and chunk_recall is not None and (chunk_precision + chunk_recall) > 0:
                chunk_f1 = 2 * chunk_precision * chunk_recall / (chunk_precision + chunk_recall)
            if expected_titles:
                chunk_stats["total"] += 1
                chunk_stats["with_citations"] += 1 if cited_titles else 0
                chunk_stats["no_citations"] += 0 if cited_titles else 1
                if chunk_precision is not None:
                    chunk_stats["precision_sum"] += chunk_precision
                    chunk_stats["precision_count"] += 1
                if chunk_recall is not None:
                    chunk_stats["recall_sum"] += chunk_recall
                    chunk_stats["recall_count"] += 1
                if chunk_precision == 1.0 and chunk_recall == 1.0:
                    chunk_stats["perfect"] += 1

        # Factual LLM-as-a-judge score
        fj_score = None
        fj_justification = None
        if llm_judge_factual and not is_unanswerable:
            fjr = factual_judge_results[row_idx]
            if fjr is not None:
                fj_score = fjr["score"]
                fj_justification = fjr["justification"]
                factual_judge_stats["total"] += 1
                factual_judge_stats["score_sum"] += fj_score
                factual_judge_stats[f"score_{fj_score}"] += 1
            else:
                factual_judge_stats["failed"] += 1

        enriched = dict(row)
        enriched["eval_answer_correct"] = is_correct
        enriched["eval_answer_match_type"] = match_type
        enriched["eval_extracted_answer"] = extracted
        if eval_chunks and not is_unanswerable:
            enriched["eval_cited_titles"] = cited_titles
            enriched["eval_chunk_precision"] = chunk_precision
            enriched["eval_chunk_recall"] = chunk_recall
            enriched["eval_chunk_f1"] = chunk_f1
        if llm_judge_factual and not is_unanswerable:
            enriched["eval_factual_judge_score"] = fj_score
            enriched["eval_factual_judge_justification"] = fj_justification
        enriched_rows.append(enriched)

        results.append(
            EvalResult(
                row_id=row.get("id", ""),
                question=row.get("question", ""),
                expected_answer=expected,
                extracted_answer=extracted,
                is_correct=is_correct,
                match_type=match_type,
                cited_titles=cited_titles,
                expected_titles=expected_titles,
                chunk_precision=chunk_precision,
                chunk_recall=chunk_recall,
            )
        )

        stats["total"] += 1
        if is_correct:
            stats["correct"] += 1
            stats[f"match_{match_type}"] += 1
        else:
            stats["incorrect"] += 1
            if extracted is None:
                stats["no_answer_extracted"] += 1

        at = row.get("answer_type", "unknown")
        af = row.get("answer_from", "unknown")
        stats_by_answer_type[at]["total"] += 1
        stats_by_answer_from[af]["total"] += 1
        if is_correct:
            stats_by_answer_type[at]["correct"] += 1
            stats_by_answer_from[af]["correct"] += 1

    return {
        "results": results,
        "enriched_rows": enriched_rows,
        "stats": dict(stats),
        "stats_by_answer_type": dict(stats_by_answer_type),
        "stats_by_answer_from": dict(stats_by_answer_from),
        "chunk_stats": dict(chunk_stats),
        "factual_judge_stats": dict(factual_judge_stats),
    }


def print_report(eval_data: dict) -> None:
    stats = eval_data["stats"]
    total = stats.get("total", 0)
    correct = stats.get("correct", 0)
    acc = (correct / total * 100) if total else 0.0

    print("\n" + "=" * 60)
    print("TATQA EVALUATION REPORT")
    print("=" * 60)
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {stats.get('incorrect', 0)}")
    print(f"Accuracy: {acc:.1f}%")

    print("\nBy answer_type")
    for key, d in sorted(eval_data["stats_by_answer_type"].items()):
        kacc = (d["correct"] / d["total"] * 100) if d["total"] else 0.0
        print(f"  {key:12} {d['correct']:4}/{d['total']:4} ({kacc:5.1f}%)")

    print("\nBy answer_from")
    for key, d in sorted(eval_data["stats_by_answer_from"].items()):
        kacc = (d["correct"] / d["total"] * 100) if d["total"] else 0.0
        print(f"  {key:12} {d['correct']:4}/{d['total']:4} ({kacc:5.1f}%)")

    chunk_stats = eval_data.get("chunk_stats", {})
    if chunk_stats:
        c_total = int(chunk_stats.get("total", 0))
        p_count = int(chunk_stats.get("precision_count", 0))
        r_count = int(chunk_stats.get("recall_count", 0))
        avg_p = (chunk_stats.get("precision_sum", 0.0) / p_count * 100) if p_count else 0.0
        avg_r = (chunk_stats.get("recall_sum", 0.0) / r_count * 100) if r_count else 0.0
        avg_f1 = (2 * avg_p * avg_r / (avg_p + avg_r)) if (avg_p + avg_r) > 0 else 0.0
        print("\nChunk citation")
        print(f"  Evaluated: {c_total}")
        print(f"  Avg Precision: {avg_p:.1f}%")
        print(f"  Avg Recall: {avg_r:.1f}%")
        print(f"  Avg F1: {avg_f1:.1f}%")
        print(f"  Perfect: {int(chunk_stats.get('perfect', 0))}")

    factual_judge_stats = eval_data.get("factual_judge_stats", {})
    if factual_judge_stats and factual_judge_stats.get("total", 0) > 0:
        fj_total = int(factual_judge_stats["total"])
        fj_failed = int(factual_judge_stats.get("failed", 0))
        fj_avg = factual_judge_stats["score_sum"] / fj_total

        print("\nLLM-AS-A-JUDGE FACTUAL (answer + supporting facts)")
        print(f"  Evaluated: {fj_total}")
        if fj_failed:
            print(f"  Failed:    {fj_failed}")
        print(f"  Avg score: {fj_avg:.2f} / 5")
        print("\n  Distribution:")
        for s in range(1, 6):
            count = int(factual_judge_stats.get(f"score_{s}", 0))
            pct = (count / fj_total * 100) if fj_total else 0
            bar = "#" * int(pct / 2)
            print(f"  {s}/5: {count:4} ({pct:5.1f}%)  {bar}")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TATQA augmented dataset")
    parser.add_argument("--input", required=True, help="Augmented JSONL file")
    parser.add_argument("--eval-chunks", action="store_true")
    parser.add_argument("--llm-judge-factual", action="store_true",
                        help="Run factual LLM-as-a-judge: rates correctness given ground-truth answer and supporting facts")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Write enriched JSONL (default: <input>_evaluated.jsonl)")
    parser.add_argument("--filter", action="store_true",
                        help="Write a filtered subset keeping only rows with eval_answer_correct=True "
                             "(and eval_chunk_f1=1.0, eval_factual_judge_score=5 when evaluated). "
                             "Output: <output>_filtered.jsonl")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: file not found: {args.input}")
        return

    eval_data = evaluate(
        file_path=args.input,
        eval_chunks=args.eval_chunks,
        llm_judge_factual=args.llm_judge_factual,
    )
    print_report(eval_data)

    output_path = args.output
    if output_path is None:
        source = Path(args.input)
        output_path = str(source.parent / f"{source.stem}_evaluated.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for row in eval_data["enriched_rows"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Enriched dataset written: {output_path} ({len(eval_data['enriched_rows'])} rows)")

    if args.filter:
        if not args.llm_judge_factual:
            print("Warning: --filter without --llm-judge-factual will not apply the factual judge threshold.")
        base = Path(output_path)
        filtered_path = str(base.parent / f"{base.stem}_filtered.jsonl")
        filtered_rows = [r for r in eval_data["enriched_rows"] if filter_row(r)]
        with open(filtered_path, "w", encoding="utf-8") as f:
            for row in filtered_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Filtered dataset written: {filtered_path} ({len(filtered_rows)}/{len(eval_data['enriched_rows'])} rows kept)")


if __name__ == "__main__":
    main()
