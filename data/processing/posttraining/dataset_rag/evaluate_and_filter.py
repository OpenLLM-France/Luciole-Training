"""
Evaluate the accuracy of generated reasoning traces by comparing
the extracted answer to the expected answer from the dataset.

Unanswerable examples are treated as correct by construction.

Also evaluates chunk citation accuracy: whether the chunks cited
in the reasoning trace (via ##Cite "title" ##) match the ground-truth
supporting facts from the original HotpotQA dataset.

Optionally runs an LLM-as-a-judge factual evaluation where Mistral rates
the relevance of each reasoning trace on a 1-5 scale knowing the answer and the expected documents to cite.
"""

import json
import asyncio
from pathlib import Path
from collections import defaultdict
from utils import (
    LLM_API_URL,
    LLM_MAX_CONCURRENT,
    LLM_MODEL,
    load_env_key,
    normalize_answer,
    extract_cited_titles,
    evaluate_chunk_citations,
    extract_answer_from_reasoning,
    filter_row,
    run_judge_ordered,
    run_factual_judge_ordered,
    run_llm_judge,
    run_factual_judge,
)

LLM_API_KEY = load_env_key("LLM_API_KEY", base_dir=Path(__file__).parent)

# OpenRouter configuration
OPENROUTER_API_KEY = load_env_key("OPENROUTER_API", base_dir=Path(__file__).parent)
OPENROUTER_API_URL = load_env_key("OPENROUTER_API_URL", base_dir=Path(__file__).parent) or ""
OPENROUTER_JUDGE_MODEL = "openai/gpt-oss-120b"


def run_llm_judge_openrouter(rows: list[dict]) -> list[dict | None]:
    """Synchronous wrapper to run the LLM judge via OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API is required for --llm-judge-openrouter (set in .env or environment)")
    print(f"\nRunning LLM-as-a-judge via OpenRouter ({OPENROUTER_JUDGE_MODEL}) on {len(rows)} rows...")
    return asyncio.run(run_judge_ordered(
        rows,
        api_url=OPENROUTER_API_URL,
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_JUDGE_MODEL,
        desc="OpenRouter judge",
    ))


def run_factual_judge_openrouter(rows: list[dict], sf_lookup: dict[str, set[str]]) -> list[dict | None]:
    """Synchronous wrapper to run the factual LLM judge via OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API is required for --llm-judge-factual-openrouter (set in .env or environment)")
    print(f"\nRunning factual LLM-as-a-judge via OpenRouter ({OPENROUTER_JUDGE_MODEL}) on {len(rows)} rows...")
    return asyncio.run(run_factual_judge_ordered(
        rows, sf_lookup,
        api_url=OPENROUTER_API_URL,
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_JUDGE_MODEL,
        desc="OpenRouter factual judge",
    ))


OUTPUT_FILE = Path(__file__).parent / "hotpotqa_augmented_evaluated.jsonl"


def _collect_judge_results(
    rows: list[dict],
    flag: bool,
    run_fn,
    **kwargs,
) -> list[dict | None]:
    """Run a judge function on answerable rows only, returning results aligned to `rows`."""
    results: list[dict | None] = [None] * len(rows)
    if not flag:
        return results
    answerable_indices = [i for i, r in enumerate(rows) if not r.get("is_unanswerable", False)]
    answerable_rows = [rows[i] for i in answerable_indices]
    raw = run_fn(answerable_rows, **kwargs)
    for idx, jr in zip(answerable_indices, raw):
        results[idx] = jr
    return results


def _extract_score(
    flag: bool,
    is_unanswerable: bool,
    results: list[dict | None],
    row_idx: int,
    stats: defaultdict,
) -> tuple[int | None, str | None]:
    """Extract score/justification from judge results and update running stats."""
    if not flag or is_unanswerable:
        return None, None
    jr = results[row_idx]
    if jr is not None:
        score = jr["score"]
        stats["total"] += 1
        stats["score_sum"] += score
        stats[f"score_{score}"] += 1
        return score, jr["justification"]
    stats["failed"] += 1
    return None, None


def normalize_title(title: str) -> str:
    """Normalize a chunk title for comparison."""
    return title.strip().lower()


def check_answer(expected: str, extracted: str | None) -> tuple[bool, str]:
    """Check if extracted answer matches expected answer."""
    if extracted is None:
        return False, "none"

    norm_expected = normalize_answer(expected)
    norm_extracted = normalize_answer(extracted)

    if norm_expected == norm_extracted:
        return True, "exact"

    if norm_expected in norm_extracted or norm_extracted in norm_expected:
        return True, "fuzzy"

    expected_words = set(norm_expected.split())
    extracted_words = set(norm_extracted.split())

    if expected_words and extracted_words:
        overlap = len(expected_words & extracted_words) / len(expected_words)
        if overlap >= 0.8:
            return True, "fuzzy"

    return False, "none"


def evaluate(file_path: str = OUTPUT_FILE, eval_chunks: bool = False,
             llm_judge: bool = False,
             llm_judge_factual: bool = False,
             llm_judge_openrouter: bool = False,
             llm_judge_factual_openrouter: bool = False) -> dict:
    """Evaluate all entries in the augmented dataset."""
    stats = defaultdict(int)
    stats_by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    stats_by_level = defaultdict(lambda: {"correct": 0, "total": 0})

    # Pre-load all rows to collect IDs for chunk eval
    rows = []
    with open(file_path, "r") as f:
        for line in f:
            rows.append(json.loads(line))

    judge_results = _collect_judge_results(rows, llm_judge, run_llm_judge, api_key=LLM_API_KEY)
    judge_stats: defaultdict = defaultdict(float)
    factual_judge_results = _collect_judge_results(rows, llm_judge_factual, run_factual_judge, sf_lookup={}, api_key=LLM_API_KEY)
    factual_judge_stats: defaultdict = defaultdict(float)
    or_judge_results = _collect_judge_results(rows, llm_judge_openrouter, run_llm_judge_openrouter)
    or_judge_stats: defaultdict = defaultdict(float)
    or_factual_judge_results = _collect_judge_results(rows, llm_judge_factual_openrouter, run_factual_judge_openrouter, sf_lookup={})
    or_factual_judge_stats: defaultdict = defaultdict(float)

    chunk_stats = defaultdict(float)
    enriched_rows = []

    for row_idx, row in enumerate(rows):
        is_unanswerable = row.get("is_unanswerable", False)
        expected = row.get("answer", "")
        reasoning = row.get("reasoning_trace", "")

        if is_unanswerable:
            extracted = None
            is_correct = True
            match_type = "unanswerable"
        else:
            extracted = extract_answer_from_reasoning(reasoning)
            is_correct, match_type = check_answer(expected, extracted)

        # Chunk citation evaluation
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
                if cited_titles:
                    chunk_stats["with_citations"] += 1
                else:
                    chunk_stats["no_citations"] += 1
                if chunk_precision is not None:
                    chunk_stats["precision_sum"] += chunk_precision
                    chunk_stats["precision_count"] += 1
                if chunk_recall is not None:
                    chunk_stats["recall_sum"] += chunk_recall
                    chunk_stats["recall_count"] += 1
                # Perfect scores
                if chunk_precision == 1.0 and chunk_recall == 1.0:
                    chunk_stats["perfect"] += 1
                # Track extra/missed citations
                cited_norm = {normalize_title(t) for t in cited_titles}
                expected_norm = {normalize_title(t) for t in expected_titles}
                extra = cited_norm - expected_norm
                missed = expected_norm - cited_norm
                chunk_stats["extra_citations"] += len(extra)
                chunk_stats["missed_citations"] += len(missed)

        j_score, j_justification = _extract_score(llm_judge, is_unanswerable, judge_results, row_idx, judge_stats)
        fj_score, fj_justification = _extract_score(llm_judge_factual, is_unanswerable, factual_judge_results, row_idx, factual_judge_stats)
        or_j_score, or_j_justification = _extract_score(llm_judge_openrouter, is_unanswerable, or_judge_results, row_idx, or_judge_stats)
        or_fj_score, or_fj_justification = _extract_score(llm_judge_factual_openrouter, is_unanswerable, or_factual_judge_results, row_idx, or_factual_judge_stats)

        # Build enriched row: original data + eval columns
        enriched = dict(row)
        enriched["eval_answer_correct"] = is_correct
        enriched["eval_answer_match_type"] = match_type
        enriched["eval_extracted_answer"] = extracted
        if eval_chunks and not is_unanswerable:
            enriched["eval_cited_titles"] = cited_titles
            enriched["eval_chunk_precision"] = chunk_precision
            enriched["eval_chunk_recall"] = chunk_recall
            enriched["eval_chunk_f1"] = chunk_f1
        for flag, score_val, just_val, score_key, just_key in [
            (llm_judge, j_score, j_justification, "eval_judge_score", "eval_judge_justification"),
            (llm_judge_factual, fj_score, fj_justification, "eval_factual_judge_score", "eval_factual_judge_justification"),
            (llm_judge_openrouter, or_j_score, or_j_justification, "eval_openrouter_judge_score", "eval_openrouter_judge_justification"),
            (llm_judge_factual_openrouter, or_fj_score, or_fj_justification, "eval_openrouter_factual_judge_score", "eval_openrouter_factual_judge_justification"),
        ]:
            if flag and not is_unanswerable:
                enriched[score_key] = score_val
                enriched[just_key] = just_val
        enriched_rows.append(enriched)

        stats["total"] += 1
        if is_correct:
            stats["correct"] += 1
            stats[f"match_{match_type}"] += 1
        else:
            stats["incorrect"] += 1
            if extracted is None:
                stats["no_answer_extracted"] += 1

        q_type = row.get("type", "unknown")
        stats_by_type[q_type]["total"] += 1
        if is_correct:
            stats_by_type[q_type]["correct"] += 1

        level = row.get("level", "unknown")
        stats_by_level[level]["total"] += 1
        if is_correct:
            stats_by_level[level]["correct"] += 1

    return {
        "enriched_rows": enriched_rows,
        "stats": dict(stats),
        "stats_by_type": dict(stats_by_type),
        "stats_by_level": dict(stats_by_level),
        "chunk_stats": dict(chunk_stats),
        "judge_stats": dict(judge_stats),
        "factual_judge_stats": dict(factual_judge_stats),
        "or_judge_stats": dict(or_judge_stats),
        "or_factual_judge_stats": dict(or_factual_judge_stats),
    }


def _print_judge_section(title: str, stats: dict) -> None:
    if not stats:
        return
    total = int(stats.get("total", 0))
    failed = int(stats.get("failed", 0))
    avg = (stats["score_sum"] / total) if total else 0
    print(f"\n{title}")
    print(f"   Evaluated:  {total}")
    if failed:
        print(f"   Failed:     {failed}")
    print(f"   Avg score:  {avg:.2f} / 5")
    print(f"\n   Distribution:")
    for s in range(1, 6):
        count = int(stats.get(f"score_{s}", 0))
        pct = (count / total * 100) if total else 0
        bar = "#" * int(pct / 2)
        print(f"   {s}/5: {count:4} ({pct:5.1f}%)  {bar}")


def print_report(eval_data: dict):
    """Print evaluation report."""
    stats = eval_data["stats"]
    stats_by_type = eval_data["stats_by_type"]
    stats_by_level = eval_data["stats_by_level"]
    chunk_stats = eval_data.get("chunk_stats", {})
    judge_stats = eval_data.get("judge_stats", {})
    factual_judge_stats = eval_data.get("factual_judge_stats", {})
    or_judge_stats = eval_data.get("or_judge_stats", {})
    or_factual_judge_stats = eval_data.get("or_factual_judge_stats", {})

    total = stats.get("total", 0)
    correct = stats.get("correct", 0)
    accuracy = (correct / total * 100) if total > 0 else 0

    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)

    print(f"\nOVERALL ACCURACY")
    print(f"   Total:     {total}")
    print(f"   Correct:   {correct}")
    print(f"   Incorrect: {stats.get('incorrect', 0)}")
    print(f"   Accuracy:  {accuracy:.1f}%")

    print(f"\nMATCH TYPES")
    print(f"   Exact matches:   {stats.get('match_exact', 0)}")
    print(f"   Fuzzy matches:   {stats.get('match_fuzzy', 0)}")
    print(f"   Unanswerable:    {stats.get('match_unanswerable', 0)}")
    print(f"   No answer found: {stats.get('no_answer_extracted', 0)}")

    print(f"\nBY QUESTION TYPE")
    for q_type, data in sorted(stats_by_type.items()):
        acc = (data["correct"] / data["total"] * 100) if data["total"] > 0 else 0
        print(f"   {q_type:15} {data['correct']:4}/{data['total']:4} ({acc:5.1f}%)")

    print(f"\nBY DIFFICULTY LEVEL")
    for level, data in sorted(stats_by_level.items()):
        acc = (data["correct"] / data["total"] * 100) if data["total"] > 0 else 0
        print(f"   {level:15} {data['correct']:4}/{data['total']:4} ({acc:5.1f}%)")

    if chunk_stats:
        c_total = int(chunk_stats.get("total", 0))
        c_with = int(chunk_stats.get("with_citations", 0))
        c_no = int(chunk_stats.get("no_citations", 0))
        c_perfect = int(chunk_stats.get("perfect", 0))
        c_extra = int(chunk_stats.get("extra_citations", 0))
        c_missed = int(chunk_stats.get("missed_citations", 0))
        c_not_found = int(chunk_stats.get("id_not_found", 0))

        p_count = chunk_stats.get("precision_count", 0)
        r_count = chunk_stats.get("recall_count", 0)
        avg_p = (chunk_stats["precision_sum"] / p_count * 100) if p_count else 0
        avg_r = (chunk_stats["recall_sum"] / r_count * 100) if r_count else 0
        avg_f1 = (2 * avg_p * avg_r / (avg_p + avg_r)) if (avg_p + avg_r) > 0 else 0

        print(f"\nCHUNK CITATION EVALUATION")
        print(f"   Evaluated (answerable w/ ground truth): {c_total}")
        print(f"   With citations:     {c_with}")
        print(f"   Without citations:  {c_no}")
        if c_not_found:
            print(f"   ID not in HotpotQA: {c_not_found}")
        print(f"\n   Avg Precision:  {avg_p:5.1f}%  (cited chunks that are relevant)")
        print(f"   Avg Recall:     {avg_r:5.1f}%  (relevant chunks that were cited)")
        print(f"   Avg F1:         {avg_f1:5.1f}%")
        print(f"\n   Perfect (P=R=100%): {c_perfect}/{c_total} ({c_perfect/c_total*100:.1f}%)" if c_total else "")
        print(f"   Extra citations (distractors cited): {c_extra}")
        print(f"   Missed citations (relevant not cited): {c_missed}")

    _print_judge_section("LLM-AS-A-JUDGE (reasoning relevance)", judge_stats)
    _print_judge_section("LLM-AS-A-JUDGE FACTUAL (answer + supporting facts)", factual_judge_stats)
    _print_judge_section(f"OPENROUTER LLM-AS-A-JUDGE (reasoning relevance, {OPENROUTER_JUDGE_MODEL})", or_judge_stats)
    _print_judge_section(f"OPENROUTER LLM-AS-A-JUDGE FACTUAL (answer + supporting facts, {OPENROUTER_JUDGE_MODEL})", or_factual_judge_stats)

    print("\n" + "="*60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate answer accuracy in augmented dataset (v2)")
    parser.add_argument("--input", type=str, default=OUTPUT_FILE, help="Path to augmented JSONL file")
    parser.add_argument("--eval-chunks", action="store_true",
                        help="Evaluate chunk citation accuracy (requires loading original HotpotQA)")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Run LLM-as-a-judge: Mistral rates reasoning relevance on 1-5 scale")
    parser.add_argument("--llm-judge-factual", action="store_true",
                        help="Run factual LLM-as-a-judge: rates correctness given ground-truth answer and supporting facts")
    parser.add_argument("--llm-judge-openrouter", action="store_true",
                        help=f"Run LLM-as-a-judge via OpenRouter ({OPENROUTER_JUDGE_MODEL}): rates reasoning relevance on 1-5 scale")
    parser.add_argument("--llm-judge-factual-openrouter", action="store_true",
                        help=f"Run factual LLM-as-a-judge via OpenRouter ({OPENROUTER_JUDGE_MODEL}): rates correctness given ground-truth")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Write enriched JSONL with eval columns added to each row")
    parser.add_argument("--export", type=str, help="Export aggregate stats to JSON file")
    parser.add_argument("--filter", action="store_true",
                        help="Write a filtered subset keeping only rows with eval_answer_correct=True, "
                             "eval_chunk_f1=1.0, eval_factual_judge_score=5 (plus unanswerable). "
                             "Requires --llm-judge-factual or --llm-judge-factual-openrouter. "
                             "Output: <output>_filtered.jsonl")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        return

    print(f"Evaluating: {args.input}")
    eval_data = evaluate(args.input, eval_chunks=args.eval_chunks,
                         llm_judge=args.llm_judge,
                         llm_judge_factual=args.llm_judge_factual,
                         llm_judge_openrouter=args.llm_judge_openrouter,
                         llm_judge_factual_openrouter=args.llm_judge_factual_openrouter)
    print_report(eval_data)

    with open(args.output, "w", encoding="utf-8") as f:
        for enriched_row in eval_data["enriched_rows"]:
            f.write(json.dumps(enriched_row, ensure_ascii=False) + "\n")
    print(f"\nEnriched dataset written to: {args.output} ({len(eval_data['enriched_rows'])} rows)")

    if args.filter:
        if not (args.llm_judge_factual or args.llm_judge_factual_openrouter):
            print("Warning: --filter requires --llm-judge-factual or --llm-judge-factual-openrouter "
                  "(eval_factual_judge_score will be missing, no rows will pass the filter).")
        base = Path(args.output)
        filtered_path = str(base.parent / f"{base.stem}_filtered.jsonl")
        filtered_rows = [r for r in eval_data["enriched_rows"] if filter_row(r)]
        with open(filtered_path, "w", encoding="utf-8") as f:
            for row in filtered_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Filtered dataset written to: {filtered_path} ({len(filtered_rows)}/{len(eval_data['enriched_rows'])} rows kept)")

    if args.export:
        export_data = {
            "stats": eval_data["stats"],
            "stats_by_type": eval_data["stats_by_type"],
            "stats_by_level": eval_data["stats_by_level"],
        }
        if args.eval_chunks:
            export_data["chunk_stats"] = eval_data["chunk_stats"]
        if args.llm_judge:
            export_data["judge_stats"] = eval_data["judge_stats"]
        if args.llm_judge_factual:
            export_data["factual_judge_stats"] = eval_data["factual_judge_stats"]
        if args.llm_judge_openrouter:
            export_data["or_judge_stats"] = eval_data["or_judge_stats"]
        if args.llm_judge_factual_openrouter:
            export_data["or_factual_judge_stats"] = eval_data["or_factual_judge_stats"]
        with open(args.export, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"Aggregate stats exported to: {args.export}")


if __name__ == "__main__":
    main()
