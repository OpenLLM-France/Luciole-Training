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
import hashlib
from pathlib import Path
from collections import defaultdict
from utils import (
    JUDGE_API_KEY,
    LLM_API_URL,
    LLM_MAX_CONCURRENT,
    JUDGE_MODEL,
    load_env_key,
    normalize_answer,
    detect_refusal,
    extract_cited_titles,
    evaluate_chunk_citations,
    extract_answer_from_reasoning,
    filter_rows,
    run_judge_ordered,
    run_factual_judge_ordered,
    run_french_quality_judge_ordered,
    run_llm_judge,
    run_factual_judge,
    run_french_quality_judge,
    run_refusal_judge,
)

from formatting_sft import parse_ref_citations, row_citation_format

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


class JudgeCache:
    """Append-only JSONL of endpoint judge verdicts, keyed by (judge, model, row id, trace hash).

    evaluate_stream.py fills it while the rows are still being generated, and this
    script reuses it (--judge-cache), which also makes the evaluation resumable:
    only the missing verdicts are paid. Failed calls (None) are never cached, so
    they are retried on the next pass. Without a path the cache lives in memory.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.entries: dict[tuple, dict] = {}
        # A killed writer can leave a torn last line: skip it on load, and start the
        # next append on a fresh line so the fragment never glues onto a verdict.
        self._needs_newline = False
        if self.path and self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if not line.endswith("\n"):
                        self._needs_newline = True
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.entries[tuple(entry["key"])] = entry["result"]

    @staticmethod
    def key(judge: str, model: str, row: dict) -> tuple:
        trace = hashlib.md5((row.get("reasoning_trace") or "").encode("utf-8")).hexdigest()
        return (judge, model, str(row.get("id")), trace)

    def get(self, key: tuple) -> dict | None:
        return self.entries.get(key)

    def add(self, key: tuple, result: dict | None) -> None:
        if result is None:
            return
        self.entries[key] = result
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                if self._needs_newline:
                    f.write("\n")
                    self._needs_newline = False
                f.write(json.dumps({"key": list(key), "result": result}, ensure_ascii=False) + "\n")


# Endpoint judges: name -> (runner, runs on unanswerable rows?, extra runner kwargs).
# The refusal judge is the only one that looks at unanswerable rows.
ENDPOINT_JUDGES = {
    "relevance": (run_llm_judge, False, {}),
    "factual": (run_factual_judge, False, {"sf_lookup": {}}),
    "french_quality": (run_french_quality_judge, False, {}),
    "refusal": (run_refusal_judge, True, {}),
}


def judge_rows(rows: list[dict], judge: str, cache: JudgeCache, **kwargs) -> list[dict | None]:
    """Verdicts of one endpoint judge aligned to `rows`: cached ones reused, the rest computed and cached."""
    run_fn, _, extra = ENDPOINT_JUDGES[judge]
    keys = [JudgeCache.key(judge, kwargs["model"], r) for r in rows]
    results = [cache.get(k) for k in keys]
    todo = [i for i, res in enumerate(results) if res is None]
    if len(todo) < len(rows):
        print(f"\n{judge} judge: {len(rows) - len(todo)}/{len(rows)} verdicts reused from the judge cache")
    if todo:
        raw = run_fn([rows[i] for i in todo], **extra, **kwargs)
        for i, jr in zip(todo, raw):
            results[i] = jr
            cache.add(keys[i], jr)
    return results


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


def _collect_endpoint_judge(
    rows: list[dict], flag: bool, judge: str, cache: JudgeCache, **kwargs,
) -> list[dict | None]:
    """Endpoint judge results aligned to `rows`, on the rows that judge applies to."""
    results: list[dict | None] = [None] * len(rows)
    if not flag:
        return results
    on_unanswerable = ENDPOINT_JUDGES[judge][1]
    indices = [i for i, r in enumerate(rows) if bool(r.get("is_unanswerable", False)) == on_unanswerable]
    raw = judge_rows([rows[i] for i in indices], judge, cache, **kwargs)
    for idx, jr in zip(indices, raw):
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
             llm_judge_factual_openrouter: bool = False,
             llm_judge_french_quality: bool = False,
             llm_judge_refusal: bool = False,
             no_answer_match: bool = False,
             judge_model: str = JUDGE_MODEL,
             judge_concurrency: int = LLM_MAX_CONCURRENT,
             judge_cache: str | None = None) -> dict:
    """Evaluate all entries in the augmented dataset."""
    stats = defaultdict(int)
    stats_by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    stats_by_level = defaultdict(lambda: {"correct": 0, "total": 0})

    # Pre-load all rows to collect IDs for chunk eval
    rows = []
    with open(file_path, "r") as f:
        for line in f:
            rows.append(json.loads(line))

    # Endpoint judges (not the OpenRouter ones) share the model, the concurrency cap
    # and the verdict cache.
    judge_kwargs = {"api_key": JUDGE_API_KEY, "model": judge_model, "concurrency": judge_concurrency}
    cache = JudgeCache(judge_cache)
    judge_results = _collect_endpoint_judge(rows, llm_judge, "relevance", cache, **judge_kwargs)
    judge_stats: defaultdict = defaultdict(float)
    factual_judge_results = _collect_endpoint_judge(rows, llm_judge_factual, "factual", cache, **judge_kwargs)
    factual_judge_stats: defaultdict = defaultdict(float)
    or_judge_results = _collect_judge_results(rows, llm_judge_openrouter, run_llm_judge_openrouter)
    or_judge_stats: defaultdict = defaultdict(float)
    or_factual_judge_results = _collect_judge_results(rows, llm_judge_factual_openrouter, run_factual_judge_openrouter, sf_lookup={})
    or_factual_judge_stats: defaultdict = defaultdict(float)
    french_quality_judge_results = _collect_endpoint_judge(rows, llm_judge_french_quality, "french_quality", cache, **judge_kwargs)
    french_quality_judge_stats: defaultdict = defaultdict(float)

    # Refusal judge runs on UNANSWERABLE rows only (opposite of the other judges).
    refusal_judge_results = _collect_endpoint_judge(rows, llm_judge_refusal, "refusal", cache, **judge_kwargs)
    refusal_judge_stats: defaultdict = defaultdict(float)

    chunk_stats = defaultdict(float)
    enriched_rows = []

    for row_idx, row in enumerate(rows):
        is_unanswerable = row.get("is_unanswerable", False)
        expected = row.get("answer", "")
        reasoning = row.get("reasoning_trace", "")

        model_refused = None
        refusal_judge_justification = None
        if is_unanswerable:
            extracted = None
            # Keep only refusals: did the model actually refuse the insufficient context?
            # Prefer the LLM refusal judge when available; fall back to regex detection.
            jr = refusal_judge_results[row_idx]
            if jr is not None:
                model_refused = jr["refused"]
                refusal_judge_justification = jr["justification"]
                refusal_judge_stats["total"] += 1
                refusal_judge_stats["refused" if model_refused else "hallucinated"] += 1
            elif llm_judge_refusal:
                refusal_judge_stats["failed"] += 1
                model_refused = detect_refusal(reasoning)
            else:
                model_refused = detect_refusal(reasoning)
            is_correct = model_refused
            match_type = "refused" if model_refused else "hallucinated"
        elif no_answer_match:
            # Unified-prompt traces carry no "Final Answer" line to extract:
            # correctness is gated by the factual judge (+ citation F1), not by
            # exact string match, so we skip answer extraction here.
            extracted = None
            is_correct = None
            match_type = "not_evaluated"
        else:
            extracted = extract_answer_from_reasoning(reasoning)
            is_correct, match_type = check_answer(expected, extracted)

        # Chunk citation evaluation
        cited_titles = []
        expected_titles = []
        chunk_precision = None
        chunk_recall = None
        chunk_f1 = None

        # Ref-format rows: every <ref id="N"> must point at a document whose text
        # holds the excerpt verbatim; refusals are checked too (they may quote).
        ref_count = ref_invalid = None
        is_ref = row_citation_format(row) == "ref"
        if is_ref:
            ref_titles, ref_count, ref_invalid = parse_ref_citations(
                reasoning, row.get("context", ""),
                end_of_completion=bool(row.get("citation_end_of_completion")),
            )
            stats["ref_rows"] += 1
            stats["ref_tags"] += ref_count
            stats["ref_tags_invalid"] += ref_invalid
            if ref_invalid:
                stats["ref_rows_invalid"] += 1

        if eval_chunks and not is_unanswerable:
            cited_titles = ref_titles if is_ref else extract_cited_titles(reasoning)
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
        fq_score, fq_justification = _extract_score(llm_judge_french_quality, is_unanswerable, french_quality_judge_results, row_idx, french_quality_judge_stats)

        # Build enriched row: original data + eval columns
        enriched = dict(row)
        enriched["eval_answer_correct"] = is_correct
        enriched["eval_answer_match_type"] = match_type
        enriched["eval_extracted_answer"] = extracted
        if is_unanswerable:
            enriched["eval_model_refused"] = model_refused
            if refusal_judge_justification is not None:
                enriched["eval_refusal_judge_justification"] = refusal_judge_justification
        if is_ref:
            enriched["eval_ref_count"] = ref_count
            enriched["eval_ref_invalid"] = ref_invalid
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
            (llm_judge_french_quality, fq_score, fq_justification, "eval_french_quality_judge_score", "eval_french_quality_judge_justification"),
        ]:
            if flag and not is_unanswerable:
                enriched[score_key] = score_val
                enriched[just_key] = just_val
        enriched_rows.append(enriched)

        # match_type "not_evaluated" => exact-match accuracy is intentionally not
        # computed (unified-prompt format); skip it in every accuracy tally.
        if match_type != "not_evaluated":
            stats["total"] += 1
            if is_correct:
                stats["correct"] += 1
                stats[f"match_{match_type}"] += 1
            else:
                stats["incorrect"] += 1
                if extracted is None and not is_unanswerable:
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
        "french_quality_judge_stats": dict(french_quality_judge_stats),
        "refusal_judge_stats": dict(refusal_judge_stats),
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
    french_quality_judge_stats = eval_data.get("french_quality_judge_stats", {})
    refusal_judge_stats = eval_data.get("refusal_judge_stats", {})

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

    if stats.get("ref_rows"):
        print(f"\nREF CITATIONS (<ref id=\"N\">, verbatim excerpt in document N)")
        print(f"   Rows:            {stats['ref_rows']}")
        print(f"   Tags:            {stats.get('ref_tags', 0)}")
        print(f"   Invalid tags:    {stats.get('ref_tags_invalid', 0)}")
        print(f"   Rows w/ invalid: {stats.get('ref_rows_invalid', 0)} (dropped by --filter)")

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
    _print_judge_section("LLM-AS-A-JUDGE FRENCH QUALITY (language quality)", french_quality_judge_stats)

    if refusal_judge_stats:
        rj_total = int(refusal_judge_stats.get("total", 0))
        rj_refused = int(refusal_judge_stats.get("refused", 0))
        rj_hallu = int(refusal_judge_stats.get("hallucinated", 0))
        rj_failed = int(refusal_judge_stats.get("failed", 0))
        rate = (rj_refused / rj_total * 100) if rj_total else 0
        print("\nLLM-AS-A-JUDGE REFUSAL (unanswerable rows: refuse vs answer)")
        print(f"   Judged:       {rj_total}")
        if rj_failed:
            print(f"   Failed:       {rj_failed}")
        print(f"   Refused:      {rj_refused} ({rate:.1f}%)")
        print(f"   Hallucinated: {rj_hallu}")

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
    parser.add_argument("--llm-judge-french-quality", action="store_true",
                        help="Run French quality LLM-as-a-judge: rates language quality of French reasoning traces on 1-5 scale")
    parser.add_argument("--llm-judge-refusal", action="store_true",
                        help="Run the binary refusal LLM-as-a-judge on unanswerable rows (refuse vs answer). "
                             "Its verdict sets eval_model_refused (used by --filter); falls back to regex detection.")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Write enriched JSONL with eval columns added to each row")
    parser.add_argument("--export", type=str, help="Export aggregate stats to JSON file")
    parser.add_argument("--filter", action="store_true",
                        help="Write a filtered subset keeping only rows with eval_answer_correct=True, "
                             "eval_chunk_f1=1.0, eval_factual_judge_score=5 (plus unanswerable). "
                             "Requires --llm-judge-factual or --llm-judge-factual-openrouter. "
                             "Output: <output>_filtered.jsonl")
    parser.add_argument("--no-answer-match", action="store_true",
                        help="Traces have no 'Final Answer' line (unified-prompt format): skip exact-match "
                             "answer extraction and gate correctness on the factual judge (+ citation F1) only.")
    parser.add_argument("--no-keep-pairs", action="store_true",
                        help="Filter each row independently. By default an unanswerable row and the "
                             "answerable row it was cloned from survive the filter together or not at "
                             "all, so every refusal keeps the minimal pair it was built to form.")
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL,
                        help=f"Model for the endpoint judges (default: LLM_JUDGE_MODEL, else LLM_MODEL, {JUDGE_MODEL})")
    parser.add_argument("--judge-concurrency", type=int, default=LLM_MAX_CONCURRENT,
                        help=f"Concurrent judge calls (default: {LLM_MAX_CONCURRENT}); judges run one after another")
    parser.add_argument("--judge-cache", type=str, default=None,
                        help="JSONL of endpoint judge verdicts to reuse and extend (filled by "
                             "evaluate_stream.py during generation; also makes this run resumable)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        parser.error(f"input file not found: {args.input}")
    if Path(args.input).stat().st_size == 0:
        parser.error(f"input file is empty: {args.input}")

    print(f"Evaluating: {args.input}")
    eval_data = evaluate(args.input, eval_chunks=args.eval_chunks,
                         llm_judge=args.llm_judge,
                         llm_judge_factual=args.llm_judge_factual,
                         llm_judge_openrouter=args.llm_judge_openrouter,
                         llm_judge_factual_openrouter=args.llm_judge_factual_openrouter,
                         llm_judge_french_quality=args.llm_judge_french_quality,
                         llm_judge_refusal=args.llm_judge_refusal,
                         no_answer_match=args.no_answer_match,
                         judge_model=args.judge_model,
                         judge_concurrency=args.judge_concurrency,
                         judge_cache=args.judge_cache)
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
        filtered_rows, filter_stats = filter_rows(
            eval_data["enriched_rows"],
            keep_pairs=not args.no_keep_pairs,
            require_answer_match=not args.no_answer_match,
        )
        with open(filtered_path, "w", encoding="utf-8") as f:
            for row in filtered_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Filtered dataset written to: {filtered_path} ({len(filtered_rows)}/{len(eval_data['enriched_rows'])} rows kept)")
        if filter_stats["pairs_dropped"]:
            print(f"  ({filter_stats['pairs_dropped']} answerable/unanswerable pairs dropped whole, "
                  f"{filter_stats['rows_dropped_by_pairing']} rows)")

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
        if args.llm_judge_french_quality:
            export_data["french_quality_judge_stats"] = eval_data["french_quality_judge_stats"]
        if args.llm_judge_refusal:
            export_data["refusal_judge_stats"] = eval_data["refusal_judge_stats"]
        with open(args.export, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"Aggregate stats exported to: {args.export}")


if __name__ == "__main__":
    main()
