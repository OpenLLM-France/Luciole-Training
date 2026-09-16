#!/usr/bin/env python3
"""DPO prompt pool from HotpotQA train questions that did NOT survive the v4 synthetic pipeline.

Why: the production model is trained on 100 % of hotpotqa_v4, so the current pool
(v4 minus the ablation SFT splits) would be seen in SFT. The v4 run augmented the whole
HotpotQA train (90 447 answerable rows + 19 867 unanswerable clones) and the filter kept
28 386 rows: 65 421 questions have no row at all in v4. Their prompts are built by the
same pipeline (same distractors, same drop_gold / partial_hop clones), so the prompt
distribution is unchanged, and no SFT model has seen them.

Filter of evaluate_and_filter.py (utils.filter_row / filter_rows), hence the drop reasons:
  answerable    judge_lt5    factual judge < 5 (hard question, or doubtful gold)
                cit_f1_lt1   judge 5 but citation F1 < 1 (generator citations, question sound)
                pair         passed, but its unanswerable twin failed (pairs kept whole)
  unanswerable  refused      generator refused, dropped because the answerable twin failed
                answered     generator answered: often answerable in disguise

Defaults keep the safe part: answerable rows with judge 5, unanswerable clones on which
the generator refused and whose twin has judge 5. The SFT target of these rows is the
failed trace: build pairs with --no-sft-fallback.

Two stages, so the 675 MB evaluated file never leaves the Airflow host:

  # 1) on the Airflow host, stdlib python
  python3 build_dpo_prompts_heldout.py extract \
      --evaluated .../v4_backfill_pairs/hotpotqa_en_augmented_evaluated.jsonl \
      --filtered  .../v4_backfill_pairs/hotpotqa_en_augmented_evaluated_filtered.jsonl \
      --out heldout_rows.jsonl
  # 2) locally, dataset_rag venv (formatting_sft imports utils -> aiohttp)
  venv/bin/python dpo/build_dpo_prompts_heldout.py format --rows heldout_rows.jsonl \
      --out dpo/data/prompts_hpv4heldout.jsonl
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

UNANS_SUFFIX = "_unanswerable"
KEEP_FIELDS = ("id", "question", "context", "answer", "reasoning_trace", "type", "level",
               "is_unanswerable", "grounded_refusal", "citation_end_of_completion",
               "supporting_facts_titles", "chunks_relevant", "chunks_total",
               "eval_factual_judge_score", "eval_chunk_f1", "eval_model_refused", "eval_answer_match_type")


def base_id(row_id):
    s = str(row_id)
    return s[: -len(UNANS_SUFFIX)] if s.endswith(UNANS_SUFFIX) else s


def drop_reason(row, twin_passes):
    if row.get("is_unanswerable"):
        return "refused" if row.get("eval_model_refused") is not False else "answered"
    if row.get("eval_factual_judge_score") != 5:
        return "judge_lt5"
    if row.get("eval_chunk_f1") != 1.0:
        return "cit_f1_lt1"
    return "pair" if twin_passes is False else "unknown"


def take(items, n, frac, rng, key):
    """n items; if frac is set, that share (at most) has key(item)=True, the rest False."""
    if frac is None:
        return items[:n]
    yes = [x for x in items if key(x)]
    no = [x for x in items if not key(x)]
    n_yes = min(len(yes), round(n * frac))
    out = yes[:n_yes] + no[: n - n_yes]
    rng.shuffle(out)
    return out


def extract(args):
    rng = random.Random(args.seed)
    v4_questions, v4_ids = set(), set()
    for line in open(args.filtered, encoding="utf-8"):
        r = json.loads(line)
        v4_questions.add(r["question"].strip())
        v4_ids.add(base_id(r["id"]))

    groups = collections.defaultdict(dict)  # base id -> {"A": row, "U": row}
    stats = collections.Counter()
    for line in open(args.evaluated, encoding="utf-8"):
        r = json.loads(line)
        b = base_id(r["id"])
        if b in v4_ids or r["question"].strip() in v4_questions:
            stats["rows_in_v4"] += 1
            continue
        side = "U" if r.get("is_unanswerable") else "A"
        if side in groups[b]:
            stats["duplicate_id_side"] += 1
            continue
        groups[b][side] = {k: r[k] for k in KEEP_FIELDS if k in r}

    # questions whose text occurs under several ids are ambiguous: skip them
    text_count = collections.Counter(g["A"]["question"].strip() for g in groups.values() if "A" in g)
    twins, answerable_only = [], []
    for b, g in groups.items():
        a, u = g.get("A"), g.get("U")
        if a is None:
            stats["unanswerable_without_twin"] += 1
            continue
        if text_count[a["question"].strip()] > 1:
            stats["ambiguous_question_text"] += 1
            continue
        a_ok = a.get("eval_factual_judge_score") == 5
        u_ok = u is not None and u.get("eval_model_refused") is not False
        a["heldout_drop_reason"] = drop_reason(a, u_ok if u is not None else None)
        stats[f"A/{a['heldout_drop_reason']}"] += 1
        if u is not None:
            u["heldout_drop_reason"] = drop_reason(u, None)
            stats[f"U/{u['heldout_drop_reason']}"] += 1
        if a["heldout_drop_reason"] == "judge_lt5" and not args.include_judge_lt5:
            continue
        if u is not None and (u_ok or args.include_unans_answered):
            twins.append((a, u))
        elif a_ok or args.include_judge_lt5:
            answerable_only.append(a)
    stats["eligible_twins"] = len(twins)
    stats["eligible_answerable_only"] = len(answerable_only)

    stats["eligible_twins_comparison"] = sum(a["type"] == "comparison" for a, _ in twins)
    stats["eligible_answerable_only_comparison"] = sum(a["type"] == "comparison" for a in answerable_only)
    rng.shuffle(twins)
    is_cmp = lambda a: a["type"] == "comparison"  # noqa: E731
    drawn = take(twins, args.n_unanswerable, args.comparison_frac, rng, lambda t: is_cmp(t[0]))
    drawn_ids = {id(t) for t in drawn}
    # answerable rows of the twins not drawn are fine as extra answerable prompts
    answerable_only += [a for t in twins if id(t) not in drawn_ids for a in t[:1]
                        if a.get("eval_factual_judge_score") == 5]
    twins = drawn
    rng.shuffle(answerable_only)
    extra = take(answerable_only, args.n_extra_answerable, args.comparison_frac, rng, is_cmp)
    out = [r for pair in twins for r in pair] + extra
    rng.shuffle(out)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["out_twins"] = len(twins)
    stats["out_extra_answerable"] = len(extra)
    stats["out_rows"] = len(out)
    manifest = {"stage": "extract", "evaluated": args.evaluated, "filtered": args.filtered, "seed": args.seed,
                "comparison_frac": args.comparison_frac, "include_judge_lt5": args.include_judge_lt5, "include_unans_answered": args.include_unans_answered,
                "stats": dict(sorted(stats.items())),
                "out_type": dict(collections.Counter(r["type"] for r in out)),
                "out_drop_reasons": dict(collections.Counter(
                    ("U/" if r.get("is_unanswerable") else "A/") + r["heldout_drop_reason"] for r in out))}
    Path(args.out).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


def format_pool(args):
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))  # dataset_rag: formatting_sft, utils
    sys.path.insert(0, str(here))
    from formatting_sft import convert_to_chat_format  # noqa: E402
    from dpo_common import context_titles, md5, requested_citation_style  # noqa: E402

    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8") if l.strip()]
    by_id = {r["id"]: r for r in rows}
    chat_path = Path(args.out).with_suffix(".chat.tmp.jsonl")
    # Same function, same seed convention as the v4 DAG (formatting_sft --format chat --seed 42).
    convert_to_chat_format(args.rows, str(chat_path), language="en", include_metadata=True, seed=args.seed)

    stats = collections.Counter()
    out = []
    for line in open(chat_path, encoding="utf-8"):
        c = json.loads(line)
        src = by_id[c["id"]]
        system, user, asst = (m["content"] for m in c["messages"])
        if len(system) > args.max_system_chars:
            stats["excluded_long"] += 1
            continue
        unans = bool(src.get("is_unanswerable"))
        twin = by_id.get(base_id(src["id"])) if unans else src
        raw_gold = list(dict.fromkeys((twin or {}).get("supporting_facts_titles") or []))
        ctx = context_titles(system)
        present = set(raw_gold) & {t.strip() for t in ctx}
        style, placement = requested_citation_style(system)
        out.append({
            "pid": md5(system + "\x00" + user)[:16],
            "qid": base_id(src["id"]),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "sft_target": asst,
            "question": user,
            "gold_answer": src["answer"],
            "hotpot_type": src["type"],
            "hotpot_level": src["level"],
            "is_unanswerable": unans,
            "unans_mode": (("partial_hop" if present else "drop_gold") if unans else None),
            "gold_titles": (src.get("supporting_facts_titles") or [] if not unans else sorted(present)),
            "raw_gold_titles": raw_gold,
            "context_titles": ctx,
            "citation_style": style,
            "citation_placement": placement,
            # from formatting_sft itself, not from the word "exactly" (which also occurs in documents)
            "strict_refusal_instruction": bool(c.get("specific_refusal_instruction")),
            "sft_target_short_refusal": unans and len(asst) < 200,
            "question_seen_in_policy_sft": False,
            "heldout": True,
            "heldout_drop_reason": src["heldout_drop_reason"],
            # answerable targets are the traces that failed the v4 filter
            "sft_target_reliable": unans,
        })
    chat_path.unlink()
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    kinds = collections.Counter(("unans:" + x["unans_mode"]) if x["is_unanswerable"] else "answerable" for x in out)
    manifest = {
        "source": f"heldout of {args.rows}", "seed": args.seed, "n_prompts": len(out), "by_kind": dict(kinds),
        "filters": dict(stats),
        "hotpot_type": dict(collections.Counter(x["hotpot_type"] for x in out)),
        "hotpot_level": dict(collections.Counter(x["hotpot_level"] for x in out)),
        "heldout_drop_reason": dict(collections.Counter(
            ("U/" if x["is_unanswerable"] else "A/") + x["heldout_drop_reason"] for x in out)),
        "citation_style": dict(collections.Counter(f'{x["citation_style"]}/{x["citation_placement"]}' for x in out)),
        "strict_refusal_instruction": sum(x["strict_refusal_instruction"] for x in out),
    }
    Path(args.out).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    e = sub.add_parser("extract", help="select held-out rows (stdlib only, run next to the evaluated file)")
    e.add_argument("--evaluated", required=True)
    e.add_argument("--filtered", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--n-unanswerable", type=int, default=1800, help="twin pairs (unanswerable + answerable)")
    e.add_argument("--n-extra-answerable", type=int, default=2400)
    e.add_argument("--comparison-frac", type=float, default=0.33,
                   help="share of comparison questions (v4 pool: 0.33; held-out natural: ~0.09, as comparisons "
                        "pass the v4 filter more often). Negative = natural distribution")
    e.add_argument("--include-judge-lt5", action="store_true", help="also answerable rows with judge < 5")
    e.add_argument("--include-unans-answered", action="store_true",
                   help="also unanswerable clones the generator answered (check answerability!)")
    e.add_argument("--seed", type=int, default=42)
    f = sub.add_parser("format", help="production chat prompts via formatting_sft + pool records")
    f.add_argument("--rows", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--max-system-chars", type=int, default=16000)
    f.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.stage == "extract" and args.comparison_frac is not None and args.comparison_frac < 0:
        args.comparison_frac = None
    extract(args) if args.stage == "extract" else format_pool(args)


if __name__ == "__main__":
    main()
