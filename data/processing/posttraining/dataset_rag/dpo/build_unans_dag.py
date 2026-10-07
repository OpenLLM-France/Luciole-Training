#!/usr/bin/env python3
"""Unanswerable versions built with the Airflow pipeline's own code (dataset_rag DAGs, run of 2026-07-22/23), for the
pool questions that only have an answerable version (2026-10-05, Jean Zay generation).

build_unans_extra.py / build_nogold_prompts*.py drop the gold chunks without replacing them: the unanswerable prompt
is always 1-2 documents shorter than its twin, so the chunk count predicts the refusal. The DAG avoids it, and this
script calls its functions instead of re-implementing them:
  HotpotQA  augment_hotpotqa (DAG dataset_rag_hotpotqa_en, run v4_backfill_pairs): source row = HotpotQA train
            (hotpotqa_cache_train); create_unanswerable_row with partial_hop (one gold paragraph removed) with
            probability 0.5, else drop_gold; every removed paragraph replaced by a distractor paragraph of another
            question (build_backfill_pool over the whole train set); then format_context_chunks as process_row
            (shuffle, up to 70 % of the original distractors removed at random, replacements protected, min 3 chunks).
  TAT-QA    augment_tatqa (DAG dataset_rag_tatqa, run full): source row = tatqa_converted.jsonl of that run (the same
            converted row, with its distractors, as the answerable prompt); create_unanswerable_row with "partial"
            (table-text questions: the table or the gold paragraphs, coin flip) with probability 0.5, else drop_gold;
            replacements of the same kind (table for table) from other reports, never containing a textual answer;
            [CONTEXT] summaries on 30 % of the rows (contextualization cache of the run, Mistral for missing reports);
            format_context_chunks (Tables and Paragraphs renumbered 1..N in display order, min 2 chunks).
Then formatting_sft.convert_to_chat_format (seed 42): citation style, placement (citation_end_of_completion drawn
as in process_row). The pool only holds generic-instruction prompts: a row drawn into the 5 % strict slice is formatted
again with the next seed. The record copies its answerable twin (judge-facing gold answer, split) and gets the
downstream names: unans_mode drop_gold / partial_hop, clone_source "dag", source_pid.

Questions: as build_unans_extra.py (answerable prompts usable for training, no unanswerable version in the pool, TAT-QA
not cloned in heldout_tatqa_u_sft4152). Prompts are then the published ones (refusal rule without its invitation clause).
Output dpo/data/unans_dag/<dataset>/prompts.jsonl, manifest.json; the context judge runs next (run_unans_extra_judge.sh).

    venv/bin/python dpo/build_unans_dag.py --tatqa-converted <run>/tatqa_converted.jsonl \
        --tatqa-ctx-cache <run>/tatqa_augmented_contextualization_cache.jsonl [--mistral-concurrency 4]
"""
import argparse
import asyncio
import collections
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import augment_hotpotqa as AH  # noqa: E402
import augment_tatqa as AT  # noqa: E402
from dpo_common import context_titles, md5, requested_citation_style  # noqa: E402
from formatting_sft import EOC_PROB, convert_to_chat_format  # noqa: E402
from release_text import published_prompt  # noqa: E402
from utils import format_context_chunks  # noqa: E402

D = HERE / "data"
OUT = Path(os.environ.get("UNANS_DAG_OUT", D / "unans_dag"))  # env: Airflow run dir
POOLS = {"hotpotqa": D / "heldout_full_sft4152", "tatqa": D / "heldout_tatqa_generic_sft4152"}


def rows(p):
    p = Path(p)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def table(qid):
    return qid.split("_", 1)[0]


def targets(ds):
    """Answerable pool prompts usable for training whose question has no unanswerable version (build_unans_extra.py)."""
    val_q = {r["qid"] for r in rows(D / "rag_dpo_onpolicy_8802410" / "val.jsonl")}
    ev_q = {p["qid"] for p in rows(D / "p9_paired_eval" / "prompts.jsonl")}
    bad_tables = {table(q) for q in val_q | ev_q if "_" in q}
    P = rows(POOLS[ds] / "prompts.jsonl")
    cv = {r["pid"]: r for r in rows(POOLS[ds] / "context_verdicts.jsonl")}
    has_u = {p["qid"] for p in P if p["is_unanswerable"]}
    if ds == "tatqa":
        has_u |= {p["qid"] for p in rows(D / "heldout_tatqa_u_sft4152" / "prompts.jsonl")}
    out = {}
    for p in sorted(P, key=lambda p: p["pid"]):
        q = p["qid"]
        if p["is_unanswerable"] or q in has_u or p.get("strict_refusal_instruction") or q in out:
            continue
        if cv.get(p["pid"], {}).get("verdict") != "answerable" or q in ev_q:
            continue
        if ds == "hotpotqa" and (q in val_q or p.get("pilot_split") == "val"):
            continue
        if ds == "tatqa" and (table(q) in bad_tables or int(hashlib.sha256(f"42:{table(q)}".encode()).hexdigest(), 16) % 100 < 10):
            continue
        out[q] = p
    return out


def format_generic(aug_rows, tmpdir):
    """convert_to_chat_format (seed 42); rows drawn into the strict slice are formatted again with the next seed."""
    done, todo, seed = {}, list(aug_rows), 42
    while todo:
        src, dst = Path(tmpdir) / f"in_{seed}.jsonl", Path(tmpdir) / f"out_{seed}.jsonl"
        src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in todo), encoding="utf-8")
        convert_to_chat_format(str(src), str(dst), language="en", include_metadata=True, seed=seed)
        by_id = {r["id"]: r for r in todo}
        again = []
        for c in rows(dst):
            if c.get("specific_refusal_instruction"):
                again.append(by_id[c["id"]])
            else:
                done[c["id"]] = (c, seed)
        todo, seed = again, seed + 1
    return done


def record(ds, a, c, seed, mode, gold_kept, extra):
    system, user = c["messages"][0]["content"], c["messages"][1]["content"]
    style, placement = requested_citation_style(system)
    msgs = published_prompt([{"role": "system", "content": system}, {"role": "user", "content": user}])
    keep = {k: v for k, v in a.items() if k not in ("sft_target", "strict_phrase", "strict_rule_id", "messages")}
    return {**keep, "pid": md5(system + "\x00" + user)[:16], "messages": msgs, "dataset": ds,
            "is_unanswerable": True, "unans_mode": mode, "gold_titles": gold_kept,
            "raw_gold_titles": a.get("raw_gold_titles") or a["gold_titles"], "context_titles": context_titles(system),
            "citation_style": style, "citation_placement": placement, "strict_refusal_instruction": False,
            "sft_target": None, "sft_target_short_refusal": False, "sft_target_reliable": False,
            "gold_from_twin": True, "heldout_drop_reason": "dag_clone", "clone_source": "dag",
            "source_pid": a["pid"], "format_seed": seed, "created": "2026-10-05", **extra}


def build_hotpotqa(stats):
    from datasets import load_from_disk
    T = targets("hotpotqa")
    ds = load_from_disk(str(ROOT / "hotpotqa_cache_train"))
    all_rows = [{"id": r["id"], "question": r["question"], "answer": r["answer"], "type": r["type"], "level": r["level"],
                 "supporting_facts": {"title": list(r["supporting_facts"]["title"]),
                                      "sent_id": list(r["supporting_facts"]["sent_id"])},
                 "context": {"title": list(r["context"]["title"]), "sentences": [list(s) for s in r["context"]["sentences"]]}}
                for r in ds]
    by_id = {r["id"]: r for r in all_rows}
    pool = AH.build_backfill_pool(all_rows)
    random.seed(42)
    aug, meta = [], {}
    for q in sorted(T):
        row = by_id.get(q)
        if row is None:
            stats["hotpotqa/source_row_missing"] += 1
            continue
        u = None
        if random.random() < 0.5:  # DAG partial_hop_ratio
            u = AH.create_unanswerable_row(row, "en", mode="partial_hop", backfill_pool=pool)
        if u is None:
            u = AH.create_unanswerable_row(row, "en", mode="drop_gold", backfill_pool=pool)
        if u is None:
            stats["hotpotqa/create_failed"] += 1
            continue
        ctx, cs = format_context_chunks(context=u["context"], relevant_titles=set(u["supporting_facts"]["title"]),
                                        shuffle=True, max_remove_background_ratio=0.7, min_chunks=3,
                                        protected_background_titles=set(u.get("_protected_background_titles", [])),
                                        contextual_summaries=None)
        mapping = cs.pop("title_mapping", {})
        gold = set(row["supporting_facts"]["title"])
        meta[u["id"]] = (q, u["_unanswerable_mode"], [mapping.get(t, t) for t in dict.fromkeys(u["context"]["title"]) if t in gold],
                         {"backfilled_titles": u["_backfilled_titles"], "chunks_total": cs.get("chunks_total")})
        aug.append({"id": u["id"], "question": u["question"], "context": ctx, "answer": "",
                    "reasoning_trace": AH.UNANSWERABLE_REFUSAL_EN, "type": u["type"], "level": u["level"],
                    "is_unanswerable": True, "grounded_refusal": True,
                    "citation_end_of_completion": random.Random(f"{u['id']}:eoc").random() < EOC_PROB,
                    "supporting_facts_titles": [], **cs})
    return T, aug, meta


async def build_tatqa(args, stats):
    T = targets("tatqa")
    conv = rows(args.tatqa_converted)
    by_id = {r["id"]: r for r in conv}
    pool = AT.build_backfill_pool(conv)
    docs, lookup = AT.load_contextualization_sources(str(ROOT / "tatqa_dataset_train.json"))
    AT._load_contextualization_cache(Path(args.tatqa_ctx_cache))
    n_cached = len(AT._ctx_cache)
    AT._ctx_cache_path = OUT / "tatqa" / "contextualization_cache_new.jsonl"  # new summaries persisted here
    random.seed(42)
    aug, meta = [], {}
    sem = asyncio.Semaphore(args.mistral_concurrency)
    async with __import__("aiohttp").ClientSession() as session:
        for q in sorted(T):
            row = by_id.get(q)
            if row is None:
                stats["tatqa/source_row_missing"] += 1
                continue
            u = None
            if random.random() < 0.5:  # DAG partial_ratio
                u = AT.create_unanswerable_row(row, mode="partial", backfill_pool=pool)
            if u is None:
                u = AT.create_unanswerable_row(row, backfill_pool=pool)
            if u is None:
                stats["tatqa/create_failed"] += 1
                continue
            summaries = None
            if random.random() < 0.3:  # DAG contextualize_ratio
                summaries = await AT._contextualize_retrieved_chunks(
                    session, u, docs, lookup, sem, api_key=AT.LLM_API_KEY, max_retries=3, request_timeout=120,
                    debug_http=False)
                stats["tatqa/contextualized"] += bool(summaries)
            ctx, cs = format_context_chunks(context=u["context"],
                                            relevant_titles=set(u.get("supporting_facts", {}).get("title", [])),
                                            shuffle=True, max_remove_background_ratio=0.7, min_chunks=2,
                                            protected_background_titles={"Table", *u.get("_protected_background_titles", [])},
                                            contextual_summaries=summaries)
            mapping = cs.pop("title_mapping", {})
            gold = set(row.get("supporting_facts", {}).get("title", []))
            mode = "partial_hop" if u["_unanswerable_mode"] == "partial" else "drop_gold"
            meta[u["id"]] = (q, mode, [mapping.get(t, t) for t in dict.fromkeys(u["context"]["title"]) if t in gold],
                             {"backfilled_titles": u["_backfilled_titles"], "chunks_total": cs.get("chunks_total"),
                              "removed_gold": sorted(gold - set(u["context"]["title"]))})
            aug.append({"id": u["id"], "question": u["question"], "context": ctx, "answer": "", "answer_raw": [],
                        "answer_type": u.get("answer_type", ""), "answer_from": u.get("answer_from", ""),
                        "scale": u.get("scale", ""), "reasoning_trace": AT.UNANSWERABLE_REFUSAL,
                        "is_unanswerable": True, "grounded_refusal": True,
                        "citation_end_of_completion": random.Random(f"{u['id']}:eoc").random() < EOC_PROB,
                        "supporting_facts_titles": [], "type": u.get("type", u.get("answer_type", "unknown")),
                        "level": u.get("level", "tatqa"), **cs})
    stats["tatqa/ctx_cache_entries_loaded"] = n_cached
    stats["tatqa/ctx_new_reports_summarised"] = len(AT._ctx_cache) - n_cached
    return T, aug, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tatqa-converted", required=True)
    ap.add_argument("--tatqa-ctx-cache", required=True)
    ap.add_argument("--mistral-concurrency", type=int, default=4, help="contextualization calls (Mistral cap: 12 in total)")
    ap.add_argument("--only", choices=("hotpotqa", "tatqa"))
    args = ap.parse_args()
    stats = collections.Counter()
    for ds in ("hotpotqa", "tatqa"):
        if args.only and ds != args.only:
            continue
        (OUT / ds).mkdir(parents=True, exist_ok=True)
        T, aug, meta = build_hotpotqa(stats) if ds == "hotpotqa" else asyncio.run(build_tatqa(args, stats))
        with tempfile.TemporaryDirectory() as tmp:
            chats = format_generic(aug, tmp)
        out = []
        for rid, (q, mode, gold_kept, extra) in meta.items():
            c, seed = chats[rid]
            r = record(ds, T[q], c, seed, mode, gold_kept, extra)
            out.append(r)
            stats[f"{ds}/{mode}"] += 1
            stats[f"{ds}/format_seed_{seed}"] += 1
        assert len({r["pid"] for r in out}) == len(out)
        with open(OUT / ds / "prompts.jsonl", "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # chunk-count check against the answerable twin: the point of the backfill
        delta = collections.Counter(len(r["context_titles"]) - len(T[r["qid"]]["context_titles"]) for r in out)
        stats[f"{ds}/questions"] = len(T)
        stats[f"{ds}/prompts"] = len(out)
        (OUT / ds / "chunk_delta_vs_answerable.json").write_text(json.dumps(dict(sorted(delta.items()))) + "\n")
    (OUT / f"manifest{'_' + args.only if args.only else ''}.json").write_text(json.dumps(dict(sorted(stats.items())), indent=1) + "\n")
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    main()
