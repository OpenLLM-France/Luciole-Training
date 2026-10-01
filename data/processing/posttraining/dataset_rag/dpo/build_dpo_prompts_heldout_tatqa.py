#!/usr/bin/env python3
"""DPO prompt pool from TatQA train questions that did NOT survive the tatqa_v4 pipeline.

TatQA counterpart of build_dpo_prompts_heldout.py. Goal: A_false_refusal pairs on TatQA
(at the judge the prod DPO refuses 62 % of answerable tatqa questions, SFT newrag step
4152 35 %, mostly calculations). The policy SFT contains all of tatqa_v4, produced by the
Airflow run dataset_rag_tatqa/full on the Airflow host: 13 215 answerable rows (all of TatQA train)
+ 1 321 unanswerable clones, 5 674 kept by the filter. Questions with no row in tatqa_v4 are
held out; their prompts are rebuilt by the same formatting (formatting_sft, seed 42) and the
same post-hoc table renumbering (fix_table_titles, no "Table dist_<hash>").

Drop reasons (utils.filter_rows, --no-answer-match, pairs kept whole):
  answerable    judge_lt5    factual judge < 5
                cit_f1_lt1   judge 5 but citation F1 < 1 (question sound)
                pair         passed, but its unanswerable twin failed
  unanswerable  refused      generator refused, dropped because the answerable twin failed
                answered     generator answered: often answerable in disguise

Three stages:

  # 1) the Airflow host, stdlib python, next to the run (one ssh call, tar back on stdout)
  python3 build_dpo_prompts_heldout_tatqa.py extract --run-dir $R --out-dir heldout_tatqa
  # 2) local, dataset_rag venv: proof that the kept rows rebuild tatqa_v4 exactly
  venv/bin/python dpo/build_dpo_prompts_heldout_tatqa.py verify --kept $R/..._filtered.jsonl \
      --v4-raw tatqa_v4_0394854.parquet --v4-fixed tatqa_v4_61dae84.parquet
  # 3) local: contamination checks (test/dev), selection, production prompts
  venv/bin/python dpo/build_dpo_prompts_heldout_tatqa.py format --rows heldout_rows.jsonl \
      --v4-fixed tatqa_v4_61dae84.parquet --test tatqa_dataset_test_gold.json --dev tatqa_dev.json \
      --kept $R/..._filtered.jsonl --verify-report verify_v4.json --out dpo/data/prompts_tatqaheldout.jsonl
  python3 dpo/add_strict_variants.py --pool dpo/data/prompts_tatqaheldout.jsonl --phrases pool
  python3 dpo/build_dpo_prompts_heldout_tatqa.py seal --pool dpo/data/prompts_tatqaheldout.jsonl
"""
import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

UNANS_SUFFIX = "_unanswerable"
KEEP_FIELDS = ("id", "question", "context", "answer", "answer_raw", "answer_type", "answer_from", "scale",
               "reasoning_trace", "type", "level", "is_unanswerable", "grounded_refusal",
               "citation_end_of_completion", "supporting_facts_titles", "chunks_relevant", "chunks_total",
               "eval_factual_judge_score", "eval_chunk_f1", "eval_model_refused", "eval_answer_correct")
# Not dpo_common.context_titles: anchored at line start, it misses the first document, whose
# "* filename:" follows "Here are the retrieved documents : `" on the same line.
DOC_TITLE_RE = re.compile(r"\* filename: (.+?)\s*$", re.M)
CHUNK_RE = re.compile(r"\* filename: (.+?)\n\n\[CHUNK_START\]\n(.*?)\n\[CHUNK_END\]", re.S)


def base_id(row_id):
    s = str(row_id)
    return s[: -len(UNANS_SUFFIX)] if s.endswith(UNANS_SUFFIX) else s


def norm_q(q):
    return " ".join(str(q).lower().split())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- extract (stdlib)

def passes(row):
    """utils.filter_row(require_answer_match=False)."""
    if row.get("is_unanswerable", False):
        return row.get("eval_model_refused", True) is not False
    if "eval_chunk_f1" in row and row["eval_chunk_f1"] != 1.0:
        return False
    if "eval_factual_judge_score" in row and row["eval_factual_judge_score"] != 5:
        return False
    return True


def drop_reason(row, twin_passes):
    if row.get("is_unanswerable"):
        return "refused" if row.get("eval_model_refused") is not False else "answered"
    if row.get("eval_factual_judge_score") != 5:
        return "judge_lt5"
    if row.get("eval_chunk_f1") != 1.0:
        return "cit_f1_lt1"
    return "pair" if twin_passes is False else "unknown"


def extract(args):
    run = Path(args.run_dir)
    evaluated = run / "tatqa_augmented_evaluated.jsonl"
    filtered = run / "tatqa_augmented_evaluated_filtered.jsonl"
    converted = run / "tatqa_converted.jsonl"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()

    kept_ids = [json.loads(l)["id"] for l in open(filtered, encoding="utf-8")]
    kept_set = set(kept_ids)
    v4_ids = {base_id(i) for i in kept_ids}

    rows = [json.loads(l) for l in open(evaluated, encoding="utf-8")]
    stats["evaluated_rows"] = len(rows)
    stats["kept_rows"] = len(kept_ids)
    # Filter replay: the kept file must be exactly filter_rows(evaluated) (keep_pairs, no answer match).
    sides_all, sides_ok = collections.defaultdict(set), collections.defaultdict(set)
    for r in rows:
        sides_all[base_id(r["id"])].add(bool(r.get("is_unanswerable")))
        if passes(r):
            sides_ok[base_id(r["id"])].add(bool(r.get("is_unanswerable")))
    replay = [r["id"] for r in rows if passes(r) and (len(sides_all[base_id(r["id"])]) < 2
                                                       or len(sides_ok[base_id(r["id"])]) == 2)]
    # The full run predates keep_pairs (utils.filter_rows): twins were filtered independently.
    filter_replay = {"keep_pairs": replay == kept_ids,
                     "independent": [r["id"] for r in rows if passes(r)] == kept_ids}

    groups = collections.defaultdict(dict)
    for r in rows:
        b = base_id(r["id"])
        if b in v4_ids:
            # a failed row whose twin is in v4: the question was seen in SFT, excluded too
            stats["rows_in_v4" if r["id"] in kept_set else
                  ("U" if r.get("is_unanswerable") else "A") + "_not_kept_but_twin_in_v4"] += 1
            continue
        side = "U" if r.get("is_unanswerable") else "A"
        if side in groups[b]:
            stats["duplicate_id_side"] += 1
            continue
        groups[b][side] = {k: r[k] for k in KEEP_FIELDS if k in r}

    # document ids of every chunk put in the context by the converter (gold table + distractors),
    # to check test/dev contamination of the distractors too
    doc_ids = {}
    for line in open(converted, encoding="utf-8"):
        c = json.loads(line)
        if c["id"] in groups:
            doc_ids[c["id"]] = sorted({str(ch.get("document_id", "")) for ch in c.get("chunks", [])} - {""})

    out = []
    for b, g in groups.items():
        a, u = g.get("A"), g.get("U")
        u_ok = None if u is None else u.get("eval_model_refused") is not False
        for side, r in (("A", a), ("U", u)):
            if r is None:
                continue
            r["heldout_drop_reason"] = drop_reason(r, u_ok if side == "A" else None)
            r["twin_in_heldout"] = (u if side == "A" else a) is not None
            r["converted_doc_ids"] = doc_ids.get(b)
            stats[f"{side}/{r['heldout_drop_reason']}"] += 1
            out.append(r)
    with open(out_dir / "heldout_rows.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "stage": "extract", "run_dir": str(run),
        "sha256": {p.name: sha256(p) for p in (evaluated, filtered, converted)},
        "filter_replay_identical_to_kept_file": filter_replay,
        "stats": dict(sorted(stats.items())), "heldout_rows": len(out),
        "heldout_groups": len(groups),
    }
    (out_dir / "heldout_rows.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


# --------------------------------------------------------------------------- local helpers

def _local_imports():
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))  # dataset_rag: formatting_sft, utils, fix_table_titles
    sys.path.insert(0, str(here))


def format_rows(rows, seed, tmp_path):
    """formatting_sft.convert_to_chat_format (as the DAG: --format chat --seed 42, metadata kept),
    then fix_table_titles.fix_row (as tatqa_v4 on HF). Returns [(raw_chat, fixed_chat)]."""
    _local_imports()
    from formatting_sft import convert_to_chat_format
    from fix_table_titles import fix_row
    tmp_in, tmp_out = Path(str(tmp_path) + ".in.jsonl"), Path(str(tmp_path) + ".out.jsonl")
    with open(tmp_in, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = convert_to_chat_format(str(tmp_in), str(tmp_out), language="en", include_metadata=True, seed=seed)
    if stats.skipped:
        raise RuntimeError(f"formatting skipped {stats.skipped} rows")
    fix_stats = collections.Counter()
    pairs = []
    for line in open(tmp_out, encoding="utf-8"):
        raw = json.loads(line)
        fixed = fix_row(json.loads(line), fix_stats)
        pairs.append((raw, fixed))
    tmp_in.unlink()
    tmp_out.unlink()
    return pairs, fix_stats


def msg_key(messages):
    return tuple(m["content"] for m in messages)


def read_parquet_messages(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


# --------------------------------------------------------------------------- verify

def verify(args):
    kept = [json.loads(l) for l in open(args.kept, encoding="utf-8")]
    pairs, fix_stats = format_rows(kept, args.seed, Path(args.kept).with_suffix(".verify"))
    report = {"kept_rows": len(kept), "fix_table_titles": dict(fix_stats)}
    for label, path, idx in (("v4_raw", args.v4_raw, 0), ("v4_fixed", args.v4_fixed, 1)):
        if not path:
            continue
        hf = read_parquet_messages(path)
        mine = [p[idx] for p in pairs]
        same_order = len(hf) == len(mine) and all(
            msg_key(h["messages"]) == msg_key(m["messages"])
            and list(h["supporting_facts_titles"]) == list(m["supporting_facts_titles"])
            and h["chunks_total"] == m["chunks_total"] for h, m in zip(hf, mine))
        diff = collections.Counter(msg_key(h["messages"]) for h in hf)
        diff.subtract(collections.Counter(msg_key(m["messages"]) for m in mine))
        report[label] = {"file": path, "sha256": sha256(path), "rows": len(hf),
                         "identical_in_order_incl_metadata": same_order,
                         "rows_only_in_hf": sum(v for v in diff.values() if v > 0),
                         "rows_only_rebuilt": -sum(v for v in diff.values() if v < 0)}
    print(json.dumps(report, indent=1))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1))


# --------------------------------------------------------------------------- format

def tatqa_split_index(path):
    """question uids, table uids, normalized question texts of a TatQA json file."""
    data = json.load(open(path, encoding="utf-8"))
    q_uids, t_uids, texts = set(), set(), set()
    for doc in data:
        t_uids.add(str(doc["table"]["uid"]))
        for q in doc["questions"]:
            q_uids.add(str(q["uid"]))
            texts.add(norm_q(q["question"]))
    return q_uids, t_uids, texts


def chunk_texts(context, titles):
    by_title = {t.strip(): body for t, body in CHUNK_RE.findall(context)}
    return [by_title[t] for t in titles if t in by_title]


def take_weighted(items, n, weight, rng):
    """n items without replacement, P proportional to weight(item) (Efraimidis-Spirakis)."""
    keyed = sorted(items, key=lambda x: rng.random() ** (1.0 / weight(x)), reverse=True)
    return keyed[:n]


def format_pool(args):
    _local_imports()
    from dpo_common import md5, requested_citation_style

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8") if l.strip()]
    excl = collections.Counter()

    # --- exclusions: tatqa_v4 (policy SFT), lighteval test, dev
    v4 = read_parquet_messages(args.v4_fixed)
    v4_texts = {norm_q(r["messages"][1]["content"]) for r in v4}
    # TatQA asks ~6 questions per table: a held-out question often has its table in tatqa_v4
    v4_tables = {base_id(json.loads(l)["id"]).split("_", 1)[0] for l in open(args.kept, encoding="utf-8")}
    test_q, test_t, test_txt = tatqa_split_index(args.test)
    dev_q, dev_t, dev_txt = tatqa_split_index(args.dev)
    eval_q, eval_t, eval_txt = test_q | dev_q, test_t | dev_t, test_txt | dev_txt

    def exclusion(r):
        table_uid, q_uid = base_id(r["id"]).split("_", 1)
        if q_uid in eval_q:
            return "question_uid_in_test_or_dev"
        if table_uid in eval_t:
            return "table_uid_in_test_or_dev"
        if set(r.get("converted_doc_ids") or []) & eval_t:
            return "distractor_table_uid_in_test_or_dev"
        if norm_q(r["question"]) in eval_txt:
            return "question_text_in_test_or_dev"
        if norm_q(r["question"]) in v4_texts:
            return "question_text_in_tatqa_v4"
        return None

    by_base = collections.defaultdict(dict)
    for r in rows:
        why = exclusion(r)
        if why:
            excl[("U/" if r.get("is_unanswerable") else "A/") + why] += 1
            continue
        by_base[base_id(r["id"])]["U" if r.get("is_unanswerable") else "A"] = r

    # --- selection
    ans_ok = {"cit_f1_lt1", "pair"}  # judge 5: generation sound, dropped for another reason
    answerable = [g["A"] for g in by_base.values() if "A" in g and g["A"]["heldout_drop_reason"] in ans_ok]
    unans = [g["U"] for g in by_base.values() if "U" in g and g["U"]["heldout_drop_reason"] == "refused"]
    avail = collections.Counter(f'{r["answer_type"]}|{r["answer_from"]}' for r in answerable)
    answerable.sort(key=lambda r: r["id"])
    unans.sort(key=lambda r: r["id"])

    def weight(r):
        w = args.w_arithmetic if r["answer_type"] == "arithmetic" else 1.0
        return w * (args.w_table if r["answer_from"] in ("table", "table-text") else 1.0)

    picked = take_weighted(answerable, min(args.n_answerable, len(answerable)), weight, rng)
    picked_bases = {base_id(r["id"]) for r in picked}
    # the answerable twin of every refused clone comes along when it is eligible
    twins_added = [by_base[base_id(u["id"])]["A"] for u in unans
                   if "A" in by_base[base_id(u["id"])]
                   and by_base[base_id(u["id"])]["A"]["heldout_drop_reason"] in ans_ok
                   and base_id(u["id"]) not in picked_bases]
    selected = picked + twins_added + unans
    rng.shuffle(selected)

    def records(selected, tmp):
        """production chat prompts + pool records of the selected rows."""
        pairs, fix_stats = format_rows(selected, args.seed, tmp)
        src_by_id = {r["id"]: r for r in selected}
        stats = collections.Counter()
        out = []
        for raw, c in pairs:
            src = src_by_id[c["id"]]
            system, user, asst = (m["content"] for m in c["messages"])
            if "dist_" in system:
                stats["excluded_dist_in_prompt"] += 1
                continue
            if "dist_" in asst:  # e.g. "Table (dist_<hash>)" in the trace, a form fix_table_titles ignores
                stats["excluded_dist_in_sft_target"] += 1
                continue
            if len(system) > args.max_system_chars:
                stats["excluded_long"] += 1
                continue
            unans_row = bool(src.get("is_unanswerable"))
            ctx = DOC_TITLE_RE.findall(system)
            style, placement = requested_citation_style(system)
            unans_mode, twin = None, by_base[base_id(src["id"])].get("A")
            if unans_row:
                # partial = some gold chunk of the answerable twin is still in the clone's context
                # (titles are renumbered per row: compare chunk texts, not titles)
                gold_txt = chunk_texts(twin["context"], twin["supporting_facts_titles"]) if twin else []
                present = [t for t in gold_txt if t in src["context"]]
                unans_mode = "partial_hop" if present else ("drop_gold" if twin else "unknown")
                gold_titles = [t for t, body in CHUNK_RE.findall(system) if body in present]
                raw_gold = twin["supporting_facts_titles"] if twin else []
            else:
                gold_titles = list(c["supporting_facts_titles"])
                raw_gold = list(c["supporting_facts_titles"])
            out.append({
                "pid": md5(system + "\x00" + user)[:16],
                "qid": base_id(src["id"]),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "sft_target": asst,
                "question": user,
                "gold_answer": (twin or src)["answer"] if unans_row else src["answer"],
                "gold_scale": (twin or src).get("scale", "") if unans_row else src.get("scale", ""),
                "dataset": "tatqa",
                # hotpot_* kept for the downstream scripts (build_pairs, to_nemo_rl): TatQA type / level
                "hotpot_type": src["answer_type"] or (twin or {}).get("answer_type"),
                "hotpot_level": src.get("level", "tatqa"),
                "answer_type": src["answer_type"] or (twin or {}).get("answer_type"),
                "answer_from": src["answer_from"] or (twin or {}).get("answer_from"),
                "is_unanswerable": unans_row,
                "unans_mode": unans_mode,
                "gold_titles": gold_titles,
                "raw_gold_titles": raw_gold,
                "context_titles": ctx,
                "citation_style": style,
                "citation_placement": placement,
                "strict_refusal_instruction": bool(c.get("specific_refusal_instruction")),
                "sft_target_short_refusal": unans_row and len(asst) < 200,
                "question_seen_in_policy_sft": False,
                "table_seen_in_policy_sft": base_id(src["id"]).split("_", 1)[0] in v4_tables,
                "heldout": True,
                "heldout_drop_reason": src["heldout_drop_reason"],
                "sft_target_reliable": unans_row,
            })
        return out, stats, fix_stats

    if args.answered_clones_out:
        # unanswerable clones the generator answered (rule 4 excludes them from the pool): kept apart,
        # for the context audit to keep only the truly unanswerable ones
        sel = sorted((g["U"] for g in by_base.values() if "U" in g and g["U"]["heldout_drop_reason"] == "answered"),
                     key=lambda r: r["id"])
        out, stats, fix_stats = records(sel, Path(args.answered_clones_out).with_suffix(".tmp"))
        with open(args.answered_clones_out, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        man = {"rows": args.rows, "seed": args.seed, "exclusions": dict(sorted(excl.items())),
               "eligible": len(sel), "n": len(out), "filters": dict(stats),
               "unans_mode": dict(collections.Counter(x["unans_mode"] for x in out)),
               "strict_refusal_instruction": sum(x["strict_refusal_instruction"] for x in out),
               "sha256": sha256(args.answered_clones_out)}
        Path(args.answered_clones_out).with_suffix(".manifest.json").write_text(json.dumps(man, indent=1))
        print(json.dumps(man, indent=1))
        return

    # --- production prompts
    out, stats, fix_stats = records(selected, Path(args.out).with_suffix(".tmp"))
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ans_out = [x for x in out if not x["is_unanswerable"]]
    cross = collections.Counter((x["answer_type"], x["answer_from"]) for x in ans_out)
    types = sorted({t for t, _ in cross} | {t for t, _ in (k.split("|") for k in avail)})
    froms = ["table", "table-text", "text"]
    manifest = {
        "source": f"heldout of {args.rows}", "seed": args.seed,
        "extract": json.load(open(Path(args.rows).with_suffix(".manifest.json"))),
        "verify_v4": json.load(open(args.verify_report)) if args.verify_report else None,
        "sha256": {"pool_base": sha256(args.out), "rows": sha256(args.rows), "kept": sha256(args.kept),
                   "v4_fixed": sha256(args.v4_fixed),
                   "test": sha256(args.test), "dev": sha256(args.dev)},
        "exclusions": dict(sorted(excl.items())),
        "eligible": {"answerable_judge5": len(answerable), "unanswerable_refused": len(unans),
                     "answerable_judge5_by_type_from": dict(sorted(avail.items()))},
        "selection": {"n_answerable": args.n_answerable, "w_arithmetic": args.w_arithmetic,
                      "w_table": args.w_table, "picked_weighted": len(picked),
                      "answerable_twins_of_refused_added": len(twins_added),
                      "unanswerable": len(unans)},
        "filters": dict(stats), "fix_table_titles": dict(fix_stats),
        "n_prompts": len(out),
        "by_kind": dict(collections.Counter(("unans:" + x["unans_mode"]) if x["is_unanswerable"] else "answerable"
                                            for x in out)),
        "answerable_answer_type_x_answer_from": {t: {f: cross.get((t, f), 0) for f in froms} for t in types},
        "answerable_answer_type": dict(collections.Counter(x["answer_type"] for x in ans_out)),
        "answerable_answer_from": dict(collections.Counter(x["answer_from"] for x in ans_out)),
        "unanswerable_answer_type_x_answer_from": dict(collections.Counter(
            f'{x["answer_type"]}|{x["answer_from"]}' for x in out if x["is_unanswerable"])),
        "heldout_drop_reason": dict(collections.Counter(
            ("U/" if x["is_unanswerable"] else "A/") + x["heldout_drop_reason"] for x in out)),
        "citation_style": dict(collections.Counter(f'{x["citation_style"]}/{x["citation_placement"]}' for x in out)),
        "strict_refusal_instruction": sum(x["strict_refusal_instruction"] for x in out),
        "table_seen_in_policy_sft": sum(x["table_seen_in_policy_sft"] for x in out),
        "gold_titles_not_in_context": sum(not set(x["gold_titles"]) <= set(x["context_titles"]) for x in out),
        "twins_in_pool": sum(1 for q, n in collections.Counter(x["qid"] for x in out).items() if n == 2),
    }
    Path(args.out).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


def seal(args):
    """After add_strict_variants: final sha256 and counts, appended to the manifest."""
    rows = [json.loads(l) for l in open(args.pool, encoding="utf-8")]
    mpath = Path(args.pool).with_suffix(".manifest.json")
    manifest = json.load(open(mpath))
    manifest["final"] = {
        "sha256": sha256(args.pool), "n_prompts": len(rows),
        "strict_variants": sum(bool(r.get("is_strict_variant")) for r in rows),
        "strict_rule_id": dict(collections.Counter(r.get("strict_rule_id") for r in rows if r.get("is_strict_variant"))),
        "distinct_strict_phrases": len({r["strict_phrase"] for r in rows if r.get("is_strict_variant")}),
        "by_kind": dict(collections.Counter(
            ("U" if r["is_unanswerable"] else "A") + ("_strict" if r.get("is_strict_variant") else "") for r in rows)),
    }
    mpath.write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["final"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    e = sub.add_parser("extract", help="held-out rows of the tatqa_v4 run (stdlib only, on the Airflow host)")
    e.add_argument("--run-dir", required=True)
    e.add_argument("--out-dir", required=True)
    v = sub.add_parser("verify", help="kept rows -> formatting -> identical to tatqa_v4 on HF")
    v.add_argument("--kept", required=True)
    v.add_argument("--v4-raw", help="tatqa_v4 parquet before the table-title fix (HF 0394854)")
    v.add_argument("--v4-fixed", help="tatqa_v4 parquet after fix_table_titles (HF 61dae84)")
    v.add_argument("--report")
    v.add_argument("--seed", type=int, default=42)
    f = sub.add_parser("format", help="exclusions, selection, production chat prompts")
    f.add_argument("--rows", required=True)
    f.add_argument("--v4-fixed", required=True)
    f.add_argument("--kept", required=True, help="kept rows of the run (tatqa_v4 ids)")
    f.add_argument("--verify-report", help="output of the verify stage, copied into the manifest")
    f.add_argument("--test", required=True, help="tatqa_dataset_test_gold.json (lighteval benchmark source)")
    f.add_argument("--dev", required=True, help="tatqa_dev.json")
    f.add_argument("--out", help="pool (required unless --answered-clones-out)")
    f.add_argument("--answered-clones-out",
                   help="write only the held-out unanswerable clones the generator answered, then stop")
    f.add_argument("--n-answerable", type=int, default=3000, help="weighted draw among judge-5 answerable rows")
    f.add_argument("--w-arithmetic", type=float, default=3.0, help="draw weight of answer_type=arithmetic")
    f.add_argument("--w-table", type=float, default=2.0, help="draw weight of answer_from in table/table-text")
    f.add_argument("--max-system-chars", type=int, default=16000)
    f.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("seal", help="final sha256/counts after add_strict_variants")
    s.add_argument("--pool", required=True)
    args = ap.parse_args()
    if args.stage == "format" and not (args.out or args.answered_clones_out):
        ap.error("format needs --out or --answered-clones-out")
    {"extract": extract, "verify": verify, "format": format_pool, "seal": seal}[args.stage](args)


if __name__ == "__main__":
    main()
