#!/usr/bin/env python3
"""Unanswerable TatQA clones built from held-out answerable prompts (no LLM call).
ATTENTION (2026-10-05) : les chunks gold sont retirés SANS remplacement, la version sans réponse a toujours 1-2 documents
de moins que sa jumelle (biais « moins de chunks => refuser »). Ne plus utiliser pour de nouvelles données : passer par
dpo/build_unans_dag.py (code du DAG, remplacement un pour un). Voir DIAGNOSTIC_dpo_2026-10-05.md § 8.


TatQA counterpart of build_nogold_prompts.py. The source prompts are the answerable prompts of
heldout_tatqa_generic_sft4152 whose context verdict is "answerable". Gold chunks are removed:
  drop_gold    every gold chunk (table and paragraphs)
  partial_hop  table-text questions only (--partial-ratio): the gold table is removed, the gold
               paragraphs stay. Removing the paragraphs instead (--partial-remove both: coin flip, as
               augment_tatqa --unanswerable-partial-ratio does) is off by default: TatQA gold paragraphs
               are often a mere heading ("The following table sets forth..."), so the clone stays
               answerable. First batch (heldout_tatqa_u_sft4152, 2026-10-01, built with "both"): judge
               of contexts, unanswerable 7 % with the paragraphs removed vs 76 % with the table removed.
A chunk goes with its [CONTEXT] summary. The remaining chunks keep their order and are
renumbered "Table 1..N" / "Paragraph 1..N" in that order: a gap ("Table 1, Table 3") would
tell that a document was removed. Distractors are untouched; the judge of contexts must
still confirm that the answer is not derivable from them (TatQA distractors are tables of
other reports with the same line items).

The clone keeps the qid, the split and the judge-facing gold of its source (the context
audit checks that this answer is NOT derivable). unans_mode uses the downstream names.

    python3 dpo/build_nogold_prompts_tatqa.py --prompts G/prompts.jsonl --contexts G/context_verdicts.jsonl \
        --n 800 --out dpo/data/heldout_tatqa/clones_nogold.jsonl
"""
import argparse
import collections
import hashlib
import json
import random
import re

MARKER = "Here are the retrieved documents : `"
SEPARATOR = "\n" + "-" * 10 + "\n\n"  # formatting_sft.CONTEXT_SEPARATOR
TITLE_RE = re.compile(r"\* filename: (.+?)\n\n\[CHUNK_START\]\n")


def rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def split_chunks(system):
    """(head, [(title, chunk_text)]) or None when the block does not parse back identically."""
    i = system.find(MARKER)
    if i < 0 or not system.endswith("`"):
        return None
    head, block = system[: i + len(MARKER)], system[i + len(MARKER): -1]
    chunks = []
    for c in block.split(SEPARATOR):
        titles = TITLE_RE.findall(c)
        if len(titles) != 1 or not c.endswith("[CHUNK_END]"):
            return None
        chunks.append((titles[0], c))
    return head, chunks


def renumber(chunks):
    """Table/Paragraph titles renumbered 1..N in context order. Returns (chunks, old->new)."""
    count, mapping, out = collections.Counter(), {}, []
    for title, c in chunks:
        kind = title.split(" ", 1)[0]
        if kind not in ("Table", "Paragraph"):
            raise ValueError(f"unexpected title {title!r}")
        count[kind] += 1
        new = f"{kind} {count[kind]}"
        mapping[title] = new
        out.append((new, c.replace(f"* filename: {title}\n\n[CHUNK_START]", f"* filename: {new}\n\n[CHUNK_START]", 1)))
    return out, mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="generic prompts (pilot_split, judge-facing gold)")
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--partial-ratio", type=float, default=0.5)
    ap.add_argument("--partial-remove", choices=("table", "both"), default="table",
                    help="what a partial clone loses: the gold table (default) or, coin flip, table or paragraphs")
    ap.add_argument("--min-docs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    verdict = {}
    for r in rows(args.contexts):
        verdict[r["pid"]] = r["verdict"]  # last row of a pid wins (retries)
    pool = [p for p in rows(args.prompts) if not p["is_unanswerable"] and not p.get("is_strict_variant")
            and not p.get("strict_refusal_instruction") and verdict.get(p["pid"]) == "answerable"]
    pool.sort(key=lambda p: p["pid"])
    rng.shuffle(pool)

    out, skipped = [], collections.Counter()
    for p in pool:
        if len(out) >= args.n:
            break
        system = p["messages"][0]["content"]
        parsed = split_chunks(system)
        if parsed is None:
            skipped["unparsed_context"] += 1
            continue
        head, chunks = parsed
        gold = list(dict.fromkeys(p["gold_titles"]))
        titles = [t for t, _ in chunks]
        if any(titles.count(g) != 1 for g in gold) or len(set(titles)) != len(titles):
            skipped["gold_not_found_once"] += 1
            continue
        g_tab = [g for g in gold if g.startswith("Table")]
        g_par = [g for g in gold if g.startswith("Paragraph")]
        mode, removed_kind = "drop_gold", "all"
        if p["answer_from"] == "table-text" and g_tab and g_par and rng.random() < args.partial_ratio:
            mode = "partial_hop"
            removed_kind = rng.choice(["table", "text"]) if args.partial_remove == "both" else "table"
        removed = set(gold) if mode == "drop_gold" else set(g_tab if removed_kind == "table" else g_par)
        kept = [(t, c) for t, c in chunks if t not in removed]
        if len(kept) < args.min_docs:
            skipped["too_few_docs_left"] += 1
            continue
        kept, mapping = renumber(kept)
        new_system = head + SEPARATOR.join(c for _, c in kept) + "`"
        user = p["messages"][1]["content"]
        out.append({
            **{k: v for k, v in p.items() if k not in ("sft_target", "strict_phrase", "strict_rule_id")},
            "pid": md5(new_system + "\x00" + user)[:16],
            "messages": [{"role": "system", "content": new_system}, p["messages"][1]],
            "sft_target": None,
            "is_unanswerable": True,
            "unans_mode": mode,
            "removed_gold": removed_kind,
            "gold_titles": [mapping[g] for g in gold if g not in removed],
            "raw_gold_titles": gold,
            "removed_gold_titles": sorted(removed),
            "context_titles": [t for t, _ in kept],
            "sft_target_short_refusal": False,
            "sft_target_reliable": False,
            "heldout_drop_reason": "nogold_clone",
            "clone_source": "nogold",
            "source_pid": p["pid"],
            "is_strict_variant": False,
        })
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"prompts": args.prompts, "contexts": args.contexts, "seed": args.seed, "n": len(out),
           "partial_ratio": args.partial_ratio, "partial_remove": args.partial_remove,
           "eligible_sources": len(pool), "skipped": dict(skipped),
           "unans_mode": dict(collections.Counter(f'{r["unans_mode"]}/{r["removed_gold"]}' for r in out)),
           "answer_type_x_from": dict(collections.Counter(f'{r["answer_type"]}|{r["answer_from"]}' for r in out)),
           "split": dict(collections.Counter(r["pilot_split"] for r in out)),
           "docs_left": dict(sorted(collections.Counter(len(r["context_titles"]) for r in out).items()))}
    with open(args.out.removesuffix(".jsonl") + ".manifest.json", "w") as f:
        f.write(json.dumps(man, indent=2) + "\n")
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
