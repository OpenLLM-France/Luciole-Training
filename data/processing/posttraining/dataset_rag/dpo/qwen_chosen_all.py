#!/usr/bin/env python3
"""Variant of the DPO data where EVERY chosen is written by Qwen3-32B (rejected stay the policy's).

Covers the pairs of heldout_full_sft4152/pairs_v1.1 and heldout_strict_sft4152/pairs. Each pair
prompt gets a chosen kind:
  answer   A_* pairs (generic and strict): a Qwen answer confirmed by the answer judge
           (score_pilot_samples: pass 1 without reasoning, confirmation with reasoning;
           build_pilot_pairs.is_good on the confirmed verdict)
  refusal  generic U_unsupported_answer: a Qwen refusal accepted by judge_refusal_candidates
  phrase   strict U_* pairs: the imposed phrase, exact (lexical, no judge)
Greedy first, then the two samples of the prompts still without an accepted chosen.

An answer chosen must also cite like the policy does (citation_ok): the requested markup and
only it, titles of the retrieved documents only, and with end-of-completion placement nothing
but references after the first citation. The judge lets uncited or malformed answers through
(35 % of the Qwen answers); uncited samples are not even sent to the judge.

  prompts  write D/prompts.jsonl (+ context_verdicts.jsonl) from the two sources
  todo     list of (pid, k) for a judging stage:
             --stage pass1   answer prompts, samples --ks not yet judged in pass 1
             --stage verify  answer prompts without a confirmed chosen: next pass-1-good sample
             --stage refusal refusal prompts without an accepted chosen: samples --ks
  build    D/pairs/{train,val}.jsonl: the source pairs with the Qwen chosen, + report.json

Global options (before the command; defaults = the Qwen3-32B run of 2026-09-30, unchanged):
  --gen NAME            generator name (samples file chosen_NAME.jsonl), e.g. qwen35_35b_a3b
  --source DIR:SUB      pair source, repeatable (sources must not share pids: one D per group)
  --prefer-quoted       answer chosen: a Qwen answer carrying the quote marker of the requested style
                        (dpo_common.STYLE_MARKERS) is preferred, and judged further until one is confirmed
                        when only unquoted ones are. Qwen3.5 quotes in 30 % of its answers, the policy in
                        81 %: without it the pairs would teach "no quote marker = better".

    python dpo/qwen_chosen_all.py prompts --out-dir D
    python dpo/qwen_chosen_all.py todo --dir D --stage pass1 --ks=-1 --out D/todo.jsonl
    python dpo/qwen_chosen_all.py build --dir D
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_pilot_pairs import chosen_order, fix_quote, is_good, rows, squash  # noqa: E402
from build_strict_pairs import is_exact, usable  # noqa: E402
from dpo_common import STYLE_MARKERS, _CITE_PATTERNS, _norm_title, _title_match, extract_citations  # noqa: E402

SOURCES = [("dpo/data/heldout_full_sft4152", "pairs_v1.1"), ("dpo/data/heldout_strict_sft4152", "pairs")]
GEN = "qwen3_32b"
PREFER_QUOTED = False


def quoted(p, text):
    return STYLE_MARKERS[p["citation_style"]] in text


def satisfied(p, acc):
    """Enough accepted chosen to stop judging this prompt."""
    if not acc:
        return False
    return not (PREFER_QUOTED and p["chosen_kind"] == "answer") or any(quoted(p, s["text"]) for s in acc)
# Not dpo_common._FILENAME_RE: anchored at line start, it misses the first document, whose
# "* filename:" follows "Here are the retrieved documents : `" on the same line (so does the
# prompts' context_titles field).
DOC_TITLE_RE = re.compile(r"\* filename: (.+?)\s*$", re.M)
REFERENCES_TAIL_RE = re.compile(r"[\s*#:\-•]*(references|sources)?[\s*#:\-•]*", re.I)


def citation_ok(p, text):
    """Cites like the policy (99 % of its chosen answers pass): the requested markup only,
    retrieved titles only, and with eoc placement nothing but references after the first one."""
    style = p["citation_style"]
    cited = extract_citations(text)
    if set(cited) != {style}:
        return False
    titles = [_norm_title(t) for t in DOC_TITLE_RE.findall(p["messages"][0]["content"])]
    if not all(any(_title_match(_norm_title(c), t) for t in titles) for c in cited[style]):
        return False
    if p["citation_placement"] == "eoc":
        pat = _CITE_PATTERNS[style]
        tail = pat.sub(" ", text[pat.search(text).start():])
        return REFERENCES_TAIL_RE.fullmatch(tail) is not None
    return True


def kind(pair):
    if pair["category"].startswith("A_"):
        return "answer"
    return "phrase" if pair.get("is_strict_variant") else "refusal"


def load_pairs():
    for src, sub in SOURCES:
        for split in ("train", "val"):
            for p in rows(Path(src) / sub / f"{split}.jsonl"):
                yield src, p


def last_by_key(path):
    out = {}
    if Path(path).exists():
        for r in rows(path):
            out[(r["pid"], r["k"])] = r
    return out


def state(d):
    prompts = {p["pid"]: p for p in rows(d / "prompts.jsonl")}
    for p in prompts.values():  # prompts of select_qwen_nochosen.py (E) carry no chosen_kind
        p.setdefault("chosen_kind", "refusal" if p["is_unanswerable"] else "answer")
    samples = {r["pid"]: {s["k"]: s for s in r["samples"]} for r in rows(d / f"chosen_{GEN}.jsonl")
               if r.get("samples") and not r.get("errors")}
    return prompts, samples, last_by_key(d / "scored.jsonl"), last_by_key(d / "verified.jsonl"), \
        last_by_key(d / f"judged_chosen_{GEN}.jsonl")


def accepted(p, ss, scored, verified, refj, check_citation=True):
    """Accepted Qwen chosen of one prompt (greedy first), or []."""
    k = p["chosen_kind"]
    ok = []
    order = (lambda s: (not quoted(p, s["text"]), s["k"] != -1, s["k"])) if PREFER_QUOTED and k == "answer" \
        else (lambda s: (s["k"] != -1, s["k"]))
    for s in sorted(ss.values(), key=order):
        if not usable(s) or (k == "answer" and check_citation and not citation_ok(p, s["text"])):
            continue
        if k == "phrase":
            if is_exact(s["text"], p["strict_phrase"]):
                ok.append({**s, "judge": {"lexical_imposed_phrase": True}})
        elif k == "refusal":
            r = refj.get((p["pid"], s["k"]))
            if r and r["judge"] and r["judge"]["good_chosen"]:
                ok.append({**s, "judge": r["judge"]})
        else:
            v = verified.get((p["pid"], s["k"]))
            if v and v["judge"]:
                fix_quote(v["judge"], squash(p["messages"][0]["content"]))
                if is_good(v["judge"], False):
                    ok.append({**s, "judge": v["judge"]})
    return ok


def cmd_prompts(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kinds = collections.defaultdict(set)
    src_of = {}
    for src, p in load_pairs():
        kinds[p["pid"]].add(kind(p))
        src_of[p["pid"]] = src
    assert all(len(v) == 1 for v in kinds.values()), "a prompt with two chosen kinds"
    by_src = collections.defaultdict(set)
    for pid, src in src_of.items():
        by_src[src].add(pid)
    n = 0
    with open(out / "prompts.jsonl", "w", encoding="utf-8") as fp, open(out / "context_verdicts.jsonl", "w", encoding="utf-8") as fc:
        for src, pids in by_src.items():
            ctx = {r["pid"]: r for r in rows(Path(src) / "context_verdicts.jsonl")}
            for p in rows(Path(src) / "prompts.jsonl"):
                if p["pid"] in pids:
                    fp.write(json.dumps({**p, "chosen_kind": next(iter(kinds[p["pid"]])), "pair_source": src},
                                        ensure_ascii=False) + "\n")
                    fc.write(json.dumps(ctx[p["pid"]], ensure_ascii=False) + "\n")
                    n += 1
    print(json.dumps({"prompts": n, "kinds": dict(collections.Counter(next(iter(v)) for v in kinds.values()))}))


def cmd_todo(args):
    d = Path(args.dir)
    prompts, samples, scored, verified, refj = state(d)
    ks = {int(k) for k in args.ks.split(",")} if args.ks else None
    todo, stats = [], collections.Counter()
    for pid, p in prompts.items():
        ss = samples.get(pid)
        if not ss:
            continue
        acc = accepted(p, ss, scored, verified, refj)
        if satisfied(p, acc):
            stats[f"{p['chosen_kind']}_accepted"] += 1
            continue
        # accepted but unquoted (--prefer-quoted): judge further, quoted samples only
        want = (lambda s: quoted(p, s["text"])) if acc else (lambda s: True)  # noqa: E731
        stats["answer_accepted_unquoted_searching"] += bool(acc)
        if args.stage == "pass1" and p["chosen_kind"] == "answer":
            todo += [{"pid": pid, "k": s["k"]} for s in ss.values()
                     if s["k"] in ks and usable(s) and citation_ok(p, s["text"])
                     and not scored.get((pid, s["k"]), {}).get("judge")
                     and want(s)]
        elif args.stage == "refusal" and p["chosen_kind"] == "refusal":
            todo += [{"pid": pid, "k": s["k"]} for s in ss.values() if s["k"] in ks and usable(s)
                     and ((pid, s["k"]) not in refj or refj[(pid, s["k"])]["judge"] is None)]
        elif args.stage == "verify" and p["chosen_kind"] == "answer":
            ctx = squash(p["messages"][0]["content"])
            cands = []
            for s in ss.values():
                j = scored.get((pid, s["k"]), {}).get("judge")
                if j is None or verified.get((pid, s["k"]), {}).get("judge") is not None \
                        or not citation_ok(p, s["text"]) or not want(s):
                    continue
                fix_quote(j, ctx)
                if is_good(j, False):
                    cands.append({**s, "judge": j})
            if cands:
                key = (lambda s: (not quoted(p, s["text"]), chosen_order(s))) if PREFER_QUOTED else chosen_order
                todo.append({"pid": pid, "k": min(cands, key=key)["k"]})
    with open(args.out, "w") as f:
        for t in todo:
            f.write(json.dumps(t) + "\n")
    print(json.dumps({**stats, "stage": args.stage, "todo": len(todo)}))


def form_by_category(pairs, prompts):
    """Per category: share of quote markers and median length, chosen vs rejected (shortcut check)."""
    out = {}
    for c in sorted({r["category"] for r in pairs}):
        rs = [r for r in pairs if r["category"] == c]
        med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
        out[c] = {"n": len(rs),
                  "quoted_chosen": round(sum(quoted(prompts[r["pid"]], r["chosen"][0]["content"]) for r in rs) / len(rs), 3),
                  "quoted_rejected": round(sum(quoted(prompts[r["pid"]], r["rejected"][0]["content"]) for r in rs) / len(rs), 3),
                  "median_len_chosen": med([r["len_chosen"] for r in rs]),
                  "median_len_rejected": med([r["len_rejected"] for r in rs])}
    return out


def cmd_build(args):
    d = Path(args.dir)
    prompts, samples, scored, verified, refj = state(d)
    pairs, missing, uncited = [], collections.Counter(), collections.Counter()
    for src, p in load_pairs():
        q = prompts[p["pid"]]
        differs = lambda ss: [s for s in ss if s["text"].strip() != p["rejected"][0]["content"].strip()]  # noqa: E731
        ok = differs(accepted(q, samples.get(p["pid"], {}), scored, verified, refj))
        if not ok:
            missing[p["category"]] += 1
            uncited[p["category"]] += bool(differs(accepted(q, samples.get(p["pid"], {}), scored, verified,
                                                            refj, check_citation=False)))
            continue
        c = ok[0]
        # The policy's samples start with a space, Qwen's do not: left as is, the first token
        # alone would tell chosen from rejected. Both sides are stripped.
        chosen, rejected = c["text"].lstrip(), p["rejected"][0]["content"].lstrip()
        pairs.append({**p, "chosen": [{"role": "assistant", "content": chosen}],
                      "rejected": [{"role": "assistant", "content": rejected}],
                      "chosen_source": GEN, "chosen_k": c["k"], "len_chosen": len(chosen),
                      "len_rejected": len(rejected), "chosen_judge": c["judge"], "chosen_verified": True,
                      "policy_chosen": p["chosen"][0]["content"], "pair_source": src})
    # Same balance as build_strict_pairs: the imposed phrase as often chosen (strict U pairs) as
    # rejected (strict A_false_refusal whose rejected is the bare phrase). Losing chosen breaks it
    # either way; the surplus side is cut by pid hash (U_phrase_over_grounded first, the pairs
    # added only for this balance).
    balance = collections.Counter()
    if not args.no_balance:
        phrase_chosen = [r for r in pairs if r["category"].startswith("U_") and r.get("is_strict_variant")]
        phrase_rejected = [r for r in pairs if r["category"] == "A_false_refusal" and r.get("rejected_is_exact_phrase")]
        by_hash = lambda rs: sorted(rs, key=lambda r: hashlib.sha256(r["pid"].encode()).hexdigest())  # noqa: E731
        if len(phrase_chosen) > len(phrase_rejected):
            surplus = len(phrase_chosen) - len(phrase_rejected)
            pog = [r for r in phrase_chosen if r["category"] == "U_phrase_over_grounded"]
            drop = by_hash(pog)[:surplus]
        else:
            drop = by_hash(phrase_rejected)[len(phrase_chosen):]
        balance.update(r["category"] for r in drop)
        ids = {id(r) for r in drop}
        pairs = [r for r in pairs if id(r) not in ids]
    out = d / "pairs"
    out.mkdir(exist_ok=True)
    for split in ("train", "val"):
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in pairs:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = collections.Counter(p["category"] for _, p in load_pairs())
    report = {"generator": GEN, "sources": [f"{a}:{b}" for a, b in SOURCES], "n_pairs": len(pairs), "source_pairs": sum(total.values()),
              "category": dict(collections.Counter(r["category"] for r in pairs)),
              "without_qwen_chosen": dict(missing),
              "lost_to_citation_rule": {c: n for c, n in uncited.items() if n},
              "dropped_for_phrase_balance": dict(balance),
              "phrase_chosen": sum(r["category"].startswith("U_") and bool(r.get("is_strict_variant")) for r in pairs),
              "phrase_rejected": sum(r["category"] == "A_false_refusal" and bool(r.get("rejected_is_exact_phrase"))
                                     for r in pairs),
              "coverage": {c: round(1 - missing[c] / n, 3) for c, n in total.items()},
              "chosen_greedy": sum(r["chosen_k"] == -1 for r in pairs),
              "prefer_quoted": PREFER_QUOTED,
              "form_by_category": form_by_category(pairs, prompts),
              "split": dict(collections.Counter(r["split"] for r in pairs))}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    global GEN, SOURCES, PREFER_QUOTED
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default=GEN)
    ap.add_argument("--source", action="append", default=None, help="DIR:SUB, repeatable")
    ap.add_argument("--prefer-quoted", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prompts")
    s.add_argument("--out-dir", required=True)
    s = sub.add_parser("todo")
    s.add_argument("--dir", required=True)
    s.add_argument("--stage", choices=("pass1", "verify", "refusal"), required=True)
    s.add_argument("--ks", default=None)
    s.add_argument("--out", required=True)
    s = sub.add_parser("build")
    s.add_argument("--dir", required=True)
    s.add_argument("--no-balance", action="store_true", help="keep the imposed phrase chosen/rejected imbalance")
    args = ap.parse_args()
    GEN, PREFER_QUOTED = args.gen, args.prefer_quoted
    if args.source:
        SOURCES = [tuple(x.split(":", 1)) for x in args.source]
    {"prompts": cmd_prompts, "todo": cmd_todo, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
