#!/usr/bin/env python3
"""Create conservative, same-prompt DPO pairs from judged policy samples."""
import argparse
import collections
import difflib
import json
import re
from pathlib import Path


def rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def normalized_answer(text):
    text = re.sub(r'##Cite\s+"[^"]+"##|<cite>.*?</cite>|\[Source:.*?\]|<<cite:.*?>>', '', text)
    return " ".join(text.lower().split())


def squash(text):
    """Quote-insensitive, case-insensitive, whitespace-collapsed form for substring checks."""
    return " ".join(re.sub(r"[\"'`\u201c\u201d\u2018\u2019]", "", text).casefold().split())


A_GOOD_KEYS = ("answer_correct", "answer_supported", "reasoning_supported",
               "citation_supported", "citation_style_ok", "support_quote_in_context")


def fix_quote(judge, ctx):
    """The judge's quote often differs from the document by quotes or spacing only."""
    if judge is not None:
        q = judge.get("support_quote", "")
        judge["support_quote_in_context"] = bool(q.strip()) and squash(q) in ctx


def is_good(judge, unanswerable):
    if unanswerable:
        return judge["refused"] and judge["missing_fact_correct"]
    return not judge["refused"] and not judge["mixed"] and all(judge[k] for k in A_GOOD_KEYS)


def chosen_order(s):
    return (s["k"] != -1, -s["judge"].get("support_quote_in_context", False), s["k"])


def candidates(p, ss, verified=None):
    """Chosen and rejected candidates of one prompt from first-pass judgments.

    `verified` maps (pid, k) to a second-pass (reasoning) judgment: a chosen must then be
    confirmed by it (the first pass is lenient, its "bad" verdicts are kept as they are).
    Returns None when fewer than two samples are usable."""
    ctx = squash(p["messages"][0]["content"])
    for s in ss:
        fix_quote(s["judge"], ctx)
    valid = [s for s in ss if s["judge"] is not None and s["finish_reason"] != "length"
             and not s["judge"]["mixed"] and s["text"].strip()]
    if len(valid) < 2:
        return None
    u = p["is_unanswerable"]
    good_first = sorted([s for s in valid if is_good(s["judge"], u)], key=chosen_order)
    good, unverified = good_first, []
    if verified is not None:
        good, unverified = [], []
        for s in good_first:
            v = verified.get((p["pid"], s["k"]))
            if v is None:
                unverified.append(s)
            elif v["judge"] is not None:
                fix_quote(v["judge"], ctx)
                if is_good(v["judge"], u):
                    good.append({**s, "judge": v["judge"], "first_pass_judge": s["judge"]})
    out = {"good": good, "good_first_pass": good_first, "good_unverified": unverified}
    if u:
        out["U_unsupported_answer"] = [s for s in valid if not s["judge"]["refused"] and not s["judge"]["answer_supported"]]
        out["bad_keys"] = ("U_unsupported_answer",)
        return out
    j = lambda s: s["judge"]  # noqa: E731
    out["A_false_refusal"] = [s for s in valid if j(s)["refused"]]
    out["A_wrong_reasoning"] = [s for s in valid if not j(s)["refused"] and (
        not j(s)["answer_correct"] or not j(s)["answer_supported"] or not j(s)["reasoning_supported"])]
    cit_bad = [s for s in valid if not j(s)["refused"]
               and all(j(s)[k] for k in ("answer_correct", "answer_supported", "reasoning_supported"))
               and (not j(s)["citation_supported"] or not j(s)["citation_style_ok"])]
    ref = good_first
    out["A_bad_citation"] = [s for s in cit_bad if ref and max(difflib.SequenceMatcher(
        None, normalized_answer(s["text"]), normalized_answer(g["text"])).ratio() for g in ref) >= 0.65]
    out["bad_keys"] = ("A_false_refusal", "A_wrong_reasoning", "A_bad_citation")
    return out


def pick_rejected(candidates, chosen):
    return min(candidates, key=lambda r: (r["k"] != -1, abs(r["n_chars"] - chosen["n_chars"]), r["k"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--verified", default=None,
                    help="second-pass (reasoning) judgments: chosen must be confirmed there")
    args = ap.parse_args()
    verified = None
    if args.verified:
        verified = {(r["pid"], r["k"]): r for r in rows(args.verified)}
    prompts = {p["pid"]: p for p in rows(args.prompts)}
    contexts = {r["pid"]: r for r in rows(args.contexts)}
    scored = collections.defaultdict(dict)
    for r in rows(args.scored):
        scored[r["pid"]][r["k"]] = r
    pairs = []
    excluded = collections.Counter()

    for pid, p in prompts.items():
        verdict = contexts.get(pid, {}).get("verdict")
        wanted = "unanswerable" if p["is_unanswerable"] else "answerable"
        if verdict != wanted:
            excluded[f"context_{verdict}"] += 1
            continue
        cand = candidates(p, list(scored[pid].values()), verified)
        if cand is None:
            excluded["fewer_than_two_valid_samples"] += 1
            continue
        def add(category, good, bad, **extra):
            if not good or not bad:
                return
            c = min(good, key=chosen_order)
            r = pick_rejected([s for s in bad if s["text_sha256"] != c["text_sha256"]], c) if any(
                s["text_sha256"] != c["text_sha256"] for s in bad) else None
            if r is None:
                return
            pairs.append({"prompt": p["messages"],
                          "chosen": [{"role": "assistant", "content": c["text"]}],
                          "rejected": [{"role": "assistant", "content": r["text"]}],
                          "pid": pid, "qid": p["qid"], "split": p["pilot_split"],
                          "category": category, "model": args.model,
                          "chosen_source": "policy", "rejected_source": "policy",
                          "chosen_k": c["k"], "rejected_k": r["k"],
                          "rejected_is_greedy": r["k"] == -1,
                          "len_chosen": c["n_chars"], "len_rejected": r["n_chars"],
                          "chosen_judge": c["judge"], "rejected_judge": r["judge"],
                          "chosen_verified": verified is not None,
                          "hotpot_type": p["hotpot_type"], "unans_mode": p["unans_mode"], **extra})

        good = cand["good"]
        if p["is_unanswerable"]:
            add("U_unsupported_answer", good, cand["U_unsupported_answer"])
            if not good:
                excluded["U_no_grounded_refusal"] += 1
        else:
            for cat in ("A_false_refusal", "A_wrong_reasoning", "A_bad_citation"):
                add(cat, good, cand[cat])
            if not good:
                excluded["A_no_fully_supported_answer"] += 1
        if verified is not None and not good and cand["good_first_pass"] and any(cand[c] for c in cand["bad_keys"]):
            excluded["chosen_not_confirmed_by_verifier"] += 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in pairs:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"model": args.model, "n_pairs": len(pairs), "excluded": dict(excluded),
               "category": dict(collections.Counter(r["category"] for r in pairs)),
               "split": dict(collections.Counter(r["split"] for r in pairs)),
               "chosen_source": dict(collections.Counter(r["chosen_source"] for r in pairs)),
               "rejected_greedy": sum(r["rejected_is_greedy"] for r in pairs),
               "length_by_category": {cat: {"n": len(xs),
                       "chosen_longer": sum(x["len_chosen"] > x["len_rejected"] for x in xs)}
                   for cat in sorted({r["category"] for r in pairs})
                   for xs in [[r for r in pairs if r["category"] == cat]]}}
    (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
