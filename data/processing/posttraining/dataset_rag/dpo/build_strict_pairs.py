#!/usr/bin/env python3
"""DPO pairs on strict-instruction prompts (imposed refusal phrase), heldout_strict_sft4152.

Categories (at most one pair per category and prompt):
  A_false_refusal       answerable: chosen = a fully supported answer confirmed by the reasoning
                        judge, rejected = a refusal (the imposed phrase, or judged refused and not
                        mixed). With --twin-dir, when no answer of the strict prompt is confirmed
                        (the policy refused in all its samples), the chosen is the answer of the
                        GENERIC twin confirmed in the full run (chosen_source policy_generic_twin)
  U_strict_inexact      unanswerable: chosen = the imposed phrase (lexical), rejected = a refusal
                        judged as such that is not the phrase and does not contain it (a grounded
                        body followed by the phrase is the v4 refusal style, never rejected)
  U_unsupported_answer  unanswerable: chosen = the imposed phrase, rejected = an answer judged
                        unsupported (not refused, not mixed, answer_supported false)
  U_phrase_over_grounded (--twin-dir) unanswerable, when neither category above applies: chosen =
                        the imposed phrase, rejected = a grounded refusal the policy wrote on the
                        GENERIC twin (same question and documents; judged a refusal, not mixed, in
                        pass 1 of the full run; at least 160 characters, i.e. not a bare phrase). Under the strict rule the policy almost always outputs the
                        phrase, so natural U pairs are rare; these pairs keep the phrase chosen as
                        often as it is rejected, and with the generic pairs (grounded refusal chosen)
                        they teach to follow the refusal instruction.

Balance (--no-balance to disable): the imposed phrase must not be rejected more often than
chosen. A_false_refusal pairs whose rejected is the bare phrase are capped at the number of
pairs where the phrase is chosen (all U pairs), dropping by pid hash. U_phrase_over_grounded
pairs are added (by pid hash) only as far as needed to reach that balance. Without it the policy
learns to avoid the phrase even when it is imposed (hpv4 v1.1: phrase recall 0.66 -> 0.04).

  select  samples to judge in pass 1 (no reasoning): on answerable prompts where the policy wrote
          the imposed phrase at least once (at most --max-a prompts, by pid hash), the samples
          without the phrase (a response containing the imposed phrase is a refusal, not judged);
          on unanswerable prompts with an exact phrase, the other samples
  verify  next chosen candidate to confirm with reasoning, for answerable prompts that have a
          judged refusal and no confirmed chosen yet (greedy first); run in rounds
  build   pairs + report

    python dpo/build_strict_pairs.py select --dir D --out D/pass1_only.jsonl
    python dpo/build_strict_pairs.py verify --dir D --out D/verify_round1.jsonl
    python dpo/build_strict_pairs.py build --dir D --model sft_luciole_1B_instruct_newrag-step_4152
"""
import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_pilot_pairs import chosen_order, fix_quote, is_good, rows, squash  # noqa: E402
from dpo_common import lexical_refusal  # noqa: E402


def is_exact(text, phrase):
    """Same rule as build_pairs.is_exact_refusal: the phrase, nothing substantial around."""
    t, ph = " ".join(text.lower().split()), " ".join(phrase.lower().split())
    return ph in t and len(t) <= len(ph) + 40


def contains_phrase(text, phrase):
    return " ".join(phrase.lower().split()) in " ".join(text.lower().split())


def lexical_phrase_only(text):
    """A bare canned refusal (no grounded body): not a grounded refusal."""
    return len(" ".join(text.split())) < 160


def usable(s):
    return bool(s["text"].strip()) and s["finish_reason"] != "length"


def load(d):
    prompts = {p["pid"]: p for p in rows(d / "prompts.jsonl")}
    samples = {r["pid"]: r["samples"] for r in rows(d / "samples.jsonl") if r.get("samples") and not r.get("errors")}
    scored = collections.defaultdict(dict)
    if (d / "scored.jsonl").exists():
        for r in rows(d / "scored.jsonl"):
            scored[r["pid"]][r["k"]] = r
    verified = {}
    if (d / "verified.jsonl").exists():
        for r in rows(d / "verified.jsonl"):
            verified[(r["pid"], r["k"])] = r
    return prompts, samples, scored, verified


def judged(scored, pid, k):
    r = scored.get(pid, {}).get(k)
    return r["judge"] if r is not None else None


def a_state(p, ss, scored, verified):
    """Answerable prompt: judged refusals, first-pass good answers, confirmed chosen."""
    ctx = squash(p["messages"][0]["content"])
    refusals, good = [], []
    for s in ss:
        if usable(s) and contains_phrase(s["text"], p["strict_phrase"]):
            refusals.append({**s, "judge": {"lexical_imposed_phrase": True}})  # unambiguous, not judged
            continue
        j = judged(scored, p["pid"], s["k"])
        if not usable(s) or j is None or j["mixed"]:
            continue
        fix_quote(j, ctx)
        if j["refused"]:
            refusals.append({**s, "judge": j})
        elif is_good(j, False):
            good.append({**s, "judge": j})
    good.sort(key=chosen_order)
    confirmed, unverified = [], []
    for s in good:
        v = verified.get((p["pid"], s["k"]))
        if v is None:
            unverified.append(s)
        elif v["judge"] is not None:
            fix_quote(v["judge"], ctx)
            if is_good(v["judge"], False):
                confirmed.append({**s, "judge": v["judge"], "first_pass_judge": s["judge"]})
    return refusals, good, confirmed, unverified


def cmd_select(args):
    d = Path(args.dir)
    prompts, samples, _, _ = load(d)
    todo, stats, a_candidates = [], collections.Counter(), []
    for pid, ss in samples.items():
        p = prompts[pid]
        ok = [s for s in ss if usable(s)]
        if not p["is_unanswerable"]:
            if any(contains_phrase(s["text"], p["strict_phrase"]) for s in ok):
                a_candidates.append((pid, [s for s in ok if not contains_phrase(s["text"], p["strict_phrase"])]))
        else:
            exact = [s for s in ok if is_exact(s["text"], p["strict_phrase"])]
            rest = [s for s in ok if not is_exact(s["text"], p["strict_phrase"])]
            if exact and rest:
                todo += [{"pid": pid, "k": s["k"]} for s in rest]
                stats["U_prompts_with_exact_and_other"] += 1
            stats["U_prompts_exact" if exact else "U_prompts_no_exact"] += 1
    stats["A_prompts_with_imposed_phrase"] = len(a_candidates)
    a_candidates.sort(key=lambda t: hashlib.sha256(t[0].encode()).hexdigest())
    for pid, rest in a_candidates[:args.max_a]:  # the balance caps A_false_refusal anyway
        todo += [{"pid": pid, "k": s["k"]} for s in rest]
    stats["A_prompts_selected"] = min(len(a_candidates), args.max_a)
    with open(args.out, "w") as f:
        for t in todo:
            f.write(json.dumps(t) + "\n")
    print(json.dumps({**stats, "samples_to_judge": len(todo)}))


def cmd_verify(args):
    d = Path(args.dir)
    prompts, samples, scored, verified = load(d)
    todo, stats = [], collections.Counter()
    for pid, ss in samples.items():
        p = prompts[pid]
        if p["is_unanswerable"]:
            continue
        refusals, good, confirmed, unverified = a_state(p, ss, scored, verified)
        if not refusals:
            continue
        if confirmed:
            stats["confirmed"] += 1
        elif unverified:
            todo.append({"pid": pid, "k": unverified[0]["k"]})
            stats["to_verify"] += 1
        else:
            stats["exhausted" if good else "no_first_pass_good"] += 1
    with open(args.out, "w") as f:
        for t in todo:
            f.write(json.dumps(t) + "\n")
    print(json.dumps(dict(stats)))


def pick(cands, ref_len):
    return min(cands, key=lambda s: (s["k"] != -1, abs(len(s["text"]) - ref_len), s["k"]))


def cmd_build(args):
    d = Path(args.dir)
    prompts, samples, scored, verified = load(d)
    pairs, excluded = [], collections.Counter()
    twin_answers = {}  # base pid -> answers confirmed by the reasoning judge on the generic twin
    if args.twin_dir:
        tw = Path(args.twin_dir)
        tw_prompts = {q["pid"]: q for q in rows(tw / "prompts.jsonl")}
        for r in rows(tw / "verified.jsonl"):
            q = tw_prompts.get(r["pid"])
            if q is None or q["is_unanswerable"] or r["judge"] is None or not usable(r):
                continue
            fix_quote(r["judge"], squash(q["messages"][0]["content"]))
            if is_good(r["judge"], False):
                twin_answers.setdefault(r["pid"], []).append(r)

    def add(p, cat, c, r):
        pairs.append({"prompt": p["messages"], "chosen": [{"role": "assistant", "content": c["text"]}],
                      "rejected": [{"role": "assistant", "content": r["text"]}],
                      "pid": p["pid"], "qid": p["qid"], "split": p["pilot_split"], "category": cat,
                      "model": args.model, "chosen_source": "policy", "rejected_source": "policy",
                      "chosen_k": c["k"], "rejected_k": r["k"], "rejected_is_greedy": r["k"] == -1,
                      "len_chosen": len(c["text"]), "len_rejected": len(r["text"]),
                      "chosen_judge": c.get("judge"), "rejected_judge": r.get("judge"),
                      "chosen_verified": cat == "A_false_refusal",
                      "hotpot_type": p["hotpot_type"], "unans_mode": p["unans_mode"],
                      "is_strict_variant": True, "strict_phrase": p["strict_phrase"],
                      "rejected_is_exact_phrase": is_exact(r["text"], p["strict_phrase"])})

    for pid, ss in samples.items():
        p = prompts[pid]
        phrase = p["strict_phrase"]
        if not p["is_unanswerable"]:
            refusals, good, confirmed, _ = a_state(p, ss, scored, verified)
            if not refusals:
                continue
            src = "policy"
            if not confirmed and twin_answers.get(p["base_pid"]):
                confirmed, src = twin_answers[p["base_pid"]], "policy_generic_twin"
            if not confirmed:
                excluded["A_no_confirmed_answer"] += 1
                continue
            c = min(confirmed, key=chosen_order)
            add(p, "A_false_refusal", c, pick(refusals, len(c["text"])))
            pairs[-1]["chosen_source"] = src
            continue
        ok = [s for s in ss if usable(s)]
        exact = [s for s in ok if is_exact(s["text"], phrase)]
        if not exact:
            excluded["U_no_exact_phrase"] += 1
            continue
        inexact, answers = [], []
        for s in ok:
            if is_exact(s["text"], phrase):
                continue
            j = judged(scored, pid, s["k"])
            if j is None or j["mixed"]:
                continue
            if j["refused"]:
                if contains_phrase(s["text"], phrase):
                    excluded["U_body_plus_phrase_not_rejected"] += 1
                else:
                    inexact.append({**s, "judge": j})
            elif not j["answer_supported"]:
                answers.append({**s, "judge": j})
        for cat, bad in (("U_strict_inexact", inexact), ("U_unsupported_answer", answers)):
            if bad:
                r = pick(bad, len(exact[0]["text"]))
                add(p, cat, pick(exact, len(r["text"])), r)

    phrase_rejected_n = sum(p["category"] == "A_false_refusal" and p["rejected_is_exact_phrase"] for p in pairs)
    if args.twin_dir:
        tw = Path(args.twin_dir)
        tw_prompts = {p["pid"]: p for p in rows(tw / "prompts.jsonl")}
        tw_ok = {}  # base pid -> refusals of the generic twin (pass 1 is enough for a rejected)
        for r in rows(tw / "scored.jsonl"):
            j = r["judge"]
            bp = tw_prompts.get(r["pid"])
            if bp and bp["is_unanswerable"] and j and j["refused"] and not j["mixed"] and usable(r):
                tw_ok.setdefault(r["pid"], {})[r["k"]] = r  # last row of a (pid, k) wins
        tw_ok = {pid: list(d.values()) for pid, d in tw_ok.items()}
        have = {p["pid"] for p in pairs if p["category"].startswith("U_")}
        extra = []
        for pid, ss in samples.items():
            p = prompts[pid]
            if not p["is_unanswerable"] or pid in have:
                continue
            exact = [s for s in ss if usable(s) and is_exact(s["text"], p["strict_phrase"])]
            grounded = [r for r in tw_ok.get(p["base_pid"], []) if not contains_phrase(r["text"], p["strict_phrase"])
                        and not lexical_phrase_only(r["text"])]
            if exact and grounded:
                extra.append((p, exact, grounded))
        need = max(0, phrase_rejected_n - sum(p["category"].startswith("U_") for p in pairs))
        extra.sort(key=lambda t: hashlib.sha256(t[0]["pid"].encode()).hexdigest())
        for p, exact, grounded in extra[:need]:
            r = min(grounded, key=lambda s: (s["k"] != -1, s["k"]))
            r = {**r, "judge": r["judge"]}
            add(p, "U_phrase_over_grounded", pick(exact, len(r["text"])), r)
            pairs[-1]["rejected_source"] = "policy_generic_twin"
        excluded["U_phrase_over_grounded_available"] = len(extra)

    phrase_chosen = sum(p["category"].startswith("U_") for p in pairs)
    phrase_rejected = [p for p in pairs if p["category"] == "A_false_refusal" and p["rejected_is_exact_phrase"]]
    if not args.no_balance and len(phrase_rejected) > phrase_chosen:
        drop = sorted(phrase_rejected, key=lambda p: hashlib.sha256(p["pid"].encode()).hexdigest())[phrase_chosen:]
        ids = {id(p) for p in drop}
        pairs = [p for p in pairs if id(p) not in ids]
        excluded["balance_A_false_refusal_bare_phrase"] = len(drop)

    out = d / "pairs"
    out.mkdir(exist_ok=True)
    for split in ("train", "val"):
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in pairs:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"model": args.model, "n_pairs": len(pairs), "excluded": dict(excluded),
              "category": dict(collections.Counter(r["category"] for r in pairs)),
              "split": dict(collections.Counter(r["split"] for r in pairs)),
              "phrase_chosen": sum(r["category"].startswith("U_") for r in pairs),
              "phrase_rejected": sum(r["category"] == "A_false_refusal" and r["rejected_is_exact_phrase"] for r in pairs),
              "rejected_greedy": sum(r["rejected_is_greedy"] for r in pairs),
              "unans_mode": dict(collections.Counter(r["unans_mode"] for r in pairs if r["unans_mode"]))}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("select", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--dir", required=True)
        s.add_argument("--out", required=True)
        if name == "select":
            s.add_argument("--max-a", type=int, default=800, help="answerable prompts judged, by pid hash")
    b = sub.add_parser("build")
    b.add_argument("--dir", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--no-balance", action="store_true")
    b.add_argument("--twin-dir", default=None, help="full run of the generic twins (prompts.jsonl, verified.jsonl)")
    args = ap.parse_args()
    {"select": cmd_select, "verify": cmd_verify, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
