#!/usr/bin/env python3
"""Judging todo of the REJECTED side of the Jean Zay delta run (Qwen3.5-4B): every usable sample is judged, with
reasoning (compute is not the limit on JZ), in the files build_p10_delta.py reads for the 4B side:
  answer prompts, citation_ok      -> verified.jsonl                 (score_pilot_samples, reasoning)
  answer prompts, not citation_ok  -> scored_citfail.jsonl           (same judge: A_bad_citation only if content right)
  refusal prompts                  -> judged_chosen_<gen>.jsonl      (judge_refusal_candidates, all ks)
qwen_chosen_all's own todo only judges until a good sample is found (chosen logic): a rejected needs the faulty ones.
Writes D/todo_rej_verify.jsonl, D/todo_rej_citfail.jsonl ({pid, k}) and D/todo_rej_refusal.pids; prints the counts.
    python dpo/jz_todo_rejected.py --dir dpo/data/jz_run/qwen35_4b/hotpotqa --gen qwen35_4b

--chosen-gen qwen35_35b_a3b (judge budget of the endpoint, run hotpotqa_fr_delta: the Jean Zay rule asked ~34,000
reasoning verdicts): only what build_p10_delta.py can use. Prompts without an accepted chosen of that generator (dir
<D>/../../<chosen-gen>/<dataset>) are left out; citfail samples are only judged on prompts where no citation_ok sample
is faulty (A_wrong_answer, A_unsupported_reasoning and A_false_refusal come first) once all of them are judged, one at
a time in the order build_p10_delta.py picks the rejected (greedy, then closest length to the chosen): rerun after
each judging round until the list is empty. Same pairs, unless the accepted chosen changes between two rounds.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qwen_chosen_all as Q  # noqa: E402
from build_pilot_pairs import is_good  # noqa: E402


def faulty(j):
    """A verdict build_p10_delta.py turns into a rejected placed before A_bad_citation."""
    return j["refused"] or (not j["mixed"] and not is_good(j, False)
                            and (not j["answer_correct"] or (j["answer_supported"] and not j["reasoning_supported"])))


def needed_citfail(p, ss, verified, citfail, chosen_text, st):
    """[{pid, k}]: the next citfail sample to judge on this answer prompt, if any (see --chosen-gen)."""
    pid = p["pid"]
    ok = [s for s in ss.values() if Q.usable(s) and Q.citation_ok(p, s["text"])]
    if any((verified.get((pid, s["k"])) or {}).get("judge") is None for s in ok):
        st["citfail_waiting_verify"] += 1  # verify first: a faulty sample there makes the citfail ones useless
        return []
    if any(faulty(verified[(pid, s["k"])]["judge"]) for s in ok):
        st["citfail_not_needed"] += 1
        return []
    bad = sorted((s for s in ss.values() if Q.usable(s) and not Q.citation_ok(p, s["text"])),
                 key=lambda s: (s["k"] != -1, abs(len(s["text"]) - len(chosen_text))))
    for s in bad:
        c = (citfail.get((pid, s["k"])) or {}).get("judge")
        if c is None:
            return [{"pid": pid, "k": s["k"]}]
        if not c["refused"] and not c["mixed"] and c["answer_correct"] and c["answer_supported"] \
                and c["reasoning_supported"]:
            st["citfail_found"] += 1
            return []
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--chosen-gen", default=None, help="judge only what the pairs with this chosen generator use")
    a = ap.parse_args()
    d = Path(a.dir)
    chosen = None
    if a.chosen_gen:
        Q.GEN, Q.PREFER_QUOTED = a.chosen_gen, True  # as build_p10_delta.py
        cp, cs, csc, cve, crj = Q.state(d.parent.parent / a.chosen_gen / d.name)
        chosen = {}
        for pid, p in cp.items():
            acc = Q.accepted(p, cs.get(pid, {}), csc, cve, crj)
            if acc:
                chosen[pid] = acc[0]["text"]
    Q.GEN = a.gen
    prompts, samples, _, verified, refj = Q.state(d)
    citfail = Q.last_by_key(d / "scored_citfail.jsonl")
    ver, cit, ref = [], [], set()
    st = collections.Counter()
    for pid, p in prompts.items():
        ss = samples.get(pid)
        if not ss:
            st["prompts_without_samples"] += 1
            continue
        if chosen is not None and p["chosen_kind"] in ("answer", "refusal"):
            if pid not in chosen:
                st["prompts_without_chosen"] += 1
                continue
            if p["chosen_kind"] == "answer":
                cit += needed_citfail(p, ss, verified, citfail, chosen[pid], st)
                ss = {k: s for k, s in ss.items() if Q.citation_ok(p, s["text"])}  # the others: just above
        for k, s in ss.items():
            if not Q.usable(s):
                st["unusable_samples"] += 1
                continue
            if p["chosen_kind"] == "answer":
                if Q.citation_ok(p, s["text"]):
                    if not (verified.get((pid, k)) or {}).get("judge"):
                        ver.append({"pid": pid, "k": k})
                elif not (citfail.get((pid, k)) or {}).get("judge"):
                    cit.append({"pid": pid, "k": k})
            elif p["chosen_kind"] == "refusal":
                if not (refj.get((pid, k)) or {}).get("judge"):
                    ref.add(pid)
                    st["refusal_samples_todo"] += 1
    for name, rs in (("todo_rej_verify.jsonl", ver), ("todo_rej_citfail.jsonl", cit)):
        with open(d / name, "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    (d / "todo_rej_refusal.pids").write_text("".join(p + "\n" for p in sorted(ref)))
    print(json.dumps({**st, "verify": len(ver), "citfail": len(cit), "refusal_prompts": len(ref)}))


if __name__ == "__main__":
    main()
