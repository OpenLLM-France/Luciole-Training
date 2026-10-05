#!/usr/bin/env python3
"""Judging todo of the REJECTED side of the Jean Zay delta run (Qwen3.5-4B): every usable sample is judged, with
reasoning (compute is not the limit on JZ), in the files build_p10_delta.py reads for the 4B side:
  answer prompts, citation_ok      -> verified.jsonl                 (score_pilot_samples, reasoning)
  answer prompts, not citation_ok  -> scored_citfail.jsonl           (same judge: A_bad_citation only if content right)
  refusal prompts                  -> judged_chosen_<gen>.jsonl      (judge_refusal_candidates, all ks)
qwen_chosen_all's own todo only judges until a good sample is found (chosen logic): a rejected needs the faulty ones.
Writes D/todo_rej_verify.jsonl, D/todo_rej_citfail.jsonl ({pid, k}) and D/todo_rej_refusal.pids; prints the counts.
    python dpo/jz_todo_rejected.py --dir dpo/data/jz_run/qwen35_4b/hotpotqa --gen qwen35_4b
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qwen_chosen_all as Q  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--gen", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
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
