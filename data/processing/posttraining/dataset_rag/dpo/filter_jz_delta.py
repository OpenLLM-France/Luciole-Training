#!/usr/bin/env python3
"""v2 of the Jean Zay delta pairs (decision 2026-10-07): dpo/data/jz_delta -> dpo/data/jz_delta_v2, jz_delta untouched.

U_bad_refusal pairs whose 4B rejected refusal was faulted by the Qwen3.8 judge for ONE defect only (a false statement
about the documents alone, or unsupported claims alone) are removed. Manual audit of 24 pairs
(jz_delta/audit_u_bad_refusal_manual.json): 8 of the 14 single-defect pairs were doubtful (pedantic verdict on "the
documents only mention X", or the chosen makes the same claim), against 1 of 10 for the other U_bad_refusal pairs.
The Mistral cross-judge was no help there (non-discriminant on refusals) and is NOT applied to the chosen either: the
chosen it rejects are mostly grounded partial refusals, kept on purpose.
Both files are filtered the same way (delta_pairs.jsonl, delta_pairs_datatrove.jsonl).
    venv/bin/python dpo/filter_jz_delta.py [--src DIR --out DIR --jz-run DIR]   (defaults: the 2026-10-06 run)
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent / "data"
SRC, OUT = D / "jz_delta", D / "jz_delta_v2"
SINGLE = ({"false_statement_about_documents"}, {"unsupported_claims"})


def rows(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def defects(j):
    return {k for k, bad in (("mixed", j["mixed"]), ("missing_fact_incorrect", not j["missing_fact_correct"]),
                             ("unsupported_claims", bool(j["unsupported_claims"])),
                             ("false_statement_about_documents", j["false_statement_about_documents"])) if bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--jz-run", default=str(D / "jz_run"))
    a = ap.parse_args()
    src, out, run = Path(a.src), Path(a.out), Path(a.jz_run)
    judge = {}
    for ds in ("hotpotqa", "tatqa"):
        for r in rows(run / "qwen35_4b" / ds / "judged_chosen_qwen35_4b.jsonl"):
            if r.get("judge"):
                judge[(r["pid"], r["k"])] = r["judge"]
    drop = set()
    for r in rows(src / "delta_pairs.jsonl"):
        if r["category"] == "U_bad_refusal" and defects(judge[(r["pid"], r["rejected_k"])]) in SINGLE:
            drop.add(r["pid"])
    out.mkdir(parents=True, exist_ok=True)
    man = {"description": "jz_delta minus the U_bad_refusal pairs with a single judge defect (filter_jz_delta.py)",
           "source_sha256": json.loads((src / "manifest.json").read_text())["sha256"], "removed": len(drop), "files": {}}
    for name, pid_of in (("delta_pairs.jsonl", lambda r: r["pid"]), ("delta_pairs_datatrove.jsonl", lambda r: r["id"])):
        kept = [r for r in rows(src / name) if pid_of(r) not in drop]
        with open(out / name, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        man["files"][name] = {"n": len(kept), "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest()}
    cat = lambda r: f"{r['dataset']}/{r['category']}"  # noqa: E731
    man["added"] = dict(sorted(collections.Counter(cat(r) for r in rows(out / "delta_pairs.jsonl")).items()))
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(man, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
