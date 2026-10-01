#!/usr/bin/env python3
"""Split the TatQA held-out run (one sampling over generic + strict prompts) into the layout of
the HotpotQA runs: a generic dir (run_two_pass.sh) and a strict dir (run_strict.sh --twin-dir).

  prompts  all prompts get pilot_split (90/10 by qid, seed 42: a strict variant shares the split
           of its base) and a judge-facing gold with the TatQA scale ("-1859 (scale: thousand)";
           gold_answer_raw keeps the original). Writes <all>/prompts_judge.jsonl and the generic
           prompts to <generic>/prompts.jsonl (context verdicts can start before sampling ends).
  merge-u  appends the generic unanswerable clones of <u>/prompts.jsonl (build_nogold_prompts_tatqa
           + answered clones, already with split and judge-facing gold) to <generic>/prompts.jsonl
           (idempotent: pids already there are skipped); the strict variants reach the strict dir
           through build_strict_prompts.py --pool <all>/prompts_judge.jsonl + <u>/prompts.jsonl.
  samples  after sampling (and build_strict_prompts.py for the strict dir): copies each dir's
           share of <all>/samples.jsonl and a sampling.manifest.json pointing to the parent one.

    python3 dpo/prep_tatqa_heldout_dirs.py prompts --all dpo/data/heldout_tatqa_sft4152 \
        --generic dpo/data/heldout_tatqa_generic_sft4152
    python3 dpo/prep_tatqa_heldout_dirs.py samples --all dpo/data/heldout_tatqa_sft4152 \
        --dirs dpo/data/heldout_tatqa_generic_sft4152 dpo/data/heldout_tatqa_strict_sft4152
"""
import argparse
import hashlib
import json
import random
from pathlib import Path


def rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompts(args):
    src = Path(args.all) / "prompts.jsonl"
    ps = list(rows(src))
    qids = sorted({p["qid"] for p in ps})
    random.Random(args.seed).shuffle(qids)
    val = set(qids[: round(len(qids) * args.val_frac)])
    for p in ps:
        p["pilot_split"] = "val" if p["qid"] in val else "train"
        raw = p.get("gold_answer_raw", p["gold_answer"])
        p["gold_answer_raw"] = raw
        p["gold_answer"] = f"{raw} (scale: {p['gold_scale']})" if p.get("gold_scale") else raw
    write(Path(args.all) / "prompts_judge.jsonl", ps)
    generic = [p for p in ps if not p.get("is_strict_variant")]
    out = Path(args.generic) / "prompts.jsonl"
    write(out, generic)
    man = {"pool": str(src), "pool_sha256": sha(src), "seed": args.seed, "val_frac": args.val_frac,
           "n": len(generic), "questions": {"train": len(qids) - len(val), "val": len(val)},
           "prompt_sha256": sha(out)}
    (Path(args.generic) / "prompts.manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps(man, indent=2))


def merge_u(args):
    g = Path(args.generic) / "prompts.jsonl"
    have = {p["pid"] for p in rows(g)}
    new = [p for p in rows(Path(args.u) / "prompts.jsonl") if not p.get("is_strict_variant") and p["pid"] not in have]
    with g.open("a", encoding="utf-8") as f:
        for p in new:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    mpath = Path(args.generic) / "prompts.manifest.json"
    man = json.loads(mpath.read_text())
    man.setdefault("merged", []).append({"from": str(Path(args.u) / "prompts.jsonl"), "added": len(new),
                                          "prompt_sha256_after": sha(g)})
    mpath.write_text(json.dumps(man, indent=2) + "\n")
    print(f"{g}: +{len(new)} prompts")


def samples(args):
    parents, last = {}, {}
    for a in map(Path, args.all):
        parents[str(a)] = json.loads((a / "sampling.manifest.json").read_text())
        for r in rows(a / "samples.jsonl"):
            last[r["pid"]] = r
    for d in map(Path, args.dirs):
        pids = [p["pid"] for p in rows(d / "prompts.jsonl")]
        missing = [pid for pid in pids if pid not in last]
        if missing:
            raise SystemExit(f"{d}: {len(missing)} prompts without samples")
        write(d / "samples.jsonl", (last[pid] for pid in pids))
        man = {**next(iter(parents.values())), "sampling_runs": parents, "subset_of": list(parents),
               "subset_prompt_sha256": sha(d / "prompts.jsonl"), "n_prompts": len(pids)}
        (d / "sampling.manifest.json").write_text(json.dumps(man, indent=2) + "\n")
        print(f"{d}: {len(pids)} prompts")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--all", required=True)
    p.add_argument("--generic", required=True)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    m = sub.add_parser("merge-u")
    m.add_argument("--u", required=True)
    m.add_argument("--generic", required=True)
    s = sub.add_parser("samples")
    s.add_argument("--all", nargs="+", required=True)
    s.add_argument("--dirs", nargs="+", required=True)
    args = ap.parse_args()
    {"prompts": prompts, "merge-u": merge_u, "samples": samples}[args.stage](args)


if __name__ == "__main__":
    main()
