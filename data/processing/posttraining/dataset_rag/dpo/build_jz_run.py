#!/usr/bin/env python3
"""Final prompt set of the Jean Zay delta run (Qwen3.5-35B-A3B chosen vs Qwen3.5-4B rejected, both judged by
Qwen3.8-27B on JZ), from dpo/data/jz_gen_prompts (build_jz_prompts.py) and dpo/data/unans_dag (build_unans_dag.py).

  - TAT-QA: the unanswerable prompts that are `nogold_clone` (gold chunks dropped WITHOUT replacement, always 1-2 chunks
    shorter than their twin: "fewer chunks => refuse" shortcut) are left out;
  - both datasets: the DAG unanswerable versions (one-for-one replacement) whose context verdict is "unanswerable" are
    added, if their answerable twin (source_pid) is in the set and they are not in the 8802410 validation / paired
    evaluation (same rules as build_jz_prompts.py).
Layout read by sample_policy.py / qwen_chosen_all.py (the judge state files carry no generator name, hence one
directory per generator, same prompts):
  dpo/data/jz_run/<gen>/<dataset>/prompts.jsonl, context_verdicts.jsonl   gen = qwen35_35b_a3b, qwen35_4b
  dpo/data/jz_run/calib/<dataset>/   200 completions already judged (with reasoning) on the endpoint / DGX, to rejudge on
                                     JZ before the full run (calib_compare.py)
  dpo/data/jz_run/manifest.json
    .venv/bin/python dpo/build_jz_run.py [hotpotqa] [tatqa]     (default: both)
"""
import collections
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE / "data"
# env overrides (Airflow run dirs): JZ_PROMPTS_OUT, UNANS_DAG_OUT, JZ_RUN_OUT; JZ_CALIB_FROM = a jz_run whose calib/ is
# copied as is (the calibration sources, dpo/data/qwen35_chosen_onpolicy, only exist on the DGX)
SRC = Path(os.environ.get("JZ_PROMPTS_OUT", D / "jz_gen_prompts"))
DAG = Path(os.environ.get("UNANS_DAG_OUT", D / "unans_dag"))
OUT = Path(os.environ.get("JZ_RUN_OUT", D / "jz_run"))
CALIB_FROM = os.environ.get("JZ_CALIB_FROM")
GENS = ("qwen35_35b_a3b", "qwen35_4b")
# completions of the same Qwen3.5-35B-A3B already judged with reasoning (endpoint + DGX local judge, calibrated)
CALIB_SRC = {"hotpotqa": D / "qwen35_chosen_onpolicy" / "hotpotqa", "tatqa": D / "qwen35_chosen_onpolicy" / "tatqa"}
CALIB_N = 100


def rows(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write(p, rs):
    with open(p, "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def table(qid):
    return qid.split("_", 1)[0]


def main():
    val_q = {r["qid"] for r in rows(D / "rag_dpo_onpolicy_8802410" / "val.jsonl")}
    ev_q = {p["qid"] for p in rows(D / "p9_paired_eval" / "prompts.jsonl")}
    bad_tables = {table(q) for q in val_q | ev_q if "_" in q}

    def excluded(ds, p):
        q = p["qid"]
        if ds == "hotpotqa":
            return q in val_q or p.get("pilot_split") == "val" or q in ev_q
        return table(q) in bad_tables or int(hashlib.sha256(f"42:{table(q)}".encode()).hexdigest(), 16) % 100 < 10

    man = {"sources": {"jz_gen_prompts": str(SRC), "unans_dag": str(DAG)},
           "datasets": {}}
    for ds in [a for a in sys.argv[1:] if not a.startswith("-")] or ("hotpotqa", "tatqa"):
        st = collections.Counter()
        base = list(rows(SRC / f"prompts_{ds}.jsonl"))
        cv = {r["pid"]: r for r in rows(SRC / f"context_verdicts_{ds}.jsonl")}
        kept = []
        for p in base:
            if p["is_unanswerable"] and p.get("heldout_drop_reason") == "nogold_clone":
                st["dropped_nogold_clone"] += 1
                continue
            kept.append(p)
        answerable = {p["pid"] for p in kept if not p["is_unanswerable"]}
        dag_v = {r["pid"]: r for r in rows(DAG / ds / "context_verdicts.jsonl")}
        todo = DAG / ds / "todo_context.jsonl"
        if len(dag_v) < sum(1 for _ in rows(todo if todo.exists() else DAG / ds / "prompts.jsonl")):
            sys.exit(f"!! {ds}: context judge of unans_dag not finished ({len(dag_v)} verdicts)")
        for p in rows(DAG / ds / "prompts.jsonl"):
            v = dag_v[p["pid"]]["verdict"]
            if v != "unanswerable":
                st[f"dag_dropped_verdict_{v}"] += 1
            elif p["source_pid"] not in answerable:
                st["dag_dropped_twin_absent"] += 1
            elif excluded(ds, p):
                st["dag_dropped_val_or_paired_eval"] += 1
            else:
                kept.append({**p, "dataset": ds, "context_verdict_ok": True, "in_8802410_validation": False,
                             "in_paired_eval": False})
                cv[p["pid"]] = dag_v[p["pid"]]
                st["dag_added/" + p["unans_mode"]] += 1
        pids = [p["pid"] for p in kept]
        assert len(pids) == len(set(pids)), f"{ds}: duplicate pid"
        assert all(p["pid"] in cv for p in kept), f"{ds}: prompt without context verdict"
        for gen in GENS:
            d = OUT / gen / ds
            d.mkdir(parents=True, exist_ok=True)
            write(d / "prompts.jsonl", kept)
            write(d / "context_verdicts.jsonl", [cv[p["pid"]] for p in kept])
        qs = collections.defaultdict(set)
        for p in kept:
            qs[p["qid"]].add(p["is_unanswerable"])
        st.update({"prompts": len(kept), "answerable": len(answerable),
                   "unanswerable": sum(p["is_unanswerable"] for p in kept),
                   "unans_dag_clones": sum(p.get("clone_source") == "dag" for p in kept),
                   "questions": len(qs), "questions_with_both_versions": sum(len(v) == 2 for v in qs.values())})
        st.update({"unans_mode/" + (p.get("unans_mode") or "answerable"): 0 for p in kept})
        for p in kept:
            st["unans_mode/" + (p.get("unans_mode") or "answerable")] += 1
        man["datasets"][ds] = {**dict(sorted(st.items())),
                               "sha256": hashlib.sha256((OUT / GENS[0] / ds / "prompts.jsonl").read_bytes()).hexdigest()}

        if CALIB_FROM:
            shutil.copytree(Path(CALIB_FROM) / "calib" / ds, OUT / "calib" / ds, dirs_exist_ok=True)
            man["calibration"] = {**man.get("calibration", {}), ds: {"copied_from": str(CALIB_FROM)}}
            continue
        if not (CALIB_SRC[ds] / "verified.jsonl").exists():  # no calibration source here: calibration is optional
            continue
        # calibration: CALIB_N completions judged with reasoning outside JZ (answer prompts), stratified on the verdict
        c = CALIB_SRC[ds]
        ver = [r for r in rows(c / "verified.jsonl") if r.get("judge")]
        rng = random.Random(42)
        good = [r for r in ver if r["judge"].get("answer_correct")]
        bad = [r for r in ver if not r["judge"].get("answer_correct")]
        pick = rng.sample(good, min(len(good), CALIB_N // 2)) + rng.sample(bad, min(len(bad), CALIB_N - CALIB_N // 2))
        want = {r["pid"] for r in pick}
        cd = OUT / "calib" / ds
        cd.mkdir(parents=True, exist_ok=True)
        write(cd / "prompts.jsonl", [p for p in rows(c / "prompts.jsonl") if p["pid"] in want])
        write(cd / "context_verdicts.jsonl", [r for r in rows(c / "context_verdicts.jsonl") if r["pid"] in want])
        write(cd / "chosen_qwen35_35b_a3b.jsonl", [r for r in rows(c / "chosen_qwen35_35b_a3b.jsonl") if r["pid"] in want])
        write(cd / "reference_verified.jsonl", pick)
        write(cd / "only.jsonl", [{"pid": r["pid"], "k": r["k"]} for r in pick])
        man["calibration"] = {**man.get("calibration", {}), ds: {"n": len(pick), "answer_correct_true": min(len(good), CALIB_N // 2)}}
    if (SRC / "README.md").exists():  # written by hand for the 2026-10-06 run
        shutil.copy(SRC / "README.md", OUT / "README_jz_gen_prompts.md")
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
