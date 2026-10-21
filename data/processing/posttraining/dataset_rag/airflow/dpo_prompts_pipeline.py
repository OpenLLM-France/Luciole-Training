#!/usr/bin/env python3
"""Airflow CLI adapter for the prompt preparation of the DPO runs (no Airflow dependency).

Two sources (param `source`):

heldout_pools  the held-out pools of the SFT newrag step 4152 policy, as for the Jean Zay run of 2026-10-06:
  check     inputs present (held-out pools, evaluation exclusions, HotpotQA cache, TAT-QA run files, calibration)
  prompts   build_jz_prompts.py: usable held-out prompts (context verdict agrees, no validation / paired-eval question)
  unans     build_unans_dag.py: unanswerable versions with the Luciole RAG pipeline code (one-for-one replacement of the
            removed gold chunks; Mistral for the missing TAT-QA [CONTEXT] summaries)
  contexts  judge_pilot_contexts.py on those versions (Qwen3.8-27B with reasoning); failed calls asked once more
  jz_run    build_jz_run.py: final prompt set (nogold clones dropped, DAG versions judged unanswerable added), calibration
sft_run        a run of an SFT DAG of dataset_rag (HotpotQA family, e.g. dataset_rag_hotpotqa_fr_abstracts/full_ref):
               the questions of which no version reached the SFT data
  check     the evaluated and filtered files of the run, calibration
  prompts   build_dpo_prompts_heldout.py extract (every held-out question: answerable rows with factual judge 5,
            unanswerable versions the generator refused, with their answerable twin) then format (formatting_sft, the
            SFT prompt with the citation format of the run, generic refusal rule only, without its invitation clause)
  unans     nothing to do: the unanswerable versions of the run are built by the SFT pipeline itself
  contexts  judge_pilot_contexts.py on every prompt
  jz_run    the prompts whose context verdict agrees with their label, in the jz_run layout, calibration
Output airflow_runs/dataset_rag_dpo_prompts/<run_name>/jz_run = input_run of the DAG dataset_rag_dpo_pairs.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATASETS = ("hotpotqa", "tatqa")
GENS = ("qwen35_35b_a3b", "qwen35_4b")  # jz_run layout: one directory per generator, same prompts
D = BASE / "dpo" / "data"
INPUTS = [D / "heldout_full_sft4152" / "prompts.jsonl", D / "heldout_full_sft4152" / "context_verdicts.jsonl",
          D / "heldout_tatqa_generic_sft4152" / "prompts.jsonl", D / "heldout_tatqa_generic_sft4152" / "context_verdicts.jsonl",
          D / "heldout_tatqa_u_sft4152" / "prompts.jsonl", D / "rag_dpo_onpolicy_8802410" / "val.jsonl",
          D / "p9_paired_eval" / "prompts.jsonl", BASE / "hotpotqa_cache_train"]
# Invitation clause of the generic refusal rule (formatting_sft._generic_refusal_instruction), absent from the
# published prompts (release_text.published_prompt, English only): removed in both languages.
INVITE_CLAUSE = {"en": " and invite the user to rephrase their query or add relevant documents",
                 "fr": " et invitez l'utilisateur à reformuler sa requête ou à ajouter des documents pertinents"}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, records):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else BASE / p


def dotenv(name):
    if os.environ.get(name):
        return os.environ[name]
    f = BASE / ".env"
    for line in (f.read_text(encoding="utf-8").splitlines() if f.is_file() else []):
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip('"').strip("'")
    return None


def run(script, *args, env=None):
    subprocess.run([sys.executable, str(BASE / "dpo" / script), *map(str, args)], cwd=BASE, env=env, check=True)


def env_for(out):
    return {**os.environ, "JZ_PROMPTS_OUT": str(out / "jz_gen_prompts"), "UNANS_DAG_OUT": str(out / "unans_dag"),
            "JZ_RUN_OUT": str(out / "jz_run")}


def sft_files(c):
    """(evaluated, filtered) files of the SFT run: <prefix>_augmented_evaluated.jsonl and its _filtered."""
    d = resolve(c["sft_run"])
    found = sorted(d.glob("*_augmented_evaluated.jsonl"))
    if len(found) != 1:
        raise FileNotFoundError(f"sft_run : un seul *_augmented_evaluated.jsonl attendu dans {d}, trouvé {len(found)}")
    return found[0], found[0].with_name(found[0].stem + "_filtered.jsonl")


def check(c):
    calib = [str(resolve(c["calib_from"]) / "calib" / ds) for ds in DATASETS
             if not (resolve(c["calib_from"]) / "calib" / ds / "reference_verified.jsonl").exists()] \
        if c["calib_from"] else []
    if c["source"] == "sft_run":
        evaluated, filtered = sft_files(c)
        missing = [str(p) for p in (filtered,) if not p.exists()] + calib
    else:
        missing = [str(p) for p in INPUTS + [resolve(c["tatqa_converted"]), resolve(c["tatqa_ctx_cache"])]
                   if not p.exists()] + calib
    if missing:
        raise FileNotFoundError("entrées absentes (à copier sous la racine du dépôt) : " + ", ".join(missing))


def sft_prompts(c, out):
    """Held-out prompts of an SFT run -> out/sft_pool/prompts.jsonl (records of build_dpo_prompts_heldout format, plus
    the fields of the jz_run prompts)."""
    evaluated, filtered = sft_files(c)
    d = out / "sft_pool"
    d.mkdir(parents=True, exist_ok=True)
    # every held-out question (no quota, natural distribution): answerable rows with factual judge 5, and the
    # unanswerable versions the generator refused (an answered one is often answerable in disguise)
    run("build_dpo_prompts_heldout.py", "extract", "--evaluated", evaluated, "--filtered", filtered,
        "--out", d / "rows.jsonl", "--n-unanswerable", 10 ** 9, "--n-extra-answerable", 10 ** 9,
        "--comparison-frac", -1, "--seed", c["seed"])
    run("build_dpo_prompts_heldout.py", "format", "--rows", d / "rows.jsonl", "--out", d / "formatted.jsonl",
        "--language", c["sft_language"], "--strict-prob", 0, "--seed", c["seed"])
    clause = INVITE_CLAUSE[c["sft_language"]]
    out_rows, st = [], collections.Counter()
    for p in rows(d / "formatted.jsonl"):
        if p.get("citation_style") is None:
            raise ValueError(f"{p['pid']}: règle de citation non reconnue dans le prompt")
        system = p["messages"][0]["content"]
        assert system.count(clause) == 1, (p["pid"], "règle de refus générique attendue")
        messages = [{**p["messages"][0], "content": system.replace(clause, "")}, p["messages"][1]]
        out_rows.append({**p, "messages": messages, "dataset": c["sft_dataset"], "language": c["sft_language"],
                         "sft_run": c["sft_run"], "pilot_split": "train",
                         # the run's question ids are not HotpotQA ids: overlap with the English SFT not checkable
                         "question_seen_in_policy_sft": None,
                         "in_8802410_validation": False, "in_paired_eval": False})
        st["U" if p["is_unanswerable"] else "A"] += 1
    write(d / "prompts.jsonl", out_rows)
    print(json.dumps({"sft_run": c["sft_run"], "prompts": len(out_rows), **st}), flush=True)


def judge_contexts(c, prompts, verdicts):
    """judge_pilot_contexts.py; a failed call ("ambiguous" + error, never asked again by the script) is dropped and
    asked once more, then tolerated up to max_missing_share."""
    key = resolve(c["judge_key_file"])
    if not key.is_file() or not key.read_text().strip():
        raise ValueError("judge_key_file doit contenir la clé du juge (ou 'none' pour vLLM local)")
    for attempt in (1, 2):
        if attempt == 2 and verdicts.exists():
            write(verdicts, [r for r in rows(verdicts) if not r.get("error")])
        run("judge_pilot_contexts.py", "--prompts", prompts, "--out", verdicts,
            "--judge-url", c["judge_url"], "--judge-model", c["judge_model"], "--judge-key-file", key,
            "--max-tokens", c["judge_max_tokens"], "--timeout", 1800, "--concurrency", c["judge_concurrency"])
    errors = sum(1 for r in rows(verdicts) if r.get("error"))
    if errors > c["max_missing_share"] * len(rows(prompts)):
        raise RuntimeError(f"{prompts.parent.name}: {errors} verdicts de contexte en échec ; relancer")


def contexts(c, out):
    if c["source"] == "sft_run":
        judge_contexts(c, out / "sft_pool" / "prompts.jsonl", out / "sft_pool" / "context_verdicts.jsonl")
        return
    for ds in DATASETS:
        d = out / "unans_dag" / ds
        judge_contexts(c, d / "prompts.jsonl", d / "context_verdicts.jsonl")


def sft_jz_run(c, out):
    """jz_run layout from the judged SFT pool: prompts whose context verdict agrees with their label (an unanswerable
    version also needs its answerable twin), calibration copied from calib_from when set."""
    d, ds = out / "sft_pool", c["sft_dataset"]
    cv = {r["pid"]: r for r in rows(d / "context_verdicts.jsonl")}
    prompts = rows(d / "prompts.jsonl")
    st = collections.Counter()
    ok = [p for p in prompts
          if cv.get(p["pid"], {}).get("verdict") == ("unanswerable" if p["is_unanswerable"] else "answerable")]
    ok_pids = {p["pid"] for p in ok}
    st.update(f"dropped_verdict_{cv.get(p['pid'], {}).get('verdict', 'missing')}/{'U' if p['is_unanswerable'] else 'A'}"
              for p in prompts if p["pid"] not in ok_pids)
    answerable_q = {p["qid"] for p in ok if not p["is_unanswerable"]}
    kept = [{**p, "context_verdict_ok": True} for p in ok if not p["is_unanswerable"] or p["qid"] in answerable_q]
    st["dropped_twin_absent"] = len(ok) - len(kept)
    run_dir = out / "jz_run"
    for gen in GENS:
        g = run_dir / gen / ds
        g.mkdir(parents=True, exist_ok=True)
        write(g / "prompts.jsonl", kept)
        write(g / "context_verdicts.jsonl", [cv[p["pid"]] for p in kept])
    if c["calib_from"]:
        shutil.copytree(resolve(c["calib_from"]) / "calib", run_dir / "calib", dirs_exist_ok=True)
    qs = collections.defaultdict(set)
    for p in kept:
        qs[p["qid"]].add(p["is_unanswerable"])
    man = {"source": {"sft_run": c["sft_run"]}, "calibration": {"copied_from": c["calib_from"]},
           "datasets": {ds: {**dict(sorted(st.items())), "prompts": len(kept),
                             "answerable": sum(not p["is_unanswerable"] for p in kept),
                             "unanswerable": sum(p["is_unanswerable"] for p in kept),
                             "questions": len(qs), "questions_with_both_versions": sum(len(v) == 2 for v in qs.values()),
                             "citation_style": dict(collections.Counter(f"{p['citation_style']}/{p['citation_placement']}"
                                                                        for p in kept)),
                             "sha256": hashlib.sha256((run_dir / GENS[0] / ds / "prompts.jsonl").read_bytes()).hexdigest()}}}
    (run_dir / "manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(man, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("check", "prompts", "unans", "contexts", "jz_run"))
    a = ap.parse_args()
    c = json.loads(os.environ["DPO_RUN_CONFIG"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", c["run_name"]):
        raise ValueError("run_name : 1 à 80 lettres/chiffres/points/tirets/underscores")
    for name, lo, hi in (("mistral_concurrency", 1, 12), ("judge_concurrency", 1, 512), ("judge_max_tokens", 1000, 32000)):
        if isinstance(c[name], bool) or not isinstance(c[name], int) or not lo <= c[name] <= hi:
            raise ValueError(f"{name} : entier entre {lo} et {hi}"
                             + (" (plafond Mistral : 12 appels au total, tous jobs confondus)" if name.startswith("mistral") else ""))
    if c["source"] not in ("heldout_pools", "sft_run"):
        raise ValueError("source : heldout_pools ou sft_run")
    if c["source"] == "sft_run":
        if not c.get("sft_run"):
            raise ValueError("sft_run : dossier d'un run d'un DAG SFT (airflow_runs/<dag_id>/<run_name>)")
        if c["sft_language"] not in INVITE_CLAUSE or not re.fullmatch(r"[a-z0-9_]+", str(c["sft_dataset"])):
            raise ValueError("sft_language : en ou fr ; sft_dataset : minuscules, chiffres, _")
        if c["sft_dataset"] in DATASETS:
            raise ValueError(f"sft_dataset : nom distinct de {DATASETS} (pools held-out de la politique actuelle)")
        if c.get("pairs"):
            raise ValueError("source sft_run : dataset_rag_dpo_pairs ne prend pas encore ces prompts (pairs = null)")
    out = BASE / "airflow_runs/dataset_rag_dpo_prompts" / c["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "config.json"
    if lock.exists() and json.loads(lock.read_text()) != c:
        raise ValueError("Configuration différente : choisir un nouveau run_name")
    if not lock.exists():
        lock.write_text(json.dumps(c, indent=2), encoding="utf-8")
    if a.stage == "check":
        check(c)
    elif a.stage == "prompts":
        sft_prompts(c, out) if c["source"] == "sft_run" else run("build_jz_prompts.py", env=env_for(out))
    elif a.stage == "unans":
        if c["source"] == "sft_run":
            print("source sft_run : versions sans réponse déjà construites par le DAG SFT")
            return
        key = dotenv("LLM_API_KEY")
        run("build_unans_dag.py", "--tatqa-converted", resolve(c["tatqa_converted"]),
            "--tatqa-ctx-cache", resolve(c["tatqa_ctx_cache"]), "--mistral-concurrency", c["mistral_concurrency"],
            env={**env_for(out), **({"LLM_API_KEY": key} if key else {})})
    elif a.stage == "contexts":
        contexts(c, out)
    else:
        if c["source"] == "sft_run":
            sft_jz_run(c, out)
        else:
            run("build_jz_run.py", env={**env_for(out), **({"JZ_CALIB_FROM": str(resolve(c["calib_from"]))}
                                                     if c["calib_from"] else {})})
        print(f"input_run pour dataset_rag_dpo_pairs : {(out / 'jz_run').relative_to(BASE)}")


if __name__ == "__main__":
    main()
