#!/usr/bin/env python3
"""Airflow CLI adapter for the prompt preparation of the DPO runs (no Airflow dependency).

Stages, as for the Jean Zay run of 2026-10-06:
  check     inputs present (held-out pools, evaluation exclusions, HotpotQA cache, TAT-QA run files, calibration)
  prompts   build_jz_prompts.py: usable held-out prompts (context verdict agrees, no validation / paired-eval question)
  unans     build_unans_dag.py: unanswerable versions with the Luciole RAG pipeline code (one-for-one replacement of the
            removed gold chunks; Mistral for the missing TAT-QA [CONTEXT] summaries)
  contexts  judge_pilot_contexts.py on those versions (Qwen3.8-27B with reasoning); failed calls asked once more
  jz_run    build_jz_run.py: final prompt set (nogold clones dropped, DAG versions judged unanswerable added), calibration
Output airflow_runs/dataset_rag_dpo_prompts/<run_name>/jz_run = input_run of the DAG dataset_rag_dpo_pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATASETS = ("hotpotqa", "tatqa")
D = BASE / "dpo" / "data"
INPUTS = [D / "heldout_full_sft4152" / "prompts.jsonl", D / "heldout_full_sft4152" / "context_verdicts.jsonl",
          D / "heldout_tatqa_generic_sft4152" / "prompts.jsonl", D / "heldout_tatqa_generic_sft4152" / "context_verdicts.jsonl",
          D / "heldout_tatqa_u_sft4152" / "prompts.jsonl", D / "rag_dpo_onpolicy_8802410" / "val.jsonl",
          D / "p9_paired_eval" / "prompts.jsonl", BASE / "hotpotqa_cache_train"]


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


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


def check(c):
    missing = [str(p) for p in INPUTS + [resolve(c["tatqa_converted"]), resolve(c["tatqa_ctx_cache"])] if not p.exists()]
    if c["calib_from"]:
        missing += [str(resolve(c["calib_from"]) / "calib" / ds) for ds in DATASETS
                    if not (resolve(c["calib_from"]) / "calib" / ds / "reference_verified.jsonl").exists()]
    if missing:
        raise FileNotFoundError("entrées absentes (à copier sous la racine du dépôt) : " + ", ".join(missing))


def contexts(c, out):
    key = resolve(c["judge_key_file"])
    if not key.is_file() or not key.read_text().strip():
        raise ValueError("judge_key_file doit contenir la clé du juge (ou 'none' pour vLLM local)")
    for ds in DATASETS:
        d = out / "unans_dag" / ds
        verdicts = d / "context_verdicts.jsonl"
        for attempt in (1, 2):
            if attempt == 2 and verdicts.exists():
                # judge_pilot_contexts writes a failed call as "ambiguous" + error and never asks again: drop those rows
                kept = [r for r in rows(verdicts) if not r.get("error")]
                verdicts.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
            run("judge_pilot_contexts.py", "--prompts", d / "prompts.jsonl", "--out", verdicts,
                "--judge-url", c["judge_url"], "--judge-model", c["judge_model"], "--judge-key-file", key,
                "--max-tokens", c["judge_max_tokens"], "--timeout", 1800, "--concurrency", c["judge_concurrency"])
        errors = sum(1 for r in rows(verdicts) if r.get("error"))
        if errors > c["max_missing_share"] * len(rows(d / "prompts.jsonl")):
            raise RuntimeError(f"{ds}: {errors} verdicts de contexte en échec ; relancer")


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
        run("build_jz_prompts.py", env=env_for(out))
    elif a.stage == "unans":
        key = dotenv("LLM_API_KEY")
        run("build_unans_dag.py", "--tatqa-converted", resolve(c["tatqa_converted"]),
            "--tatqa-ctx-cache", resolve(c["tatqa_ctx_cache"]), "--mistral-concurrency", c["mistral_concurrency"],
            env={**env_for(out), **({"LLM_API_KEY": key} if key else {})})
    elif a.stage == "contexts":
        contexts(c, out)
    else:
        run("build_jz_run.py", env={**env_for(out), **({"JZ_CALIB_FROM": str(resolve(c["calib_from"]))}
                                                     if c["calib_from"] else {})})
        print(f"input_run pour dataset_rag_dpo_pairs : {(out / 'jz_run').relative_to(BASE)}")


if __name__ == "__main__":
    main()
