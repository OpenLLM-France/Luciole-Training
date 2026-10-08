#!/usr/bin/env python3
"""Airflow CLI adapter for the existing Delta DPO scripts (no Airflow dependency).

Stages: prepare -> chosen / rejected -> judge -> build -> filter -> crossjudge -> release -> audit, i.e. the steps of
the Jean Zay run of 2026-10-06/07 up to the parquet configs of the published release (no upload)."""
from __future__ import annotations

import argparse
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
GENERATORS = ("qwen35_35b_a3b", "qwen35_4b")


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, records):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else BASE / p


def run(script, *args, env=None, capture=False):
    r = subprocess.run([sys.executable, str(BASE / "dpo" / script), *map(str, args)],
                       cwd=BASE, env=env, check=True, capture_output=capture, text=True)
    return r.stdout if capture else None


def dotenv(name):
    """Value of name from the environment, else from the .env of dataset_rag (LLM_API_KEY lives there)."""
    if os.environ.get(name):
        return os.environ[name]
    f = BASE / ".env"
    for line in (f.read_text(encoding="utf-8").splitlines() if f.is_file() else []):
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip('"').strip("'")
    return None


def served_models(url):
    """/v1/models of an OpenAI-compatible server (vLLM: served name, root path, max_model_len), or the error."""
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=30) as r:
            return json.load(r).get("data")
    except Exception as e:  # recorded, not fatal: some endpoints restrict /models
        return {"error": repr(e)[:300]}


def prepare(c, out):
    source = resolve(c["input_run"])
    manifest = {"config": c, "inputs": {}}
    for ds in DATASETS:
        p = rows(source / GENERATORS[0] / ds / "prompts.jsonl")
        other = rows(source / GENERATORS[1] / ds / "prompts.jsonl")
        if p != other or not p or len({r["pid"] for r in p}) != len(p):
            raise ValueError(f"{ds}: prompts vides, dupliqués ou différents entre générateurs")
        cv = {r["pid"]: r for r in rows(source / GENERATORS[0] / ds / "context_verdicts.jsonl")}
        for r in p:
            expected = "unanswerable" if r["is_unanswerable"] else "answerable"
            if cv.get(r["pid"], {}).get("verdict") != expected:
                raise ValueError(f"{ds}/{r['pid']}: verdict de contexte absent ou contradictoire")
            if r.get("in_8802410_validation") or r.get("in_paired_eval") \
                    or (ds == "hotpotqa" and r.get("pilot_split") == "val"):
                raise ValueError(f"{ds}/{r['pid']}: fuite de validation/évaluation")
            if r["is_unanswerable"] and r.get("heldout_drop_reason") == "nogold_clone":
                raise ValueError(f"{ds}/{r['pid']}: ancien clone sans rembourrage")
        # A question's answerable and unanswerable versions stay together in a smoke run.
        if c.get("limit_questions"):
            qids = sorted({r["qid"] for r in p}, key=lambda q: hashlib.sha256(
                f"{c['seed']}:{q}".encode()).hexdigest())[:int(c["limit_questions"])]
            p = [r for r in p if r["qid"] in set(qids)]
        for gen in GENERATORS:
            d = out / gen / ds
            d.mkdir(parents=True, exist_ok=True)
            write(d / "prompts.jsonl", p)
            write(d / "context_verdicts.jsonl", [cv[r["pid"]] for r in p])
        manifest["inputs"][ds] = {"prompts": len(p), "sha256": {
            name: hashlib.sha256((out / GENERATORS[0] / ds / name).read_bytes()).hexdigest()
            for name in ("prompts.jsonl", "context_verdicts.jsonl")}}
    if c["calibrate"]:
        for ds in DATASETS:
            d = out / "calib" / ds
            d.mkdir(parents=True, exist_ok=True)
            for name in ("prompts.jsonl", "context_verdicts.jsonl", "chosen_qwen35_35b_a3b.jsonl",
                         "reference_verified.jsonl", "only.jsonl"):
                shutil.copyfile(source / "calib" / ds / name, d / name)
                manifest["inputs"][f"calib/{ds}/{name}"] = hashlib.sha256((d / name).read_bytes()).hexdigest()
    (out / "inputs.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def generate(c, out, chosen):
    gen = GENERATORS[0 if chosen else 1]
    prefix = "chosen" if chosen else "rejected"
    for ds in DATASETS:
        d = out / gen / ds
        samples = d / f"chosen_{gen}.jsonl"
        run("sample_policy.py", "--prompts", d / "prompts.jsonl", "--out", samples,
            "--api-url", c[f"{prefix}_url"], "--model", c[f"{prefix}_model"],
            "--k", 2 if chosen else 4, "--temperature", 0.7 if chosen else 1.0,
            "--top-p", 0.80, "--top-k", 20, "--min-p", 0,
            "--max-tokens", 2048, "--seed", c["seed"],
            "--concurrency", c["generation_concurrency"],
            "--chat-template-kwargs", '{"enable_thinking": false}')
        last = {r["pid"]: r for r in rows(samples)}
        expected = {-1, *range(2 if chosen else 4)}
        for p in rows(d / "prompts.jsonl"):
            r = last.get(p["pid"], {})
            if r.get("errors") or {s["k"] for s in r.get("samples", [])} != expected:
                raise RuntimeError(f"{ds}/{p['pid']}: génération incomplète ; relancer")
            if any("<think>" in s["text"] or "</think>" in s["text"] for s in r["samples"]):
                raise RuntimeError("Le serveur a activé le raisonnement malgré enable_thinking=false")
        # what the JZ job recorded in generation_<gen>.manifest.json (weights, precision, server)
        (d / f"generation_{gen}.manifest.json").write_text(json.dumps({
            "served_model": c[f"{prefix}_model"], "url": c[f"{prefix}_url"], "dtype": c[f"{prefix}_dtype"],
            "server_models": served_models(c[f"{prefix}_url"]),
            "prompt_sha256": hashlib.sha256((d / "prompts.jsonl").read_bytes()).hexdigest(),
            "samples_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
            "k_sampled": 2 if chosen else 4, "temperature": 0.7 if chosen else 1.0, "top_p": 0.80, "top_k": 20,
            "min_p": 0.0, "seed": c["seed"], "greedy": True, "max_tokens": 2048, "enable_thinking": False,
            "concurrency": c["generation_concurrency"]}, indent=2), encoding="utf-8")


def judge(c, out):
    key = resolve(c["judge_key_file"])
    if not key.is_file() or not key.read_text().strip():
        raise ValueError("judge_key_file doit contenir la clé du juge (ou 'none' pour vLLM local)")
    # each side (35B / 4B) gets 3/4 of judge_concurrency (judge_chain.sh default): a side left alone keeps the server
    # nearly full (JZ run: at 1/2 each, the 4B judged alone on half the server for 2.5 h)
    env = {**os.environ, "RAG_ROOT": str(BASE), "R": str(out), "URL": c["judge_url"], "JUDGE_MODEL": c["judge_model"],
           "KEY": str(key), "DPO_PYTHON": sys.executable, "LOG_DIR": str(out / "logs"),
           "CONC": str(c["judge_concurrency"]),
           "JUDGE_MAX_TOKENS": str(c["judge_max_tokens"]),
           "CALIB_MIN": str(c["calibration_min"]), "FORCE": "0",
           "SKIP_CALIB": "0" if c["calibrate"] else "1"}
    subprocess.run(["bash", str(BASE / "dpo/judge_chain.sh")], cwd=BASE, env=env, check=True)
    report = missing_verdicts(out)
    if any(v["missing"] for v in report.values()):
        # one more pass for transient failures (the chain resumes: only the missing verdicts are asked again)
        subprocess.run(["bash", str(BASE / "dpo/judge_chain.sh")], cwd=BASE, env={**env, "SKIP_CALIB": "1"},
                       check=True)
        report = missing_verdicts(out)
    # Some verdicts never come back (reasoning beyond judge_max_tokens, or ended without a verdict): JZ run, about
    # 110 of 63,000, the same at every retry since the judge is greedy. Tolerated up to max_missing_share per
    # generator and dataset, recorded, and the pairs simply skip them; above, fail so that Airflow retries.
    (out / "judge_missing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    over = {k: v for k, v in report.items() if v["share"] > c["max_missing_share"]}
    if over:
        raise RuntimeError(f"verdicts manquants au-delà de {c['max_missing_share']:.1%} : {over} ; relancer")


def missing_verdicts(out):
    """Missing judge verdicts per generator/dataset: 4B = usable samples left unjudged (all are judged), 35B = items
    left in its todo stages, over its prompts."""
    sys.path.insert(0, str(BASE / "dpo"))
    import qwen_chosen_all as Q
    report = {}
    for ds in DATASETS:
        d = out / GENERATORS[1] / ds
        st = json.loads(run("jz_todo_rejected.py", "--dir", d, "--gen", GENERATORS[1], capture=True).strip().splitlines()[-1])
        Q.GEN = GENERATORS[1]
        _, samples, *_ = Q.state(d)
        usable = sum(Q.usable(s) for ss in samples.values() for s in ss.values())
        n = st.get("verify", 0) + st.get("citfail", 0) + st.get("refusal_samples_todo", 0)
        report[f"{GENERATORS[1]}/{ds}"] = {"missing": n, "of": usable, "share": n / max(usable, 1), "detail": st}
        d = out / GENERATORS[0] / ds
        n, detail = 0, {}
        for stage in ("pass1", "verify", "refusal"):
            todo = d / f"remaining_{stage}.jsonl"
            run("qwen_chosen_all.py", "--gen", GENERATORS[0], "--prefer-quoted", "todo",
                "--dir", d, "--stage", stage, "--ks=-1,0,1", "--out", todo, capture=True)
            detail[stage] = len(rows(todo))
            n += detail[stage]
        prompts = len(rows(d / "prompts.jsonl"))
        report[f"{GENERATORS[0]}/{ds}"] = {"missing": n, "of": prompts, "share": n / max(prompts, 1), "detail": detail}
    return report


def crossjudge(c, out):
    """Mistral cross-judge of the chosen refusals (column chosen_mistral_judge of the release; informational, no
    pair is removed on it: decision of 2026-10-07). Skipped unless crossjudge is true (column left null)."""
    if not c.get("crossjudge"):
        print("crossjudge: false : pas de contre-jugement Mistral (colonne chosen_mistral_judge vide)")
        return
    key = dotenv("LLM_API_KEY")
    if not key:
        raise ValueError("LLM_API_KEY absente (environnement ou .env) : clé de l'endpoint Mistral")
    run("crossjudge_mistral.py", "run", "--set", "jz_delta", "--out", out / "crossjudge/judged.jsonl",
        "--url", c["mistral_url"], "--model", c["mistral_model"], "--concurrency", c["mistral_concurrency"],
        env={**os.environ, "LLM_API_KEY": key, "JZ_RUN": str(out), "JZ_PAIRS": str(out / "pairs_v2/delta_pairs.jsonl"),
             "JZ_CROSSJUDGE_REJECTED": "1" if c["crossjudge_rejected"] else "0"})
    judged = {r["key"] for r in rows(out / "crossjudge/judged.jsonl") if r["mistral_judge"] is not None}
    want = {f"{r['dataset']}:{r['pid']}" for r in rows(out / "pairs_v2/delta_pairs.jsonl") if r["category"].startswith("U_")}
    lost = len(want - judged)
    if lost > c["max_missing_share"] * max(len(want), 1):
        raise RuntimeError(f"contre-jugement : {lost}/{len(want)} chosen sans verdict ; relancer")


def load_config(raw):
    """Run config: DAG params, plus the JSON handed over by dataset_rag_dpo_prompts (from_prompts) when it triggered
    the run; dropped from the config afterwards so that the resume lock compares the effective values."""
    c = json.loads(raw)
    extra = c.pop("from_prompts", None)
    if extra:
        c.update(json.loads(extra) if isinstance(extra, str) else extra)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("prepare", "chosen", "rejected", "judge", "build", "filter", "crossjudge",
                                      "release", "audit"))
    a = ap.parse_args()
    c = load_config(os.environ["DPO_RUN_CONFIG"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", c["run_name"]):
        raise ValueError("run_name : 1 à 80 lettres/chiffres/points/tirets/underscores")
    for name in ("seed", "generation_concurrency", "judge_concurrency", "judge_max_tokens"):
        if isinstance(c[name], bool) or not isinstance(c[name], int) or c[name] < (0 if name == "seed" else 1):
            raise ValueError(f"{name}: entier invalide")
    if c["judge_concurrency"] < 2:
        raise ValueError("judge_concurrency >= 2 (deux branches de jugement)")
    if isinstance(c["mistral_concurrency"], bool) or not isinstance(c["mistral_concurrency"], int) \
            or not 1 <= c["mistral_concurrency"] <= 12:
        raise ValueError("mistral_concurrency : 1 à 12 (plafond Mistral : 12 appels au total, tous jobs confondus)")
    if not 0 <= float(c["max_missing_share"]) <= 0.1:
        raise ValueError("max_missing_share : entre 0 et 0.1")
    if c.get("limit_questions") is not None and (
            isinstance(c["limit_questions"], bool) or not isinstance(c["limit_questions"], int)
            or c["limit_questions"] < 1):
        raise ValueError("limit_questions doit être null ou un entier positif")
    out = BASE / "airflow_runs/dataset_rag_dpo_delta" / c["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "config.json"
    if lock.exists() and json.loads(lock.read_text()) != c:
        raise ValueError("Configuration différente : choisir un nouveau run_name")
    if not lock.exists():
        lock.write_text(json.dumps(c, indent=2), encoding="utf-8")
    if a.stage == "prepare":
        # Refuse changed prompts on resume before touching existing judge state.
        if (out / "inputs.manifest.json").exists():
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                prepare(c, Path(tmp))
                if (Path(tmp) / "inputs.manifest.json").read_bytes() != (out / "inputs.manifest.json").read_bytes():
                    raise ValueError("Sources modifiées : choisir un nouveau run_name")
        else:
            prepare(c, out)
    elif a.stage in ("chosen", "rejected"):
        generate(c, out, a.stage == "chosen")
    elif a.stage == "judge":
        judge(c, out)
    elif a.stage == "build":
        if not (BASE / "tatqa_dataset_train.json").is_file():
            raise ValueError("tatqa_dataset_train.json requis pour le contrôle des fuites TAT-QA")
        pairs = out / "pairs"
        pairs.mkdir(exist_ok=True)
        run("build_p10_delta.py", "--jz", "--tokenizer", c["tokenizer"],
            env={**os.environ, "JZ_RUN": str(out), "JZ_OUT": str(pairs)})
        if not rows(pairs / "delta_pairs.jsonl"):
            raise RuntimeError("Aucune paire DPO retenue")
    elif a.stage == "filter":
        # decision of 2026-10-07: U_bad_refusal pairs with a single judge defect removed (pairs_v2 = jz_delta_v2)
        run("filter_jz_delta.py", "--src", out / "pairs", "--out", out / "pairs_v2", "--jz-run", out)
    elif a.stage == "crossjudge":
        crossjudge(c, out)
    elif a.stage == "release":
        if not c.get("policy_tokenizer") or not resolve(c["policy_tokenizer"]).is_dir():
            raise ValueError("policy_tokenizer : dossier local du tokenizer de la politique (règle des 4 096 tokens)")
        # parquet configs as published (policy tokenizer for the 4,096-token rule, splits by question/table against
        # the configs of the hub repo at split_reference_revision, or by hash alone when it is null); no upload
        run("publish_jz_full_offpolicy.py", "--tokenizer", c["policy_tokenizer"], "--out", out / "release",
            "--revision", c["split_reference_revision"] or "none", "--pairs-dir", out / "pairs_v2", "--jz-run", out,
            "--crossjudge", out / "crossjudge/judged.jsonl" if c.get("crossjudge") else "none")
    else:
        for name in ("pairs", "pairs_v2"):
            run("check_pairs_static.py", "--pairs", out / name / "delta_pairs.jsonl",
                "--tokenizer", c["tokenizer"], "--out", out / name / "static_audit.json")


if __name__ == "__main__":
    main()
