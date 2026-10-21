#!/usr/bin/env python3
"""Airflow CLI adapter of the DPO RAG pair generation (no Airflow dependency). Two modes (param `mode`):

delta     chosen Qwen3.5-35B-A3B confirmed by the judge, rejected Qwen3.5-4B judged faulty: the Jean Zay run of
          2026-10-06/07 up to the configs *_full_offpolicy_qwen35_35b_a3b as published (27e169e)
          prepare -> chosen / rejected -> judge -> build -> filter -> crossjudge -> release -> audit
onpolicy  chosen and rejected both sampled from the SFT policy (vLLM started on a GPU host over SSH, or policy_url):
          the chain of the configs hotpotqa / tatqa as published (27e169e)
          prepare -> policy (+ stop_policy) -> onpolicy_judge -> onpolicy_build -> recheck -> refine -> onpolicy_release
          -> audit

On-policy prompts (`prompt_source`): jz_run (input_run, the output of dataset_rag_dpo_prompts, generic instruction) or
heldout_v2 (the four held-out pools of the published configs, generic and strict, in heldout_v2_dir). With replay, the
samples and verdicts of the published run are reused: no LLM call, and the release must give the published files.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATASETS = ("hotpotqa", "tatqa")
GENERATORS = ("qwen35_35b_a3b", "qwen35_4b")
STAGES = {"delta": ("prepare", "chosen", "rejected", "judge", "build", "filter", "crossjudge", "release", "audit"),
          "onpolicy": ("prepare", "policy", "stop_policy", "onpolicy_judge", "onpolicy_build", "recheck", "refine",
                       "onpolicy_release", "audit")}
# Policy sampling of the published on-policy configs (run_pilot.sh, sampling.manifest.json of the four pools)
SAMPLING = {"k": 4, "temperature": 0.8, "top_p": 0.95, "max_tokens": 1024, "greedy": True}
# Four pools of the published configs hotpotqa / tatqa (prepare_onpolicy_publication.CONFIGS), in heldout_v2_dir.
# replay: the audit files of that run; a new run judges again with the rule of 2026-10-01 (run_audit_onpolicy.sh) and
# keeps the reviews that concern the prompt (excluded pids, context found answerable), not those of a chosen text.
HELDOUT_V2 = (
    {"name": "hotpotqa_generic", "dataset": "hotpotqa", "instruction": "generic", "src": "heldout_full_sft4152",
     "replay": {"audit": "v1.1", "recheck": "audit_heldout_sft4152/rejected_think.jsonl",
                "u_review": "audit_heldout_sft4152/u_chosen_manual_review.json", "drop_u_supported": False}},
    {"name": "hotpotqa_strict", "dataset": "hotpotqa", "instruction": "strict", "src": "heldout_strictpool_sft4152",
     "twin": "hotpotqa_generic",
     "replay": {"recheck": "audit_onpolicy_20261001/rejected_think_hotpotqa_strictpool.jsonl"}},
    {"name": "tatqa_generic", "dataset": "tatqa", "instruction": "generic", "src": "heldout_tatqa_generic_sft4152",
     "u_review": "audit_manual/u_chosen_manual_review_tatqa.json",
     "exclude": "audit_manual/exclude_heldout_tatqa_generic_sft4152.json",
     "replay": {"recheck": "audit_onpolicy_20261001/rejected_think_tatqa.jsonl"}},
    {"name": "tatqa_strict", "dataset": "tatqa", "instruction": "strict", "src": "heldout_tatqa_strict_sft4152",
     "twin": "tatqa_generic", "exclude": "audit_manual/exclude_heldout_tatqa_strict_sft4152.json",
     "replay": {"recheck": "audit_onpolicy_20261001/rejected_think_tatqa_strict.jsonl"}},
)
# files of the published configs hotpotqa / tatqa (HF commit 27e169e, unchanged at 8e143a6) that a replay must give
PUBLISHED_ONPOLICY = {
    "hotpotqa/train.parquet": "84439ddd8229928343ec33f498f596ef1ca4657c0ffc4fa93d07736e04289e68",
    "hotpotqa/val.parquet": "16ad8b12d7eb0fd91721cfd3ea68fc4fa0a84fb437855013ec997b12f58dc442",
    "tatqa/train.parquet": "fc48fd8aa6385e36ee5e8facb6fbd834601108c64700cc9e0a2db0b023b35d9a",
    "tatqa/val.parquet": "1b12b4fb50486b7f12c8272c89d3d907f4f6c01b4073f7ef19bc82a5473b2a07",
}
SERVE_DEFAULTS = {
    "ssh_host": "dgx",               # alias or user@host; the GPU host where vLLM runs in Docker
    "ssh_config": None,              # ssh config file (null: the worker's ~/.ssh/config)
    "model_dir": "lora_instruct_ablation/models/sft_newrag_step4152",  # on the host, relative to its $HOME
    "image": "nvcr.io/nvidia/vllm:26.02-py3",
    "remote_port": 8016,
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.30,
    "min_free_gb": 50,               # MemAvailable below that: the host is busy, the task fails (no cache flush)
    "thermal_pauser": None,  # script on the host that freezes the container above 80C (null: none)
    "startup_timeout": 900,
}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, records):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def served_models(url, key=None):
    """/v1/models of an OpenAI-compatible server (vLLM: served name, root path, max_model_len), or the error."""
    import urllib.request
    req = urllib.request.Request(url.rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r).get("data")
    except Exception as e:  # recorded, not fatal: some endpoints restrict /models
        return {"error": repr(e)[:300]}


def judge_key(c):
    key = resolve(c["judge_key_file"])
    if not key.is_file() or not key.read_text().strip():
        raise ValueError("judge_key_file doit contenir la clé du juge (ou 'none' pour vLLM local)")
    return key


def generator_env(c, prefix):
    """Environment of sample_policy.py: SAMPLER_API_KEY from <prefix>_key_file (null = server without a key)."""
    name = c.get(f"{prefix}_key_file")
    if not name:
        return None
    key = resolve(name)
    if not key.is_file() or not key.read_text().strip():
        raise ValueError(f"{prefix}_key_file : fichier absent ou vide ({key})")
    return {**os.environ, "SAMPLER_API_KEY": key.read_text().strip()}


def check_prompts(ds, p, cv):
    """Prompt set of dataset_rag_dpo_prompts: context verdict agrees with the label, no validation / evaluation
    question, no legacy unpadded clone."""
    for r in p:
        expected = "unanswerable" if r["is_unanswerable"] else "answerable"
        if cv.get(r["pid"], {}).get("verdict") != expected:
            raise ValueError(f"{ds}/{r['pid']}: verdict de contexte absent ou contradictoire")
        if r.get("in_8802410_validation") or r.get("in_paired_eval") \
                or (ds == "hotpotqa" and r.get("pilot_split") == "val"):
            raise ValueError(f"{ds}/{r['pid']}: fuite de validation/évaluation")
        if r["is_unanswerable"] and r.get("heldout_drop_reason") == "nogold_clone":
            raise ValueError(f"{ds}/{r['pid']}: ancien clone sans rembourrage")


def jz_datasets(run):
    """Datasets of a jz_run (or of a prepared delta run): the directories of the first generator holding prompts,
    hotpotqa / tatqa first. A prompt set from an SFT run has only its own (e.g. hotpotqa_fr)."""
    found = {d.name for d in (run / GENERATORS[0]).glob("*") if (d / "prompts.jsonl").is_file()}
    if not found:
        raise ValueError(f"{run}: aucun jeu de prompts ({GENERATORS[0]}/<dataset>/prompts.jsonl)")
    return [ds for ds in DATASETS if ds in found] + sorted(found - set(DATASETS))


def check_delta_datasets(out, stage):
    """The delta stages after the generation (judge chain, pairs, filter, cross-judge, release) still know
    hotpotqa / tatqa only: stop before any judge call rather than half-way."""
    other = set(jz_datasets(out)) - set(DATASETS)
    if other:
        raise ValueError(f"étape {stage} pas encore adaptée aux jeux {sorted(other)} (seule la génération l'est)")


def limit(c, p, qids=None):
    """Smoke run: the first limit_questions questions by hash, so that a question's answerable and unanswerable
    versions (and its strict twin, same qid) stay together."""
    if not c.get("limit_questions"):
        return p
    if qids is None:
        qids = set(sorted({r["qid"] for r in p}, key=lambda q: hashlib.sha256(
            f"{c['seed']}:{q}".encode()).hexdigest())[:int(c["limit_questions"])])
    return [r for r in p if r["qid"] in qids]


def copy_calib(c, out, source, manifest):
    for ds in DATASETS:
        d = out / "calib" / ds
        d.mkdir(parents=True, exist_ok=True)
        for name in ("prompts.jsonl", "context_verdicts.jsonl", "chosen_qwen35_35b_a3b.jsonl",
                     "reference_verified.jsonl", "only.jsonl"):
            shutil.copyfile(source / "calib" / ds / name, d / name)
            manifest["inputs"][f"calib/{ds}/{name}"] = sha256(d / name)


def prepare(c, out):
    if c["mode"] == "onpolicy":
        return prepare_onpolicy(c, out)
    source = resolve(c["input_run"])
    manifest = {"config": c, "inputs": {}}
    for ds in jz_datasets(source):
        p = rows(source / GENERATORS[0] / ds / "prompts.jsonl")
        other = rows(source / GENERATORS[1] / ds / "prompts.jsonl")
        if p != other or not p or len({r["pid"] for r in p}) != len(p):
            raise ValueError(f"{ds}: prompts vides, dupliqués ou différents entre générateurs")
        cv = {r["pid"]: r for r in rows(source / GENERATORS[0] / ds / "context_verdicts.jsonl")}
        check_prompts(ds, p, cv)
        p = limit(c, p)
        for gen in GENERATORS:
            d = out / gen / ds
            d.mkdir(parents=True, exist_ok=True)
            write(d / "prompts.jsonl", p)
            write(d / "context_verdicts.jsonl", [cv[r["pid"]] for r in p])
        manifest["inputs"][ds] = {"prompts": len(p), "sha256": {
            name: sha256(out / GENERATORS[0] / ds / name) for name in ("prompts.jsonl", "context_verdicts.jsonl")}}
    if c["calibrate"]:
        copy_calib(c, out, source, manifest)
    (out / "inputs.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- on-policy pools


def pools(c):
    """On-policy pools of the run: name, dataset, instruction (generic | strict), twin (generic pool of a strict one),
    refine options and, for heldout_v2, the source dir. Generic pools come before the strict ones (twins)."""
    if c["prompt_source"] == "jz_run":
        return [{"name": f"{ds}_generic", "dataset": ds, "instruction": "generic", "twin": None, "audit": "recheck",
                 "u_review": None, "exclude": None, "drop_u_supported": True, "recheck": None} for ds in DATASETS]
    out = []
    for spec in HELDOUT_V2:
        p = {"twin": None, "audit": "recheck", "u_review": spec.get("u_review"), "exclude": spec.get("exclude"),
             "drop_u_supported": True, "recheck": None,
             **{k: v for k, v in spec.items() if k != "replay"}}
        if c["replay"]:
            p.update(spec["replay"])
        out.append(p)
    return out


def pool_dir(out, pool):
    return out / "policy" / pool["name"]


def prepare_onpolicy(c, out):
    manifest = {"config": c, "inputs": {}}
    if c["prompt_source"] == "jz_run":
        source = resolve(c["input_run"])
        for pool in pools(c):
            ds = pool["dataset"]
            p = rows(source / GENERATORS[0] / ds / "prompts.jsonl")
            if not p or len({r["pid"] for r in p}) != len(p):
                raise ValueError(f"{ds}: prompts vides ou dupliqués")
            cv = {r["pid"]: r for r in rows(source / GENERATORS[0] / ds / "context_verdicts.jsonl")}
            check_prompts(ds, p, cv)
            p = limit(c, p)
            d = pool_dir(out, pool)
            d.mkdir(parents=True, exist_ok=True)
            write(d / "prompts.jsonl", p)
            write(d / "context_verdicts.jsonl", [cv[r["pid"]] for r in p])
            manifest["inputs"][pool["name"]] = {"prompts": len(p), "sha256": {
                name: sha256(d / name) for name in ("prompts.jsonl", "context_verdicts.jsonl")}}
    else:
        v2 = resolve(c["heldout_v2_dir"])
        qids = {}
        for pool in pools(c):
            src, d = v2 / pool["src"], pool_dir(out, pool)
            d.mkdir(parents=True, exist_ok=True)
            p = rows(src / "prompts.jsonl")
            # a strict pool keeps the questions of its generic twin (same qid)
            if c.get("limit_questions") and pool["twin"] is None:
                qids[pool["dataset"]] = set(sorted({r["qid"] for r in p}, key=lambda q: hashlib.sha256(
                    f"{c['seed']}:{q}".encode()).hexdigest())[:int(c["limit_questions"])])
            p = limit(c, p, qids.get(pool["dataset"]))
            pids = {r["pid"] for r in p}
            write(d / "prompts.jsonl", p)
            write(d / "context_verdicts.jsonl", [r for r in rows(src / "context_verdicts.jsonl") if r["pid"] in pids])
            files = {"prompts.jsonl": None, "context_verdicts.jsonl": None}
            if c["replay"]:
                for name in ("samples.jsonl", "sampling.manifest.json", "scored.jsonl", "verified.jsonl"):
                    shutil.copyfile(src / name, d / name)
                    files[name] = None
                shutil.copyfile(v2 / pool["recheck"], d / "rejected_think.jsonl")
                files["rejected_think.jsonl"] = pool["recheck"]
            for key, name in (("u_review", "u_review.json"), ("exclude", "exclude_pids.json")):
                if pool[key]:
                    shutil.copyfile(v2 / pool[key], d / name)
                    files[name] = pool[key]
            manifest["inputs"][pool["name"]] = {"source": pool["src"], "prompts": len(p), "sha256": {
                name: sha256(d / name) for name in files}, "from": {k: v for k, v in files.items() if v}}
    if c["calibrate"] and not c["replay"]:
        copy_calib(c, out, resolve(c.get("calib_from") or c["input_run"]), manifest)
    (out / "inputs.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- policy server (vLLM over SSH)


class Remote:
    """Commands on the GPU host through ONE multiplexed SSH connection (ControlMaster): one connection per command
    gets the worker banned by fail2ban on the jump hosts. Optional local forward to the vLLM port."""

    def __init__(self, s):
        self.s = s
        self.tmp = Path(tempfile.mkdtemp(prefix="dpo-ssh-"))
        # unix socket paths are limited to 108 bytes: short name under /tmp
        self.sock = f"/tmp/dpo-{secrets.token_hex(4)}.sock"
        cfg = []
        if s["ssh_config"]:
            # ssh refuses a -F file owned by another user (the results mount belongs to the host user, the worker
            # runs as uid 998): private copy, mode 600
            copy = self.tmp / "config"
            shutil.copyfile(resolve(s["ssh_config"]), copy)
            copy.chmod(0o600)
            cfg = ["-F", str(copy)]
        self.base = ["ssh", *cfg, "-o", f"ControlPath={self.sock}", "-o", "BatchMode=yes",
                     "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
        self.port = None

    def open(self, remote_port=None):
        fwd = []
        if remote_port:
            with socket.socket() as sk:
                sk.bind(("127.0.0.1", 0))
                self.port = sk.getsockname()[1]
            fwd = ["-L", f"127.0.0.1:{self.port}:127.0.0.1:{remote_port}"]
        subprocess.run(self.base + ["-o", "ControlMaster=yes", "-o", "ControlPersist=yes",
                                    "-o", "ExitOnForwardFailure=yes", "-fN", *fwd, self.s["ssh_host"]],
                       check=True, timeout=180)

    def sh(self, script, check=True, timeout=600):
        r = subprocess.run(self.base + [self.s["ssh_host"], "bash -s"], input=script, text=True,
                           capture_output=True, timeout=timeout)
        if check and r.returncode:
            raise RuntimeError(f"{self.s['ssh_host']} : code {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        return r

    def close(self):
        subprocess.run(self.base + ["-O", "exit", self.s["ssh_host"]], capture_output=True, timeout=60)
        shutil.rmtree(self.tmp, ignore_errors=True)


def serve_config(c):
    s = {**SERVE_DEFAULTS, **(c.get("policy_serve") or {})}
    unknown = set(s) - set(SERVE_DEFAULTS)
    if unknown:
        raise ValueError(f"policy_serve : clés inconnues {sorted(unknown)}")
    if not re.fullmatch(r"[A-Za-z0-9@._-]+", str(s["ssh_host"])):
        raise ValueError("policy_serve.ssh_host : alias ou user@host")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(c["policy_model"])):
        raise ValueError("policy_model : nom servi en lettres/chiffres/points/tirets/underscores")
    for k in ("model_dir", "image") + (("thermal_pauser",) if s["thermal_pauser"] else ()):
        if not re.fullmatch(r"[A-Za-z0-9/@:._-]+", str(s[k])) or ".." in str(s[k]):
            raise ValueError(f"policy_serve.{k} : chemin ou image invalide")
    return s


def container_name(c):
    return "vllm_dpo_pairs_" + re.sub(r"[^A-Za-z0-9_.-]", "_", c["run_name"])[:40]


def stop_script(c):
    n = container_name(c)
    return (f'P="$HOME/.{n}.pauser.pid"; if [ -f "$P" ]; then kill "$(cat "$P")" 2>/dev/null || true; rm -f "$P"; fi\n'
            f"docker unpause {n} >/dev/null 2>&1 || true\ndocker rm -f {n} >/dev/null 2>&1 || true\n"
            f'echo "conteneur {n} arrêté"\n')


def start_script(c, s):
    n = container_name(c)
    model = s["model_dir"] if s["model_dir"].startswith("/") else "$HOME/" + s["model_dir"]
    pauser = ""
    if s["thermal_pauser"]:
        tp = s["thermal_pauser"] if s["thermal_pauser"].startswith("/") else "$HOME/" + s["thermal_pauser"]
        pauser = (f'if [ -f "{tp}" ]; then LOG="$HOME/.{n}.pauser.log" nohup bash "{tp}" {n} 80 72 '
                  f'>/dev/null 2>&1 < /dev/null & echo $! > "$HOME/.{n}.pauser.pid"; '
                  f'else echo "régulateur thermique absent : {tp}"; fi\n')
    return f"""set -euo pipefail
M="{model}"
[ -f "$M/model.safetensors" ] || {{ echo "politique absente : $M"; exit 3; }}
free_gb=$(awk '/^MemAvailable/{{print int($2/1048576)}}' /proc/meminfo)
echo "free_gb=$free_gb"
docker ps --format '   {{{{.Names}}}}  {{{{.Image}}}}  {{{{.Status}}}}'
[ "$free_gb" -ge {int(s["min_free_gb"])} ] || {{ echo "moins de {int(s["min_free_gb"])} Go disponibles : hôte occupé"; exit 4; }}
# transformers 5.x tokenizer class unknown to the vLLM image (run_pilot.sh); idempotent
sed -i 's/"TokenizersBackend"/"PreTrainedTokenizerFast"/' "$M/tokenizer_config.json"
{stop_script(c)}docker run -d --name {n} --gpus all --ipc=host -p 127.0.0.1:{int(s["remote_port"])}:8000 -v "$M:/model:ro" \\
    {s["image"]} vllm serve /model --served-model-name {c["policy_model"]} --max-model-len {int(s["max_model_len"])} \\
    --gpu-memory-utilization {float(s["gpu_memory_utilization"])} --chat-template /model/chat_template.jinja >/dev/null
{pauser}echo "host=$(hostname)"
echo "model_path=$(readlink -f "$M")"
echo "model_sha256=$(sha256sum "$M/model.safetensors" | cut -d' ' -f1)"
"""


@contextlib.contextmanager
def policy_server(c):
    """(url, served name, server facts): policy_url as given, else vLLM started in Docker on policy_serve.ssh_host,
    reached through an SSH forward, stopped on exit (stop_policy also stops it if this task is killed)."""
    if c.get("policy_url"):
        yield c["policy_url"], c["policy_model"], {"launched": False, "url": c["policy_url"]}
        return
    s = serve_config(c)
    r = Remote(s)
    r.open(s["remote_port"])
    try:
        out = r.sh(start_script(c, s)).stdout
        print(out, flush=True)
        facts = dict(line.split("=", 1) for line in out.splitlines() if re.match(r"^(host|model_path|model_sha256)=", line))
        url = f"http://127.0.0.1:{r.port}/v1"
        deadline = time.time() + int(s["startup_timeout"])
        while not isinstance(served_models(url), list):
            if time.time() > deadline:
                raise RuntimeError("vLLM pas prêt à temps")
            st = r.sh(f"docker inspect -f '{{{{.State.Running}}}}' {container_name(c)}", check=False).stdout.strip()
            if st != "true":
                logs = r.sh(f"docker logs --tail 40 {container_name(c)} 2>&1", check=False).stdout
                raise RuntimeError(f"le conteneur vLLM s'est arrêté :\n{logs}")
            time.sleep(10)
        yield url, c["policy_model"], {"launched": True, "ssh_host": s["ssh_host"], "image": s["image"],
                                       "container": container_name(c), **facts}
    finally:
        try:
            print(r.sh(stop_script(c), check=False, timeout=120).stdout, flush=True)
        finally:
            r.close()


def stop_policy(c):
    """Teardown (trigger_rule all_done): the policy container is removed even when the generation task was killed."""
    if c["mode"] != "onpolicy" or c.get("policy_url") or c["replay"]:
        print("rien à arrêter")
        return
    s = serve_config(c)
    r = Remote(s)
    try:
        r.open()
        print(r.sh(stop_script(c), check=False, timeout=120).stdout)
    finally:
        r.close()


def complete_samples(d, k=SAMPLING["k"]):
    """Prompts of d whose samples are missing, failed or incomplete (greedy + k)."""
    last = {r["pid"]: r for r in rows(d / "samples.jsonl")} if (d / "samples.jsonl").exists() else {}
    expected = {-1, *range(k)}
    return [p["pid"] for p in rows(d / "prompts.jsonl") if last.get(p["pid"], {}).get("errors")
            or {s["k"] for s in last.get(p["pid"], {}).get("samples", [])} != expected]


def policy(c, out):
    if c["replay"]:
        bad = {pool["name"]: len(complete_samples(pool_dir(out, pool))) for pool in pools(c)}
        if any(bad.values()):
            raise RuntimeError(f"rejeu : échantillons incomplets {bad}")
        print("rejeu : échantillons de la publication réutilisés")
        return
    with policy_server(c) as (url, served, facts):
        for pool in pools(c):
            d = pool_dir(out, pool)
            for _ in range(3):  # a failed request is asked again on the next pass (run_pilot.sh)
                run("sample_policy.py", "--prompts", d / "prompts.jsonl", "--out", d / "samples.jsonl",
                    "--api-url", url, "--model", served, "--k", SAMPLING["k"],
                    "--temperature", SAMPLING["temperature"], "--top-p", SAMPLING["top_p"],
                    "--max-tokens", SAMPLING["max_tokens"], "--seed", c["seed"],
                    "--concurrency", c["generation_concurrency"])
                if not complete_samples(d):
                    break
            else:
                raise RuntimeError(f"{pool['name']}: génération incomplète ; relancer")
            (d / "sampling.manifest.json").write_text(json.dumps({
                "model": facts.get("model_path") or c["policy_name"], "model_sha256": facts.get("model_sha256"),
                "host": facts.get("host"), "prompt_sha256": sha256(d / "prompts.jsonl"),
                "k_sampled": SAMPLING["k"], "temperature": SAMPLING["temperature"], "top_p": SAMPLING["top_p"],
                "seed": c["seed"], "greedy": SAMPLING["greedy"], "max_tokens": SAMPLING["max_tokens"],
                "served_model": served, "server": facts, "server_models": served_models(url),
                "samples_sha256": sha256(d / "samples.jsonl"), "concurrency": c["generation_concurrency"]},
                indent=2), encoding="utf-8")


# ---------------------------------------------------------------- on-policy judging


def score(c, key, d, out_file, only=None, thinking=True, samples=None, twice=True):
    """score_pilot_samples.py; run twice (a failed call is judged again on the next run)."""
    args = ["--prompts", d / "prompts.jsonl", "--samples", samples or d / "samples.jsonl",
            "--contexts", d / "context_verdicts.jsonl", "--out", out_file,
            "--judge-url", c["judge_url"], "--judge-model", c["judge_model"], "--judge-key-file", key,
            "--concurrency", c["judge_concurrency"],
            *(["--max-tokens", c["judge_max_tokens"], "--timeout", 1800] if thinking else
              ["--no-thinking", "--max-tokens", 800, "--timeout", 300]),
            *(["--only", only] if only else [])]
    for _ in range(2 if twice else 1):
        run("score_pilot_samples.py", *args)


def missing(path):
    """(verdicts missing, verdicts asked) in a score_pilot_samples output: last row of a (pid, k), judge None on a
    non-empty, non-truncated response."""
    if not Path(path).exists():
        return 0, 0
    last = {(r["pid"], r["k"]): r for r in rows(path)}
    n = sum(r["judge"] is None and r["finish_reason"] != "length" and bool(r["text"].strip()) for r in last.values())
    return n, len(last)


def tolerate(c, report, what):
    """Verdicts that never come back (reasoning beyond judge_max_tokens, no verdict): tolerated up to
    max_missing_share, recorded; above, fail so that Airflow retries."""
    for v in report.values():
        v["share"] = v["missing"] / max(v["of"], 1)
    over = {k: v for k, v in report.items() if v["share"] > c["max_missing_share"]}
    if over:
        raise RuntimeError(f"{what} : verdicts manquants au-delà de {c['max_missing_share']:.1%} : {over} ; relancer")


def calibrate(c, out, key):
    """Step 0 of dpo/judge_chain.sh: ~190 completions already judged with reasoning, judged again by this judge;
    stop when the answer_correct agreement is below calibration_min."""
    for ds in DATASETS:
        d = out / "calib" / ds
        score(c, key, d, d / "verified_jz.jsonl", only=d / "only.jsonl", samples=d / "chosen_qwen35_35b_a3b.jsonl",
              twice=False)
    r = subprocess.run([sys.executable, str(BASE / "dpo/calib_compare.py"), "--dir", out / "calib/hotpotqa",
                        "--dir", out / "calib/tatqa", "--min", str(c["calibration_min"])],
                       cwd=BASE, capture_output=True, text=True)
    (out / "calib/agreement.json").write_text(r.stdout, encoding="utf-8")
    print(r.stdout, r.stderr, flush=True)
    if r.returncode:
        raise RuntimeError("calibration du juge en échec : " + r.stderr.strip()[-500:])


def onpolicy_judge(c, out):
    """Generic pools: run_two_pass.sh (pass 1 without reasoning on every sample, then the chosen candidates confirmed
    with reasoning, 3 rounds, greedy first). Strict pools: run_strict.sh (build_strict_pairs select / verify, 4 rounds)."""
    if c["replay"]:
        print("rejeu : verdicts de la publication réutilisés")
        return
    key = judge_key(c)
    if c["calibrate"]:
        calibrate(c, out, key)
    report = {}
    for pool in pools(c):
        d = pool_dir(out, pool)
        if pool["instruction"] == "generic":
            score(c, key, d, d / "scored.jsonl", thinking=False)
            for r in range(1, 4):
                todo = d / f"verify_round{r}.jsonl"
                run("select_chosen_candidates.py", "--prompts", d / "prompts.jsonl",
                    "--contexts", d / "context_verdicts.jsonl", "--scored", d / "scored.jsonl",
                    "--verified", d / "verified.jsonl", "--out", todo)
                if not todo.exists() or not todo.read_text().strip():
                    break
                score(c, key, d, d / "verified.jsonl", only=todo)
        else:
            run("build_strict_pairs.py", "select", "--dir", d, "--out", d / "pass1_only.jsonl")
            score(c, key, d, d / "scored.jsonl", only=d / "pass1_only.jsonl", thinking=False)
            for r in range(1, 5):
                todo = d / f"verify_round{r}.jsonl"
                run("build_strict_pairs.py", "verify", "--dir", d, "--out", todo)
                if not todo.exists() or not todo.read_text().strip():
                    break
                score(c, key, d, d / "verified.jsonl", only=todo)
        for name in ("scored.jsonl", "verified.jsonl"):
            n, of = missing(d / name)
            report[f"{pool['name']}/{name}"] = {"missing": n, "of": of}
    (out / "judge_missing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    tolerate(c, report, "jugement")


def onpolicy_build(c, out):
    n = 0
    for pool in pools(c):
        d = pool_dir(out, pool)
        (d / "verified.jsonl").touch()  # absent when no chosen candidate needed confirming (small runs)
        if pool["instruction"] == "generic":
            run("build_pilot_pairs.py", "--prompts", d / "prompts.jsonl", "--contexts", d / "context_verdicts.jsonl",
                "--scored", d / "scored.jsonl", "--verified", d / "verified.jsonl", "--out-dir", d / "pairs",
                "--model", c["policy_name"])
        else:
            run("build_strict_pairs.py", "build", "--dir", d, "--model", c["policy_name"],
                "--twin-dir", pool_dir(out, next(p for p in pools(c) if p["name"] == pool["twin"])))
        n += sum(len(rows(d / "pairs" / f"{s}.jsonl")) for s in ("train", "val"))
    if not n:
        raise RuntimeError("Aucune paire DPO retenue")


def recheck_todo(pool, d):
    """Rejected to judge again with reasoning (run_audit_onpolicy.sh, 2026-10-01): every policy rejected of the pairs;
    on a strict pool, not the bare imposed phrase (checked lexically)."""
    todo = []
    for s in ("train", "val"):
        for p in rows(d / "pairs" / f"{s}.jsonl"):
            if p.get("rejected_source") != "policy":
                continue
            j = p.get("rejected_judge") or {}
            if pool["instruction"] == "strict" and (j.get("lexical_imposed_phrase") or p.get("rejected_is_exact_phrase")):
                continue
            todo.append({"pid": p["pid"], "k": p["rejected_k"]})
    return todo


def recheck(c, out):
    """Rejected judged again with reasoning (their pass-1 verdict had none), and audit of the chosen refusal bodies of
    the generic pools (audit_refusal_bodies.py: informational, u_chosen_bodies.jsonl, nothing is removed on it)."""
    if c["replay"]:
        print("rejeu : contre-jugements de la publication réutilisés")
        return
    key = judge_key(c)
    report = {}
    for pool in pools(c):
        d = pool_dir(out, pool)
        write(d / "recheck_only.jsonl", recheck_todo(pool, d))
        if (d / "recheck_only.jsonl").read_text().strip():
            score(c, key, d, d / "rejected_think.jsonl", only=d / "recheck_only.jsonl")
        else:
            (d / "rejected_think.jsonl").touch()
        n, of = missing(d / "rejected_think.jsonl")
        report[f"{pool['name']}/rejected_think.jsonl"] = {"missing": n, "of": of}
        bodies = d / "u_chosen_bodies.jsonl"
        n_u = sum(p["category"].startswith("U_") for s in ("train", "val") for p in rows(d / "pairs" / f"{s}.jsonl"))
        done = bodies.exists() and len(b := rows(bodies)) == n_u and all(r["judge"] is not None for r in b)
        if pool["instruction"] == "generic" and not done:  # the script rewrites its output: skipped once complete
            run("audit_refusal_bodies.py", "--pairs", d / "pairs/train.jsonl", d / "pairs/val.jsonl",
                "--out", d / "u_chosen_bodies.jsonl", "--judge-url", c["judge_url"], "--judge-model", c["judge_model"],
                "--judge-key-file", key, "--concurrency", c["judge_concurrency"])
    (out / "recheck_missing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    tolerate(c, report, "contre-jugement")


def refine(c, out):
    """refine_pairs.py: rejected that the reasoning judge accepts, unanswerable pairs whose rejected answer it finds
    supported (drop_u_supported), and the reviewed pids (u_review, exclude) -> pairs_audited/."""
    for pool in pools(c):
        d = pool_dir(out, pool)
        run("refine_pairs.py", "--pairs-dir", d / "pairs", "--prompts", d / "prompts.jsonl",
            "--rejected-recheck", d / "rejected_think.jsonl", "--out-dir", d / "pairs_audited",
            *(["--u-review", d / "u_review.json"] if pool["u_review"] else []),
            *(["--exclude-pids", d / "exclude_pids.json"] if pool["exclude"] else []),
            *(["--drop-u-supported"] if pool["drop_u_supported"] else []))


def onpolicy_release(c, out):
    """prepare_onpolicy_publication.py with the corrections of the 2026-10-02 audit, the 4,096-token rule (policy
    tokenizer) and the gold columns of 27e169e. HotpotQA of jz_run (no validation question left): ~10 % of the
    questions by hash. A replay must give the published files."""
    for f in ("tatqa_dataset_test_gold.json", "tatqa_dev.json", "hotpotqa_cache_validation",
              "tatqa_dataset_train.json", "hotpotqa_cache_train"):
        if not (BASE / f).exists():
            raise ValueError(f"{f} requis à la racine de dataset_rag (contrôles de fuite, colonnes gold)")
    sources = {}
    for pool in pools(c):
        d = pool_dir(out, pool)
        sources.setdefault(pool["dataset"], []).append(
            [str(d / "pairs_audited"), str(d / "prompts.jsonl"), pool["instruction"], pool["audit"]])
    (out / "release_sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    run("prepare_onpolicy_publication.py", "--out", out / "release", "--tokenizer", resolve(c["policy_tokenizer"]),
        "--sources", out / "release_sources.json", "--gold-columns",
        *(["--hotpotqa-split", "qid"] if c["prompt_source"] == "jz_run" else []))
    empty = json.loads((out / "release/manifest.json").read_text())["empty_configs"]
    if len(empty) == len(sources):
        raise RuntimeError("Aucune paire publiable")
    if empty:
        print(f"!! configs sans paire après corrections (non écrites) : {empty}")
    if c["replay"]:
        got = {f: sha256(out / "release" / f) for f in PUBLISHED_ONPOLICY}
        diff = {f: h[:12] for f, h in got.items() if h != PUBLISHED_ONPOLICY[f]}
        if diff:
            raise RuntimeError(f"rejeu différent de la publication (27e169e) : {diff}")
        print("rejeu identique à la publication (27e169e)")


# ---------------------------------------------------------------- delta


def generate(c, out, chosen):
    gen = GENERATORS[0 if chosen else 1]
    prefix = "chosen" if chosen else "rejected"
    env = generator_env(c, prefix)
    for ds in jz_datasets(out):
        d = out / gen / ds
        samples = d / f"chosen_{gen}.jsonl"
        run("sample_policy.py", "--prompts", d / "prompts.jsonl", "--out", samples,
            "--api-url", c[f"{prefix}_url"], "--model", c[f"{prefix}_model"],
            "--k", 2 if chosen else 4, "--temperature", 0.7 if chosen else 1.0,
            "--top-p", 0.80, "--top-k", 20, "--min-p", 0,
            "--max-tokens", 2048, "--seed", c["seed"],
            "--concurrency", c["generation_concurrency"],
            "--chat-template-kwargs", '{"enable_thinking": false}', env=env)
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
            "server_models": served_models(c[f"{prefix}_url"], (env or {}).get("SAMPLER_API_KEY")),
            "prompt_sha256": sha256(d / "prompts.jsonl"), "samples_sha256": sha256(samples),
            "k_sampled": 2 if chosen else 4, "temperature": 0.7 if chosen else 1.0, "top_p": 0.80, "top_k": 20,
            "min_p": 0.0, "seed": c["seed"], "greedy": True, "max_tokens": 2048, "enable_thinking": False,
            "concurrency": c["generation_concurrency"]}, indent=2), encoding="utf-8")


def judge(c, out):
    key = judge_key(c)
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
    tolerate(c, report, "jugement")


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
        report[f"{GENERATORS[1]}/{ds}"] = {"missing": n, "of": usable, "detail": st}
        d = out / GENERATORS[0] / ds
        n, detail = 0, {}
        for stage in ("pass1", "verify", "refusal"):
            todo = d / f"remaining_{stage}.jsonl"
            run("qwen_chosen_all.py", "--gen", GENERATORS[0], "--prefer-quoted", "todo",
                "--dir", d, "--stage", stage, "--ks=-1,0,1", "--out", todo, capture=True)
            detail[stage] = len(rows(todo))
            n += detail[stage]
        report[f"{GENERATORS[0]}/{ds}"] = {"missing": n, "of": len(rows(d / "prompts.jsonl")), "detail": detail}
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


def audit(c, out):
    if c["mode"] == "delta":
        for name in ("pairs", "pairs_v2"):
            run("check_pairs_static.py", "--pairs", out / name / "delta_pairs.jsonl",
                "--tokenizer", c["tokenizer"], "--out", out / name / "static_audit.json")
        return
    for ds in DATASETS:
        d = out / "release" / ds
        if not (d / "train.parquet").exists():  # config without any pair (onpolicy_release says so)
            continue
        run("check_pairs_static.py", "--pairs", d / "train.parquet", d / "val.parquet", "--group-by", "category",
            "--tokenizer", resolve(c["policy_tokenizer"]), "--out", d / "static_audit.json")


# ---------------------------------------------------------------- config


def load_config(raw):
    """Run config: DAG params, plus the JSON handed over by dataset_rag_dpo_prompts (from_prompts) when it triggered
    the run; dropped from the config afterwards so that the resume lock compares the effective values."""
    c = json.loads(raw)
    extra = c.pop("from_prompts", None)
    if extra:
        c.update(json.loads(extra) if isinstance(extra, str) else extra)
    return c


def is_int(v, lo):
    return not isinstance(v, bool) and isinstance(v, int) and v >= lo


def validate(c):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", c["run_name"]):
        raise ValueError("run_name : 1 à 80 lettres/chiffres/points/tirets/underscores")
    if c["mode"] not in STAGES:
        raise ValueError("mode : delta ou onpolicy")
    if c["prompt_source"] not in ("jz_run", "heldout_v2"):
        raise ValueError("prompt_source : jz_run ou heldout_v2")
    if c["mode"] == "delta" and (c["prompt_source"] != "jz_run" or c["replay"]):
        raise ValueError("mode delta : prompt_source jz_run, sans replay")
    if c["replay"] and (c["prompt_source"] != "heldout_v2" or c.get("limit_questions")):
        raise ValueError("replay : prompt_source heldout_v2 et limit_questions null")
    for name in ("seed", "generation_concurrency", "judge_concurrency", "judge_max_tokens"):
        if not is_int(c[name], 0 if name == "seed" else 1):
            raise ValueError(f"{name}: entier invalide")
    if c["judge_concurrency"] < 2:
        raise ValueError("judge_concurrency >= 2 (deux branches de jugement)")
    if not is_int(c["mistral_concurrency"], 1) or c["mistral_concurrency"] > 12:
        raise ValueError("mistral_concurrency : 1 à 12 (plafond Mistral : 12 appels au total, tous jobs confondus)")
    if not 0 <= float(c["max_missing_share"]) <= 0.1:
        raise ValueError("max_missing_share : entre 0 et 0.1")
    if c.get("limit_questions") is not None and not is_int(c["limit_questions"], 1):
        raise ValueError("limit_questions doit être null ou un entier positif")
    if c["mode"] == "onpolicy":
        if not c.get("policy_tokenizer"):
            raise ValueError("policy_tokenizer : dossier local du tokenizer de la politique (règle des 4 096 tokens)")
        if not c["replay"] and not c.get("policy_url"):
            serve_config(c)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=sorted({s for v in STAGES.values() for s in v}))
    a = ap.parse_args()
    c = load_config(os.environ["DPO_RUN_CONFIG"])
    validate(c)
    if a.stage not in STAGES[c["mode"]] and a.stage != "stop_policy":  # teardown: runs (no-op) in delta too
        raise ValueError(f"étape {a.stage} hors du mode {c['mode']}")
    out = BASE / "airflow_runs/dataset_rag_dpo_pairs" / c["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "config.json"
    if lock.exists() and json.loads(lock.read_text()) != c:
        raise ValueError("Configuration différente : choisir un nouveau run_name")
    if not lock.exists():
        lock.write_text(json.dumps(c, indent=2), encoding="utf-8")
    # Airflow stops a task with SIGTERM: exit through the finally blocks (vLLM container, SSH connection)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if c["mode"] == "delta" and a.stage in ("judge", "build", "filter", "crossjudge", "release", "audit"):
        check_delta_datasets(out, a.stage)
    if a.stage == "prepare":
        # Refuse changed prompts on resume before touching existing judge state.
        if (out / "inputs.manifest.json").exists():
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
        # the configs of the hub repo at split_reference_revision, or by hash alone when it is null, gold columns of
        # 27e169e); no upload
        run("publish_jz_full_offpolicy.py", "--tokenizer", c["policy_tokenizer"], "--out", out / "release",
            "--revision", c["split_reference_revision"] or "none", "--pairs-dir", out / "pairs_v2", "--jz-run", out,
            "--crossjudge", out / "crossjudge/judged.jsonl" if c.get("crossjudge") else "none", "--gold-columns")
    elif a.stage == "policy":
        policy(c, out)
    elif a.stage == "stop_policy":
        stop_policy(c)
    elif a.stage == "onpolicy_judge":
        onpolicy_judge(c, out)
    elif a.stage == "onpolicy_build":
        onpolicy_build(c, out)
    elif a.stage == "recheck":
        recheck(c, out)
    elif a.stage == "refine":
        refine(c, out)
    elif a.stage == "onpolicy_release":
        onpolicy_release(c, out)
    else:
        audit(c, out)


if __name__ == "__main__":
    main()
