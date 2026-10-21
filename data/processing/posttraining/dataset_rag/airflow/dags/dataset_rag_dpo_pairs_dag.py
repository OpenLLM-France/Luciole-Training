"""DPO RAG pairs, two modes (param mode):

delta     validated held-out prompts -> 35B/4B -> judge -> pairs -> filter -> Mistral cross-judge -> release parquets
          (the Jean Zay run of 2026-10-06/07, without the upload)
onpolicy  prompts -> policy samples (vLLM started over SSH on the GPU host) -> two-pass judge -> pairs -> rejected
          recheck -> refine -> release parquets (configs hotpotqa / tatqa, without the upload)

Prompts: DAG dataset_rag_dpo_prompts (jz_run), or for onpolicy the held-out pools of the published configs
(prompt_source heldout_v2, replay to rebuild them without any LLM call). See airflow/DPO.md for prerequisites.
"""
from __future__ import annotations

import os
import shlex
import shutil
from datetime import timedelta
from pathlib import Path

try:
    from airflow.sdk import DAG
except ImportError:
    from airflow import DAG
try:
    from airflow.providers.standard.operators.bash import BashOperator
    from airflow.providers.standard.operators.python import BranchPythonOperator
except ImportError:
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import BranchPythonOperator

local = Path(__file__).resolve().parents[2]
BASE_DIR = Path(os.environ.get("DATASET_RAG_DIR", str(local)))
PYTHON = os.environ.get("DATASET_RAG_PYTHON") or (
    str(BASE_DIR / "venv/bin/python") if (BASE_DIR / "venv/bin/python").exists()
    else shutil.which("python3") or "python3")
LLM_POOL = os.environ.get("DATASET_RAG_LLM_POOL", "llm_api")

dag = DAG(
    dag_id="dataset_rag_dpo_pairs",
    description="DPO RAG : paires delta (35B / 4B) ou on-policy (politique SFT)",
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "dataset_rag", "retries": 2,
                  "retry_delay": timedelta(minutes=5)},
    tags=["dataset_rag", "dpo", "delta", "onpolicy"],
    params={
        "run_name": "default",
        "mode": "delta",  # delta | onpolicy
        "input_run": "dpo/data/jz_run",
        "limit_questions": None,
        "seed": 42,
        # delta: the three generator / judge endpoints are supplied
        "chosen_url": "http://127.0.0.1:8000/v1",
        "chosen_model": "Qwen3.5-35B-A3B",
        "chosen_dtype": "bf16",
        "rejected_url": "http://127.0.0.1:8001/v1",
        "rejected_model": "Qwen3.5-4B",
        "rejected_dtype": "bf16",
        # onpolicy: prompts of input_run (jz_run) or the held-out pools of the published configs (heldout_v2);
        # replay (heldout_v2 only) reuses their samples and verdicts and must give the published files
        "prompt_source": "jz_run",
        "heldout_v2_dir": "dpo/data/onpolicy_v2_sources",
        "replay": False,
        # onpolicy: policy endpoint as given, or null = vLLM started by the DAG on policy_serve.ssh_host (Docker),
        # reached through an SSH forward and removed at the end; keys and defaults in dpo_pipeline.SERVE_DEFAULTS
        "policy_url": None,
        "policy_model": "luciole_policy",  # served name
        "policy_name": "sft_luciole_1B_instruct_newrag-step_4152",  # `model` field of the pairs
        "policy_serve": {"ssh_host": "dgx", "ssh_config": None,
                         "model_dir": "lora_instruct_ablation/models/sft_newrag_step4152"},
        "calib_from": None,  # calibration set (onpolicy, heldout_v2): null = input_run
        "judge_url": "http://127.0.0.1:8002/v1",
        "judge_model": "Qwen3.8-27B",
        "judge_dtype": "bf16",
        "judge_key_file": ".judge_key",
        "generation_concurrency": 16,
        "judge_concurrency": 12,
        "judge_max_tokens": 16000,
        "calibrate": False,  # true: judge calibration on calib/ of calib_from (or input_run) first
        "calibration_min": 0.9,
        "max_missing_share": 0.01,
        "tokenizer": "Qwen/Qwen3.5-4B",
        "mistral_url": "http://localhost:8000/v1",
        "mistral_model": "Mistral-Small-3.2-24B-Instruct-2506-FP8",
        "mistral_concurrency": 6,
        "crossjudge": False,  # Mistral second opinion on the chosen refusals (column chosen_mistral_judge)
        "crossjudge_rejected": False,
        "policy_tokenizer": None,  # dossier local du tokenizer de la politique (sft_newrag_step4152)
        "split_reference_revision": None,
        # set by dataset_rag_dpo_prompts when it triggers this DAG: JSON of the params above, merged before validation
        "from_prompts": None,
    },
)


# Only the task that calls Mistral takes a slot of llm_api (the pool is the Mistral budget: 2 slots = 12 calls); the
# generators, the policy server and the Qwen judge are bounded by generation_concurrency / judge_concurrency.
def task(task_id: str, stage: str, llm: bool = False, **kw) -> BashOperator:
    # Jinja is rendered into an environment value, never interpolated as shell
    # source. The runner builds subprocess argument lists and validates run_name.
    return BashOperator(
        task_id=task_id, dag=dag, cwd=str(BASE_DIR), append_env=True,
        env={"DPO_RUN_CONFIG": "{{ params | tojson }}"},
        bash_command="set -euo pipefail\n" + shlex.quote(PYTHON) + " "
                     + shlex.quote(str(BASE_DIR / "airflow/dpo_pipeline.py"))
                     + " " + stage,
        **({"pool": LLM_POOL} if llm else {}), **kw,
    )


def pick_mode(**ctx):
    # the mode, as the runner reads it: from_prompts (trigger of dataset_rag_dpo_prompts) wins over the params
    import json
    p = dict(ctx["params"])
    extra = p.get("from_prompts")
    if extra:
        p.update(json.loads(extra) if isinstance(extra, str) else extra)
    return ["generate_policy"] if p.get("mode") == "onpolicy" else ["generate_chosen_35b", "generate_rejected_4b"]


prepare = task("prepare_prompts", "prepare")
mode = BranchPythonOperator(task_id="mode", dag=dag, python_callable=pick_mode)

chosen = task("generate_chosen_35b", "chosen")
rejected = task("generate_rejected_4b", "rejected")
judge = task("calibrate_and_judge", "judge")
build = task("build_delta_pairs", "build")
filt = task("filter_u_bad_refusal", "filter")
cross = task("crossjudge_mistral", "crossjudge", llm=True)
release = task("format_release_configs", "release")

policy = task("generate_policy", "policy")
stop = task("stop_policy_server", "stop_policy", trigger_rule="all_done")
ojudge = task("calibrate_and_judge_policy", "onpolicy_judge")
obuild = task("build_onpolicy_pairs", "onpolicy_build")
recheck = task("recheck_rejected", "recheck")
refine = task("refine_pairs", "refine")
orelease = task("format_onpolicy_configs", "onpolicy_release")

audit = task("audit_lengths_and_format", "audit", trigger_rule="none_failed_min_one_success")

prepare >> mode >> [chosen, rejected] >> judge >> build >> filt >> cross >> release >> audit
mode >> policy >> ojudge >> obuild >> recheck >> refine >> orelease >> audit
policy >> stop
