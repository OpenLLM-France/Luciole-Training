"""Delta DPO RAG: validated held-out prompts -> 35B/4B -> judge -> pairs -> filter -> Mistral cross-judge -> release
parquets (the Jean Zay run of 2026-10-06/07, without the upload). Prompts: DAG dataset_rag_dpo_prompts.

Inference servers are supplied by the run configuration; no GPU server is
started on the Airflow worker. See airflow/DPO.md for prerequisites.
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
except ImportError:
    from airflow.operators.bash import BashOperator

local = Path(__file__).resolve().parents[2]
BASE_DIR = Path(os.environ.get("DATASET_RAG_DIR", str(local)))
PYTHON = os.environ.get("DATASET_RAG_PYTHON") or (
    str(BASE_DIR / "venv/bin/python") if (BASE_DIR / "venv/bin/python").exists()
    else shutil.which("python3") or "python3")
LLM_POOL = os.environ.get("DATASET_RAG_LLM_POOL", "llm_api")

dag = DAG(
    dag_id="dataset_rag_dpo_delta",
    description="DPO RAG Delta : chosen 35B confirmé, rejected 4B jugé fautif",
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "dataset_rag", "retries": 2,
                  "retry_delay": timedelta(minutes=5)},
    tags=["dataset_rag", "dpo", "delta"],
    params={
        "run_name": "default",
        "input_run": "dpo/data/jz_run",
        "limit_questions": None,
        "seed": 42,
        "chosen_url": "http://127.0.0.1:8000/v1",
        "chosen_model": "Qwen3.5-35B-A3B",
        "chosen_dtype": "bf16",
        "rejected_url": "http://127.0.0.1:8001/v1",
        "rejected_model": "Qwen3.5-4B",
        "rejected_dtype": "bf16",
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


def task(task_id: str, stage: str, llm: bool = False) -> BashOperator:
    # Jinja is rendered into an environment value, never interpolated as shell
    # source. The runner builds subprocess argument lists and validates run_name.
    return BashOperator(
        task_id=task_id, dag=dag, cwd=str(BASE_DIR), append_env=True,
        env={"DPO_RUN_CONFIG": "{{ params | tojson }}"},
        bash_command="set -euo pipefail\n" + shlex.quote(PYTHON) + " "
                     + shlex.quote(str(BASE_DIR / "airflow/dpo_pipeline.py"))
                     + " " + stage,
        **({"pool": LLM_POOL} if llm else {}),
    )

prepare = task("prepare_prompts", "prepare")
chosen = task("generate_chosen_35b", "chosen", llm=True)
rejected = task("generate_rejected_4b", "rejected", llm=True)
judge = task("calibrate_and_judge", "judge", llm=True)
build = task("build_delta_pairs", "build")
filt = task("filter_u_bad_refusal", "filter")
cross = task("crossjudge_mistral", "crossjudge", llm=True)
release = task("format_release_configs", "release")
audit = task("audit_lengths_and_format", "audit")
prepare >> [chosen, rejected] >> judge >> build >> filt >> cross >> release >> audit
