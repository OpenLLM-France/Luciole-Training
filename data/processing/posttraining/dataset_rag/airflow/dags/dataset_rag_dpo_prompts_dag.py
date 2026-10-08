"""Delta DPO RAG, prompts: held-out pools -> usable prompts -> DAG unanswerable versions -> context judge -> jz_run.

Output airflow_runs/dataset_rag_dpo_prompts/<run_name>/jz_run, the input_run of dataset_rag_dpo_delta. With a "delta"
config (dict of dataset_rag_dpo_delta params), that DAG is triggered at the end with input_run filled in; without, the
prompt set is only built (one prompt set, several generation runs). See airflow/DPO.md.
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
    from airflow.providers.standard.operators.python import ShortCircuitOperator
    from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
except ImportError:
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import ShortCircuitOperator
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator

local = Path(__file__).resolve().parents[2]
BASE_DIR = Path(os.environ.get("DATASET_RAG_DIR", str(local)))
PYTHON = os.environ.get("DATASET_RAG_PYTHON") or (
    str(BASE_DIR / "venv/bin/python") if (BASE_DIR / "venv/bin/python").exists()
    else shutil.which("python3") or "python3")
LLM_POOL = os.environ.get("DATASET_RAG_LLM_POOL", "llm_api")

dag = DAG(
    dag_id="dataset_rag_dpo_prompts",
    description="DPO RAG Delta : prompts held-out, versions sans réponse du DAG, juge de contexte",
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "dataset_rag", "retries": 2,
                  "retry_delay": timedelta(minutes=5)},
    tags=["dataset_rag", "dpo", "delta"],
    params={
        "run_name": "default",
        # TAT-QA run of the dataset_rag_tatqa DAG whose converted rows and [CONTEXT] cache are reused (2026-07-22/23)
        "tatqa_converted": "airflow_runs/dataset_rag_tatqa/full/tatqa_converted.jsonl",
        "tatqa_ctx_cache": "airflow_runs/dataset_rag_tatqa/full/tatqa_augmented_contextualization_cache.jsonl",
        "mistral_concurrency": 4,
        "judge_url": "http://localhost:8000/v1",
        "judge_model": "Qwen3.8-27B",
        "judge_key_file": ".judge_key",
        "judge_concurrency": 16,
        "judge_max_tokens": 16000,
        "max_missing_share": 0.01,
        "calib_from": None,  # jz_run whose calib/ (judge calibration set) is copied; null: none
        # params of dataset_rag_dpo_delta (endpoints, tokenizers...); input_run is set here, run_name defaults to this
        # run_name. null = build the prompts only.
        "delta": None,
    },
)


def task(task_id: str, stage: str, llm: bool = False) -> BashOperator:
    # Jinja is rendered into an environment value, never interpolated as shell source.
    return BashOperator(
        task_id=task_id, dag=dag, cwd=str(BASE_DIR), append_env=True,
        env={"DPO_RUN_CONFIG": "{{ params | tojson }}"},
        bash_command="set -euo pipefail\n" + shlex.quote(PYTHON) + " "
                     + shlex.quote(str(BASE_DIR / "airflow/dpo_prompts_pipeline.py")) + " " + stage,
        **({"pool": LLM_POOL} if llm else {}),
    )


check = task("check_inputs", "check")
prompts = task("build_heldout_prompts", "prompts")
unans = task("build_unanswerable_versions", "unans", llm=True)
contexts = task("judge_unanswerable_contexts", "contexts", llm=True)
jz_run = task("build_prompt_set", "jz_run")
# The delta config travels as ONE JSON string (from_prompts): templated values are rendered as text, so ints and
# booleans passed one by one would reach dataset_rag_dpo_delta as strings.
has_delta = ShortCircuitOperator(task_id="delta_requested", dag=dag,
                                 python_callable=lambda **ctx: bool(ctx["params"].get("delta")))
trigger = TriggerDagRunOperator(
    task_id="trigger_dataset_rag_dpo_delta", dag=dag, trigger_dag_id="dataset_rag_dpo_delta",
    conf={"from_prompts": "{{ dict(dict({'run_name': params.run_name}, **params.delta), "
                          "input_run='airflow_runs/dataset_rag_dpo_prompts/' ~ params.run_name ~ '/jz_run') | tojson }}"},
    wait_for_completion=False)
check >> prompts >> unans >> contexts >> jz_run >> has_delta >> trigger
