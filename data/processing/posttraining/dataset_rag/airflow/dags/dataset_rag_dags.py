"""Airflow DAGs for the Luciole RAG dataset pipeline.

One DAG per dataset family, all built by the same factory. Each task wraps an
existing CLI script from dataset_rag/ via BashOperator, so the pipeline logic
stays in the scripts and Airflow only handles orchestration, retries and logs.

Pipelines:
  - dataset_rag_hotpotqa_en : augment -> evaluate/filter -> formatting_sft
  - dataset_rag_hotpotqa_fr : idem, prompts FR + juge qualité FR
  - dataset_rag_tatqa       : convert -> augment -> evaluate/filter -> formatting_sft
  - dataset_rag_frenchqa_piaf : convert multilingual -> augment -> evaluate/filter
                                -> formatting_sft (--mixed-system-prompt)

Reasoning traces are generated with the shared OpenRAG prompt
(formatting_sft.build_chat_system_prompt) so the generation prompt matches the
inference prompt, per row (citation placement inline vs References is drawn at
generation and reused at formatting). Unanswerable rows are generated with that
same prompt too, and the evaluate/filter step keeps only the ones where the model
actually refused (``eval_model_refused``). Traces carry no "Final Answer" line, so
the evaluate/filter tasks run with ``--no-answer-match``: correctness is gated by
the factual judge (+ citation F1) rather than exact answer-string match.

Outputs land in <dataset_rag>/airflow_runs/<dag_id>/<run_name>/. Re-triggering
with the same run_name resumes the augmentation from its checkpoint file
(no LLM calls are re-paid); use a fresh run_name for a from-scratch build.

The LLM-heavy tasks share the "llm_api" pool (create it once, see README) so
two DAGs never hammer the LLM endpoint at the same time.
"""

from __future__ import annotations

import os
import shutil
import sys
from datetime import timedelta
from pathlib import Path

try:  # Airflow >= 3 public DAG-authoring API
    from airflow.sdk import DAG
except ImportError:  # Airflow 2.x
    from airflow import DAG

try:  # Airflow >= 3
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:  # Airflow 2.x
    from airflow.operators.bash import BashOperator

def _resolve_base_dir() -> Path:
    """Repository root: DATASET_RAG_DIR, else two levels above this file.

    Set DATASET_RAG_DIR when the DAG file is deployed apart from the code
    (e.g. mounted alone in a shared Airflow instance)."""
    override = os.environ.get("DATASET_RAG_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2]  # airflow/dags/ -> repository root


BASE_DIR = _resolve_base_dir()


def _resolve_python() -> str:
    """Interpreter that has the pipeline deps (aiohttp, datasets, rank-bm25, ...)."""
    override = os.environ.get("DATASET_RAG_PYTHON")
    if override:
        return override
    venv_python = BASE_DIR / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return shutil.which("python3") or sys.executable


PYTHON = _resolve_python()
LLM_POOL = os.environ.get("DATASET_RAG_LLM_POOL", "llm_api")

# Jinja fragments shared by every command (rendered by Airflow at run time).
WORKDIR = 'WORKDIR="' + str(BASE_DIR) + '/airflow_runs/{{ dag.dag_id }}/{{ params.run_name }}"'
LIMIT_OPT = "{% if params.limit %}--limit {{ params.limit }}{% endif %}"
SEED_OPT = "--seed {{ params.seed }}"

DEFAULT_ARGS = {
    "owner": "dataset_rag",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

DEFAULT_PARAMS = {
    # Same run_name => augmentation resumes from checkpoint. New name => fresh run.
    "run_name": "default",
    # Row cap for smoke tests (e.g. 100); null = full dataset.
    "limit": None,
    "seed": 42,
    # Passed to formatting_sft.py --format: "chat" or "prompt_completion".
    "sft_format": "chat",
}

# Mixed-difficulty distractor selection (converters). Empty distractor_mix
# falls back to the legacy BM25 start-rank behaviour.
MIX_PARAMS = {
    "distractor_mix": "hard=2,medium=5,random=2",
    "verify_hard": True,
    "evidence_query": True,
    "dense_model": "",
}

MIX_OPTS = (
    "{% if params.distractor_mix %}"
    "--distractor-mix {{ params.distractor_mix }}"
    "{% if params.verify_hard %} --verify-hard{% endif %}"
    "{% if params.evidence_query %} --evidence-query{% endif %}"
    "{% if params.dense_model %} --dense-model {{ params.dense_model }}{% endif %}"
    "{% endif %}"
)


def _bash(
    dag: DAG,
    task_id: str,
    body: str,
    llm: bool = False,
    required_outputs: tuple[str, ...] = (),
) -> BashOperator:
    """Wrap a pipeline command and reject missing or empty task artifacts."""
    output_checks = [
        f'if [ ! -s {path} ]; then '
        f'echo "Task {task_id}: required output is missing or empty" >&2; exit 1; fi'
        for path in required_outputs
    ]
    command = "\n".join(
        ["set -euo pipefail", WORKDIR, 'mkdir -p "$WORKDIR"', body, *output_checks]
    )
    kwargs = {"pool": LLM_POOL} if llm else {}
    return BashOperator(
        task_id=task_id,
        bash_command=command,
        cwd=str(BASE_DIR),
        append_env=True,
        dag=dag,
        **kwargs,
    )


def _formatting_cmd(input_path: str, language: str, mixed_system_prompt: bool = False) -> str:
    parts = [
        PYTHON, "formatting_sft.py",
        "--input", input_path,
        "--language", language,
        "--format {{ params.sft_format }}",
        SEED_OPT,
    ]
    if mixed_system_prompt:
        parts.append("--mixed-system-prompt")
    return " ".join(parts)


def _formatting_output(input_path: str) -> str:
    """Return the output filename produced by formatting_sft.py for the run format."""
    return input_path.removesuffix(".jsonl") + "_{{ params.sft_format }}.jsonl"


def build_hotpotqa_dag(dag_id: str, language: str, dataset_cache: str) -> DAG:
    dag = DAG(
        dag_id=dag_id,
        description=f"HotpotQA {language.upper()}: augment -> evaluate/filter -> SFT format",
        schedule=None,
        catchup=False,
        default_args=DEFAULT_ARGS,
        params={
            **DEFAULT_PARAMS,
            "dataset_cache": dataset_cache,
            # Fraction of created unanswerable rows keeping one gold hop
            # (hard refusal with partial evidence).
            "partial_hop_ratio": 0.5,
        },
        tags=["dataset_rag", "hotpotqa", language],
    )

    prefix = f'"$WORKDIR"/hotpotqa_{language}_augmented'

    augment = _bash(
        dag,
        "augment",
        f"{PYTHON} augment_hotpotqa.py"
        ' --dataset "{{ params.dataset_cache }}"'
        f" --language {language}"
        f" --output {prefix}"
        " --unanswerable-partial-hop-ratio {{ params.partial_hop_ratio }}"
        f" {SEED_OPT} {LIMIT_OPT}",
        llm=True,
        required_outputs=(f"{prefix}.jsonl",),
    )

    # --llm-judge-factual is mandatory: the filter always requires judge score 5,
    # so without it only unanswerable rows would survive.
    judge_fr = " --llm-judge-french-quality" if language == "fr" else ""
    evaluate = _bash(
        dag,
        "evaluate_and_filter",
        f"{PYTHON} evaluate_and_filter.py"
        f" --input {prefix}.jsonl"
        f" --output {prefix}_evaluated.jsonl"
        f" --eval-chunks --llm-judge-factual{judge_fr} --llm-judge-refusal --no-answer-match --filter"
        f" --export {prefix}_stats.json",
        llm=True,
        required_outputs=(
            f"{prefix}_evaluated.jsonl",
            f"{prefix}_evaluated_filtered.jsonl",
            f"{prefix}_stats.json",
        ),
    )

    formatting = _bash(
        dag,
        "formatting_sft",
        _formatting_cmd(f"{prefix}_evaluated_filtered.jsonl", language),
        required_outputs=(
            _formatting_output(f"{prefix}_evaluated_filtered.jsonl"),
        ),
    )

    augment >> evaluate >> formatting
    return dag


def build_tatqa_dag() -> DAG:
    dag = DAG(
        dag_id="dataset_rag_tatqa",
        description="TATQA: convert -> augment -> evaluate/filter -> SFT format",
        schedule=None,
        catchup=False,
        default_args=DEFAULT_ARGS,
        params={
            **DEFAULT_PARAMS,
            **MIX_PARAMS,
            "input_json": str(BASE_DIR / "tatqa_dataset_train.json"),
            "num_distractors": 9,
            "distractor_start_rank": 1,
            # Fraction of created unanswerable rows keeping partial gold
            # evidence (table-text questions).
            "partial_ratio": 0.5,
        },
        tags=["dataset_rag", "tatqa", "en"],
    )

    prefix = '"$WORKDIR"/tatqa'

    # convert may call the LLM (--verify-hard) => llm pool.
    convert = _bash(
        dag,
        "convert",
        f"{PYTHON} convert_tatqa_to_augment_format.py"
        ' --input "{{ params.input_json }}"'
        f" --output {prefix}_converted.jsonl"
        " --num-distractors {{ params.num_distractors }}"
        " --distractor-start-rank {{ params.distractor_start_rank }}"
        f" {MIX_OPTS} {SEED_OPT} {LIMIT_OPT}",
        llm=True,
        required_outputs=(f"{prefix}_converted.jsonl",),
    )

    augment = _bash(
        dag,
        "augment",
        f"{PYTHON} augment_tatqa.py"
        f" --dataset {prefix}_converted.jsonl"
        f" --output {prefix}_augmented"
        " --unanswerable-partial-ratio {{ params.partial_ratio }}"
        f" {SEED_OPT} {LIMIT_OPT}",
        llm=True,
        required_outputs=(f"{prefix}_augmented.jsonl",),
    )

    evaluate = _bash(
        dag,
        "evaluate_and_filter",
        f"{PYTHON} evaluate_tatqa.py"
        f" --input {prefix}_augmented.jsonl"
        f" --output {prefix}_augmented_evaluated.jsonl"
        " --eval-chunks --llm-judge-factual --llm-judge-refusal --no-answer-match --filter"
        f" --export {prefix}_stats.json",
        llm=True,
        required_outputs=(
            f"{prefix}_augmented_evaluated.jsonl",
            f"{prefix}_augmented_evaluated_filtered.jsonl",
        ),
    )

    formatting = _bash(
        dag,
        "formatting_sft",
        _formatting_cmd(f"{prefix}_augmented_evaluated_filtered.jsonl", "en"),
        required_outputs=(
            _formatting_output(f"{prefix}_augmented_evaluated_filtered.jsonl"),
        ),
    )

    convert >> augment >> evaluate >> formatting
    return dag


def build_multilingual_dag(dag_id: str, dataset: str, language: str) -> DAG:
    dag = DAG(
        dag_id=dag_id,
        description=f"{dataset} ({language}): convert -> augment -> evaluate/filter -> SFT format",
        schedule=None,
        catchup=False,
        default_args=DEFAULT_ARGS,
        params={
            **DEFAULT_PARAMS,
            **MIX_PARAMS,
            "num_distractors": 9,
            "distractor_start_rank": 10,
        },
        tags=["dataset_rag", "multilingual", language],
    )

    slug = dataset.replace("-", "_")
    prefix = f'"$WORKDIR"/{slug}_{language}'

    # convert may call the LLM (--verify-hard) => llm pool.
    convert = _bash(
        dag,
        "convert",
        f"{PYTHON} convert_multilingual_qa.py"
        f" --dataset {dataset}"
        f" --language {language}"
        f" --output {prefix}_converted.jsonl"
        " --num-distractors {{ params.num_distractors }}"
        " --distractor-start-rank {{ params.distractor_start_rank }}"
        f" {MIX_OPTS} {SEED_OPT} {LIMIT_OPT}",
        llm=True,
        required_outputs=(f"{prefix}_converted.jsonl",),
    )

    augment = _bash(
        dag,
        "augment",
        f"{PYTHON} augment_hotpotqa.py"
        f" --dataset {prefix}_converted.jsonl"
        f" --language {language}"
        f" --output {prefix}_augmented"
        f" {SEED_OPT}",
        llm=True,
        required_outputs=(f"{prefix}_augmented.jsonl",),
    )

    judge_fr = " --llm-judge-french-quality" if language == "fr" else ""
    evaluate = _bash(
        dag,
        "evaluate_and_filter",
        f"{PYTHON} evaluate_and_filter.py"
        f" --input {prefix}_augmented.jsonl"
        f" --output {prefix}_augmented_evaluated.jsonl"
        f" --eval-chunks --llm-judge-factual{judge_fr} --llm-judge-refusal --no-answer-match --filter"
        f" --export {prefix}_stats.json",
        llm=True,
        required_outputs=(
            f"{prefix}_augmented_evaluated.jsonl",
            f"{prefix}_augmented_evaluated_filtered.jsonl",
            f"{prefix}_stats.json",
        ),
    )

    formatting = _bash(
        dag,
        "formatting_sft",
        _formatting_cmd(
            f"{prefix}_augmented_evaluated_filtered.jsonl",
            language,
            mixed_system_prompt=(language != "en"),
        ),
        required_outputs=(
            _formatting_output(f"{prefix}_augmented_evaluated_filtered.jsonl"),
        ),
    )

    convert >> augment >> evaluate >> formatting
    return dag


hotpotqa_en = build_hotpotqa_dag(
    "dataset_rag_hotpotqa_en", "en", str(BASE_DIR / "hotpotqa_cache_train")
)
# Le cache FR actuel est en .bak sur le disque ; ajustable via le param dataset_cache.
hotpotqa_fr = build_hotpotqa_dag(
    "dataset_rag_hotpotqa_fr", "fr", str(BASE_DIR / "hotpotqa_french_cache_train.bak")
)
tatqa = build_tatqa_dag()
frenchqa_piaf = build_multilingual_dag(
    "dataset_rag_frenchqa_piaf", "frenchqa-piaf", "fr"
)
