"""Airflow DAGs for the Luciole RAG dataset pipeline.

One DAG per dataset family, all built by the same factory. Each task wraps an
existing CLI script from dataset_rag/ via BashOperator, so the pipeline logic
stays in the scripts and Airflow only handles orchestration, retries and logs.

Pipelines:
  - dataset_rag_hotpotqa_en : augment -> evaluate/filter -> formatting_sft
  - dataset_rag_hotpotqa_fr : idem, prompts FR + juge qualité FR
  - dataset_rag_hotpotqa_fr_abstracts : convert (Mvanypersele/hotpotqa-fr-abstracts)
                                -> augment || evaluate_stream -> evaluate/filter -> formatting ;
                                générateur Qwen3.8-27B, juges Mistral à 6 pendant la génération
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

For TAT-QA only, a ``contextualize_ratio`` fraction of the rows gets OpenRAG
``[CONTEXT]`` chunk summaries. Each retrieved chunk is mapped back to its full
original financial report before contextualization. HotpotQA and the
multilingual QA sources do not expose those original documents in this pipeline,
so contextual retrieval is deliberately disabled for them rather than treating
a mixed retrieval result as one document.

Outputs land in <dataset_rag>/airflow_runs/<dag_id>/<run_name>/. Re-triggering
with the same run_name resumes the augmentation from its checkpoint file
(no LLM calls are re-paid); use a fresh run_name for a from-scratch build.

The LLM-heavy tasks share the "llm_api" pool (create it once with 2 slots, see
README) so two DAGs never hammer the LLM endpoint at the same time. A slot stands
for half of the Mistral budget (12 concurrent calls across all jobs): a task that
may send more than 6 concurrent Mistral calls takes both slots, which keeps the
historical one-LLM-task-at-a-time behaviour. Parallel evaluation (HotpotQA FR
abstracts) runs a non-Mistral generator (1 slot) next to the Mistral judges
capped at 6 (1 slot), streamed by evaluate_stream.py while rows are generated.

Generator and judges may run on different servers (``gen_url`` / ``judge_url``
params, keys read from ``gen_key_file`` / ``judge_key_file``). The judge server
also serves chunk contextualization and hard-distractor verification. A judge
bound to localhost on a GPU host is reached through an SSH forward opened by each
LLM task for its own duration (``judge_ssh_host``).
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

# Tag shared by every DAG of this pipeline, to filter them on a shared Airflow instance.
BASE_TAGS = ["dataset_rag"]

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
    # Generator server (chat/completions URL) and a file holding its key; empty =
    # LLM_BASE_URL / LLM_API_KEY of .env. Relative paths start at dataset_rag/.
    "gen_url": "",
    "gen_key_file": "",
    # Judge server, also used for contextualization and --verify-hard.
    "judge_url": "",
    "judge_key_file": "",
    # Judge bound to 127.0.0.1:judge_remote_port on an SSH host (vLLM on a GPU host):
    # each LLM task forwards it for its own duration. Overrides judge_url.
    "judge_ssh_host": "",
    "judge_ssh_config": "airflow/ssh/config",
    "judge_remote_port": 8030,
}

# Exports the server routing read by utils.py (LLM_GEN_* / LLM_JUDGE_*), opening
# the SSH forward to the judge first when judge_ssh_host is set.
ENDPOINT_ENV = """\
{% if params.gen_url %}export LLM_GEN_BASE_URL="{{ params.gen_url }}"{% endif %}
{% if params.gen_key_file %}export LLM_GEN_API_KEY_FILE="{{ params.gen_key_file }}"{% endif %}
{% if params.judge_key_file %}export LLM_JUDGE_API_KEY_FILE="{{ params.judge_key_file }}"{% endif %}
{% if params.get("judge_model") %}export LLM_JUDGE_MODEL="{{ params.judge_model }}"{% endif %}
{% if params.judge_ssh_host %}
JUDGE_PORT=$(@PYTHON@ -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
ssh -F "{{ params.judge_ssh_config }}" -N -o ExitOnForwardFailure=yes \\
    -L "127.0.0.1:$JUDGE_PORT:127.0.0.1:{{ params.judge_remote_port }}" "{{ params.judge_ssh_host }}" &
JUDGE_TUNNEL=$!
trap 'kill "$JUDGE_TUNNEL" 2>/dev/null || true' EXIT
export LLM_JUDGE_BASE_URL="http://127.0.0.1:$JUDGE_PORT/v1/chat/completions"
for _ in $(seq 60); do
    @PYTHON@ airflow/check_judge_server.py "http://127.0.0.1:$JUDGE_PORT/v1/models" && break
    kill -0 "$JUDGE_TUNNEL" 2>/dev/null || { echo "SSH forward to the judge exited" >&2; exit 1; }
    sleep 2
done
@PYTHON@ airflow/check_judge_server.py "http://127.0.0.1:$JUDGE_PORT/v1/models" \\
    || { echo "judge not answering through the SSH forward" >&2; exit 1; }
{% elif params.judge_url %}export LLM_JUDGE_BASE_URL="{{ params.judge_url }}"{% endif %}
""".replace("@PYTHON@", PYTHON)

CONTEXTUALIZE_OPT = "--contextualize-ratio {{ params.contextualize_ratio }}"

# Generator / judge models (HotpotQA DAGs). Empty model = LLM_MODEL from .env.
# Judges run one after another, so judge_concurrency is the peak load on the
# judge model (Mistral: 12 concurrent calls max across all jobs).
LLM_PARAMS = {
    "gen_model": "",
    "gen_max_tokens": 1024,
    "gen_timeout": 120,
    # JSON for chat_template_kwargs, e.g. {"enable_thinking": false} for Qwen3.
    "gen_chat_template_kwargs": "",
    "gen_concurrency": 10,
    "judge_model": "",
    "judge_concurrency": 10,
    # "markup" (quote/cite markers restyled at formatting) or "ref" (numbered
    # documents + inline <ref id="N">excerpt</ref>, kept as generated).
    "citation_format": "markup",
}

GEN_OPTS = (
    "--concurrency {{ params.gen_concurrency }}"
    " --citation-format {{ params.citation_format }}"
    " --max-tokens {{ params.gen_max_tokens }} --timeout {{ params.gen_timeout }}"
    "{% if params.gen_model %} --model {{ params.gen_model }}{% endif %}"
    "{% if params.gen_chat_template_kwargs %}"
    " --chat-template-kwargs '{{ params.gen_chat_template_kwargs }}'{% endif %}"
)

JUDGE_OPTS = (
    "--judge-concurrency {{ params.judge_concurrency }}"
    "{% if params.judge_model %} --judge-model {{ params.judge_model }}{% endif %}"
)

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


# parallel_eval: generator and judges hit their servers together, so the
# generator must not be Mistral and the judges stay within 6 concurrent calls
# (half of the shared Mistral budget, whichever server judges).
PARALLEL_EVAL_GUARD = (
    "{% if not params.gen_model or 'mistral' in params.gen_model | lower"
    " or params.judge_concurrency | int > 6 %}"
    'echo "parallel_eval: non-Mistral gen_model and judge_concurrency <= 6 required" >&2; exit 1'
    "{% endif %}"
)


def _bash(
    dag: DAG,
    task_id: str,
    body: str,
    llm: bool = False,
    required_outputs: tuple[str, ...] = (),
    llm_slots: int = 2,
) -> BashOperator:
    """Wrap a pipeline command and reject missing or empty task artifacts."""
    output_checks = [
        f'if [ ! -s {path} ]; then '
        f'echo "Task {task_id}: required output is missing or empty" >&2; exit 1; fi'
        for path in required_outputs
    ]
    endpoints = [ENDPOINT_ENV] if llm else []
    command = "\n".join(
        ["set -euo pipefail", WORKDIR, 'mkdir -p "$WORKDIR"', *endpoints, body, *output_checks]
    )
    kwargs = {"pool": LLM_POOL, "pool_slots": llm_slots} if llm else {}
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


def build_hotpotqa_dag(
    dag_id: str,
    language: str,
    dataset_cache: str,
    *,
    abstracts_convert: bool = False,
    param_overrides: dict | None = None,
    parallel_eval: bool = False,
) -> DAG:
    """HotpotQA pipeline; ``abstracts_convert`` prepends the hotpotqa-fr-abstracts
    conversion, whose output then replaces ``dataset_cache``.

    ``parallel_eval`` judges rows while they are generated (evaluate_stream task
    next to augment). Generation and judges then hit the endpoint together, so the
    generator must not be Mistral and the judges stay at 6 concurrent calls (one
    pool slot each, see the module docstring).
    """
    steps = "convert -> " if abstracts_convert else ""
    extra_params = {}
    if abstracts_convert:
        extra_params = {
            "hf_config": "filtered_qwen_v2",
            "hf_split": "train",
            # "" = latest commit, resolved and logged by the convert task.
            "hf_revision": "",
            # Rows whose documents cannot fit this training budget are dropped.
            "max_seq_length": 8192,
            "reserve_tokens": 1536,
            # HF id or local dir. The Luciole repo is private: on hosts without
            # HF_TOKEN, point this at a local copy of the tokenizer.
            "tokenizer": "OpenLLM-France/Luciole-1B-SFT-1.1",
        }
    dag = DAG(
        dag_id=dag_id,
        description=f"HotpotQA {language.upper()}: {steps}augment -> evaluate/filter -> SFT format",
        schedule=None,
        catchup=False,
        default_args=DEFAULT_ARGS,
        params={
            **DEFAULT_PARAMS,
            "dataset_cache": dataset_cache,
            # Fraction of created unanswerable rows keeping one gold hop
            # (hard refusal with partial evidence).
            "partial_hop_ratio": 0.5,
            # A refusal now survives the filter only together with the
            # answerable row it was cloned from, which roughly halves the
            # yield. 0.22 keeps the ~11% final refusal share, matching tatqa.
            "unanswerable_ratio": 0.22,
            **LLM_PARAMS,
            **extra_params,
            **(param_overrides or {}),
        },
        tags=[*BASE_TAGS, "hotpotqa", language],
    )
    if parallel_eval:
        overrides = param_overrides or {}
        if "mistral" in overrides.get("gen_model", "mistral").lower() or overrides.get("judge_concurrency", 99) > 6:
            raise ValueError(f"{dag_id}: parallel_eval needs a non-Mistral gen_model and judge_concurrency <= 6")
    # One slot when generation and judging run side by side, both otherwise.
    llm_slots = 1 if parallel_eval else 2
    # The same rule at run time, since a trigger conf can override the defaults.
    guard = PARALLEL_EVAL_GUARD + "\n" if parallel_eval else ""

    prefix = f'"$WORKDIR"/hotpotqa_{language}_augmented'
    judge_cache = f"{prefix}_judge_cache.jsonl"
    dataset_arg = '"{{ params.dataset_cache }}"'

    convert = None
    if abstracts_convert:
        converted = f'"$WORKDIR"/hotpotqa_{language}_converted.jsonl'
        dataset_arg = converted
        # Local only (HF download + tokenizer), no LLM call => no pool.
        convert = _bash(
            dag,
            "convert",
            f"{PYTHON} convert_hotpotqa_fr_abstracts.py"
            " --config {{ params.hf_config }} --split {{ params.hf_split }}"
            "{% if params.hf_revision %} --revision {{ params.hf_revision }}{% endif %}"
            " --max-seq-length {{ params.max_seq_length }}"
            " --reserve-tokens {{ params.reserve_tokens }}"
            ' --tokenizer "{{ params.tokenizer }}"'
            f" --output {converted}",
            required_outputs=(converted,),
        )

    augment = _bash(
        dag,
        "augment",
        f"{guard}{PYTHON} augment_hotpotqa.py"
        f" --dataset {dataset_arg}"
        f" --language {language}"
        f" --output {prefix}"
        " --unanswerable-partial-hop-ratio {{ params.partial_hop_ratio }}"
        " --unanswerable-ratio {{ params.unanswerable_ratio }}"
        f" {GEN_OPTS} {SEED_OPT} {LIMIT_OPT}",
        llm=True,
        llm_slots=llm_slots,
        required_outputs=(f"{prefix}.jsonl",),
    )

    # --llm-judge-factual is mandatory: the filter always requires judge score 5,
    # so without it only unanswerable rows would survive.
    judge_fr = " --llm-judge-french-quality" if language == "fr" else ""
    judges = f"--llm-judge-factual{judge_fr} --llm-judge-refusal {JUDGE_OPTS}"

    stream = None
    if parallel_eval:
        # Judges each line as soon as augment appends it; verdicts go to the
        # judge cache, so evaluate_and_filter below only pays for failed calls.
        stream = _bash(
            dag,
            "evaluate_stream",
            f"{guard}{PYTHON} evaluate_stream.py"
            f" --input {prefix}.jsonl"
            f" --judge-cache {judge_cache}"
            f" {judges}",
            llm=True,
            llm_slots=llm_slots,
        )

    # The judge cache also makes a retry of this task resume instead of re-judging.
    evaluate = _bash(
        dag,
        "evaluate_and_filter",
        f"{guard}{PYTHON} evaluate_and_filter.py"
        f" --input {prefix}.jsonl"
        f" --output {prefix}_evaluated.jsonl"
        f" --eval-chunks {judges} --no-answer-match --filter"
        f" --judge-cache {judge_cache}"
        f" --export {prefix}_stats.json",
        llm=True,
        llm_slots=llm_slots,
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

    first = [augment, stream] if stream is not None else [augment]
    if convert is not None:
        convert >> first
    first >> evaluate >> formatting
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
            # Answerable and unanswerable rows are drawn alike, so [CONTEXT]
            # carries no signal about the label. Contexts are cached per source
            # report within the augmentation task.
            "contextualize_ratio": 0.3,
            # Fraction of created unanswerable rows keeping partial gold
            # evidence (table-text questions).
            "partial_ratio": 0.5,
            # A refusal now survives the filter only together with the
            # answerable row it was cloned from, which roughly halves the
            # yield: ~0.21 of the created refusals reach the final dataset.
            # 0.22 lands back on the ~11% refusal share of the `full` run.
            "unanswerable_ratio": 0.22,
            **LLM_PARAMS,
            # Single citation format of the SFT mix (as hotpotqa_fr_abstracts):
            # inline <ref id="N">excerpt</ref>, References list on REF_EOC_PROB.
            "citation_format": "ref",
            "gen_max_tokens": 2048,
            "gen_timeout": 300,
        },
        tags=[*BASE_TAGS, "tatqa", "en"],
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
        " --verify-concurrency {{ params.judge_concurrency }}"
        f" {MIX_OPTS} {SEED_OPT} {LIMIT_OPT}",
        llm=True,
        required_outputs=(f"{prefix}_converted.jsonl",),
    )

    augment = _bash(
        dag,
        "augment",
        f"{PYTHON} augment_tatqa.py"
        f" --dataset {prefix}_converted.jsonl"
        ' --source-documents "{{ params.input_json }}"'
        f" --output {prefix}_augmented"
        " --unanswerable-partial-ratio {{ params.partial_ratio }}"
        " --unanswerable-ratio {{ params.unanswerable_ratio }}"
        " --judge-concurrency {{ params.judge_concurrency }}"
        f" {CONTEXTUALIZE_OPT} {GEN_OPTS} {SEED_OPT} {LIMIT_OPT}",
        llm=True,
        required_outputs=(f"{prefix}_augmented.jsonl",),
    )

    evaluate = _bash(
        dag,
        "evaluate_and_filter",
        f"{PYTHON} evaluate_tatqa.py"
        f" --input {prefix}_augmented.jsonl"
        f" --output {prefix}_augmented_evaluated.jsonl"
        f" --eval-chunks --llm-judge-factual --llm-judge-refusal {JUDGE_OPTS} --no-answer-match --filter"
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
        tags=[*BASE_TAGS, "multilingual", language],
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


# Qwen3.6-35B-A3B (vLLM, gen_url) generates without thinking while the judges
# (Mistral, .env endpoint by default, or judge_ssh_host) score rows as they come.
hotpotqa_en = build_hotpotqa_dag(
    "dataset_rag_hotpotqa_en", "en", str(BASE_DIR / "hotpotqa_cache_train"),
    parallel_eval=True,
    param_overrides={
        "gen_model": "Qwen3.6-35B-A3B",
        "gen_max_tokens": 2048,
        "gen_timeout": 300,
        "gen_chat_template_kwargs": '{"enable_thinking": false}',
        "gen_concurrency": 16,
        "judge_model": "Mistral-Small-3.2-24B-Instruct-2506-FP8",
        "judge_concurrency": 6,
        "citation_format": "ref",
    },
)
# Le cache FR actuel est en .bak sur le disque ; ajustable via le param dataset_cache.
hotpotqa_fr = build_hotpotqa_dag(
    "dataset_rag_hotpotqa_fr", "fr", str(BASE_DIR / "hotpotqa_french_cache_train.bak")
)
# HotpotQA-fr natif (abstracts Wikipédia FR, sans traduction). Générateur
# Qwen3.8-27B sans raisonnement ; juges Mistral bridés à 6 appels concurrents ;
# citations au seul format <ref id="N">.
hotpotqa_fr_abstracts = build_hotpotqa_dag(
    "dataset_rag_hotpotqa_fr_abstracts", "fr", "",
    abstracts_convert=True,
    parallel_eval=True,
    param_overrides={
        "gen_model": "Qwen3.8-27B",
        "gen_max_tokens": 2048,
        "gen_timeout": 300,
        "gen_chat_template_kwargs": '{"enable_thinking": false}',
        "judge_model": "Mistral-Small-3.2-24B-Instruct-2506-FP8",
        "judge_concurrency": 6,
        "citation_format": "ref",
    },
)
tatqa = build_tatqa_dag()
frenchqa_piaf = build_multilingual_dag(
    "dataset_rag_frenchqa_piaf", "frenchqa-piaf", "fr"
)
