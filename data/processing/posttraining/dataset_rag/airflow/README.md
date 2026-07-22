# Airflow: SFT data pipeline

The DAGs in `dags/dataset_rag_dags.py` call the CLI scripts of `dataset_rag/` unchanged; Airflow only handles
orchestration, retries and logs.

| DAG | Steps |
|---|---|
| `dataset_rag_hotpotqa_en` | augment → evaluate/filter → formatting_sft |
| `dataset_rag_hotpotqa_fr` | augment (FR) → evaluate/filter (+ French quality judge) → formatting_sft |
| `dataset_rag_tatqa` | convert (distractors) → augment → evaluate/filter → formatting_sft |
| `dataset_rag_frenchqa_piaf` | convert → augment → evaluate/filter → formatting_sft (`--mixed-system-prompt`) |

Outputs go to `airflow_runs/<dag_id>/<run_name>/` under `dataset_rag/`.

## Installation

Keep Airflow in its own environment; tasks run the pipeline interpreter given by `DATASET_RAG_PYTHON` (default:
`venv/bin/python` of `dataset_rag/`), so Airflow never touches the ML dependencies:

```bash
python3 -m venv ~/venvs/airflow-rag
source ~/venvs/airflow-rag/bin/activate
AIRFLOW_VERSION=3.1.5
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
pip install "apache-airflow==${AIRFLOW_VERSION}" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
```

`run_airflow.sh` sets `AIRFLOW_HOME`, the DAG folder and the pipeline interpreter (`DATASET_RAG_AIRFLOW_VENV`
overrides the Airflow venv). The DAG files also work with Airflow 2.x.

```bash
./run_airflow.sh db migrate
./run_airflow.sh pools set llm_api 1 "Serialises the tasks that call the LLM API"
./run_airflow.sh standalone
```

The `llm_api` pool is **mandatory** (LLM tasks wait for it): with one slot, the tasks that call the LLM endpoint
run one at a time.

The LLM key is read by the scripts from `.env` (`LLM_API_KEY`). Optional environment variables, set before starting
the scheduler: `DATASET_RAG_DIR` (`dataset_rag/` directory, needed when the DAG files are deployed apart from the code),
`DATASET_RAG_PYTHON`, `DATASET_RAG_LLM_POOL`.

Run long jobs on a host that does not sleep: a suspended laptop loses the heartbeat and Airflow kills the task.

## Triggering a pipeline

```bash
# smoke test on 100 rows
./run_airflow.sh dags trigger dataset_rag_tatqa --conf '{"run_name": "smoke", "limit": 100}'
# full run
./run_airflow.sh dags trigger dataset_rag_hotpotqa_en --conf '{"run_name": "v1"}'
```

Common parameters:

- `run_name` (default `default`): output sub-folder. **Re-triggering with the same `run_name` resumes the augmentation
  from its checkpoint** (no LLM call is paid twice); use a new name to start from scratch.
- `limit`: row cap for tests (null = full dataset). For TAT-QA it applies at conversion time, before `--verify-hard`.
- `seed` (default 42), `sft_format` (`chat` or `prompt_completion`).

Dataset parameters: `dataset_cache` (HotpotQA), `input_json`, `num_distractors`, `distractor_start_rank`
(TAT-QA / multilingual).

Distractors and hard refusals (enabled by default):

- `distractor_mix` (default `hard=2,medium=5,random=2`): mixed-difficulty selection; empty = plain BM25 from
  `distractor_start_rank`.
- `verify_hard` (default `true`): an LLM checks that the hard candidates do not answer the question (one call per row,
  `llm_api` pool).
- `evidence_query` (default `true`): RRF fusion with a ranking by similarity to the gold passage.
- `dense_model` (default empty): sentence-transformers model for hybrid dense + BM25 ranking
  (e.g. `intfloat/multilingual-e5-base`).
- `partial_hop_ratio` (HotpotQA) / `partial_ratio` (TAT-QA), default 0.5: share of the created refusals that keep
  partial gold evidence (one hop out of two / one of the table-text modalities).

## Answerable / refusal pairs

A refusal is a **clone** of an answerable row whose gold evidence was removed: same question, same distractors. The
two rows form a minimal pair whose only difference is what the refusal should depend on. The answerable source row is
kept.

**Padding.** Removing gold chunks without replacement shortens the context, and length alone then predicts the refusal
(TAT-QA: AUC 0.63, median 8 chunks vs 10). Each removed chunk is therefore replaced by a distractor of the same kind
(a table for a table) from another report, never from the question's report, and discarded if it contains the answer.
Replacements are protected from later truncation, like the gold evidence kept by `partial` / `partial_hop` refusals.
After the fix the AUC is 0.50 on both datasets. `--no-unanswerable-backfill` restores the old behaviour.

## Retries and resumption

- Each task has 2 retries (5 min delay).
- Augmentation scripts write `<prefix>_checkpoint.json`: a retry or a re-run resumes where it stopped. Do not resume a
  `run_name` created by an older version of the code: the refusal draw may differ.
- `evaluate_and_filter.py` always runs with `--llm-judge-factual`: the filter requires a judge score of 5, without
  it only unanswerable rows would pass.

## Adding a multilingual dataset

```python
newsquadfr = build_multilingual_dag("dataset_rag_newsquadfr", "newsquadfr", "fr")
```

The `(dataset, language)` pair must exist in `DATASET_CONFIGS` of `convert_multilingual_qa.py`.
