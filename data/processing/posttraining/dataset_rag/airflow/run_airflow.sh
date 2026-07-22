#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
AIRFLOW_ENV="${DATASET_RAG_AIRFLOW_VENV:-$HOME/venvs/airflow-rag}"
PIPELINE_PYTHON="${DATASET_RAG_PYTHON:-${DATASET_DIR}/venv/bin/python}"

if [[ ! -x "${AIRFLOW_ENV}/bin/airflow" ]]; then
    echo "Apache Airflow is not installed in ${AIRFLOW_ENV}" >&2
    exit 1
fi
if [[ ! -x "${PIPELINE_PYTHON}" ]]; then
    echo "Pipeline interpreter is not executable: ${PIPELINE_PYTHON}" >&2
    exit 1
fi

export AIRFLOW_HOME="${AIRFLOW_HOME:-${DATASET_DIR}/.airflow_home}"
export AIRFLOW__CORE__DAGS_FOLDER="${DATASET_DIR}/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export DATASET_RAG_DIR="${DATASET_DIR}"
export DATASET_RAG_PYTHON="${PIPELINE_PYTHON}"
# `airflow standalone` spawns its subcomponents (scheduler, api-server, dag
# processor, triggerer) via `airflow` looked up on PATH, so the venv bin must be
# discoverable there; execing the absolute path alone is not enough.
export PATH="${AIRFLOW_ENV}/bin:${PATH}"

exec "${AIRFLOW_ENV}/bin/airflow" "$@"
