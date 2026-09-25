#!/usr/bin/env bash
# Nuit : pool held-out complet (hors pilote) sur la politique SFT newrag, juge Qwen3.8-27B uniquement.
# Échantillonnage (GPU) et verdicts de contexte (juge) en parallèle, puis jugement des réponses et paires.
# Tout reprend là où ça s'est arrêté : relancer le script suffit.
#
#   cd $RAG_ROOT && setsid nohup bash dpo/run_heldout_full.sh > dpo/logs/heldout_full_sft4152.log 2>&1 < /dev/null &
set -uo pipefail
RAG_ROOT="${RAG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$RAG_ROOT"
source .venv/bin/activate
DIR="${DIR:-dpo/data/heldout_full_sft4152}"
POLICY="${POLICY:-models/sft_newrag_step4152}"
KEY="$RAG_ROOT/.judge_key"
(
    for attempt in 1 2; do
        python dpo/judge_pilot_contexts.py --prompts "$DIR/prompts.jsonl" --out "$DIR/context_verdicts.jsonl" \
            --judge-key-file "$KEY" --concurrency "${JUDGE_CONCURRENCY:-16}"
    done
) &
CTX_PID=$!
POLICY="$POLICY" DIR="$DIR" STAGES=sample bash dpo/run_pilot.sh
wait "$CTX_PID"
POLICY="$POLICY" DIR="$DIR" STAGES="score pairs" JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}" bash dpo/run_pilot.sh
