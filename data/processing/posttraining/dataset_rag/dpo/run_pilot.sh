#!/usr/bin/env bash
# Pilote DPO held-out : échantillonnage de la politique -> jugement des réponses -> paires.
# Les verdicts de contexte (judge_pilot_contexts.py) doivent déjà être dans $DIR.
# Chaque étape reprend là où elle s'est arrêtée : relancer le script suffit.
#
#   cd $RAG_ROOT && POLICY=models/sft_newrag_step4152 DIR=dpo/data/pilot_hpv4heldout_sft4152 \
#       setsid nohup bash dpo/run_pilot.sh > dpo/logs/pilot_sft4152.log 2>&1 < /dev/null &
set -uo pipefail
RAG_ROOT="${RAG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$RAG_ROOT"
source .venv/bin/activate

POLICY="$RAG_ROOT/${POLICY:?POLICY=models/<modèle>}"
DIR="${DIR:?DIR=dpo/data/<pilote>}"
STAGES="${STAGES:-sample score pairs}"
K="${K:-4}"
TEMP="${TEMP:-0.8}"
IMAGE="nvcr.io/nvidia/vllm:26.02-py3"
CONTAINER="${CONTAINER:-vllm_dpo_pilot}"
SERVED="luciole_policy_pilot"
PORT="${PORT:-8013}"
GPU_UTIL="${GPU_UTIL:-0.30}"
KEY="$RAG_ROOT/.judge_key"

log(){ echo "[pilot $(date '+%F %H:%M:%S')] $*"; }
PAUSER_PID=""
stop_container(){
    [ -n "$PAUSER_PID" ] && kill "$PAUSER_PID" 2>/dev/null
    docker unpause "$CONTAINER" >/dev/null 2>&1
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap stop_container EXIT
[ -s "$DIR/prompts.jsonl" ] || { log "!! absent: $DIR/prompts.jsonl"; exit 1; }
if [[ " $STAGES " == *" score "* || " $STAGES " == *" pairs "* ]]; then
    [ -s "$DIR/context_verdicts.jsonl" ] || { log "!! absent: $DIR/context_verdicts.jsonl"; exit 1; }
fi
[ -f "$POLICY/model.safetensors" ] || { log "!! politique absente: $POLICY"; exit 1; }

if [[ " $STAGES " == *" sample "* ]]; then
    sed -i 's/"TokenizersBackend"/"PreTrainedTokenizerFast"/' "$POLICY/tokenizer_config.json"
    stop_container
    docker run -d --name "$CONTAINER" --gpus all --ipc=host -p "$PORT:8000" -v "$POLICY:/model:ro" "$IMAGE" \
        vllm serve /model --served-model-name "$SERVED" --max-model-len 8192 \
        --gpu-memory-utilization "$GPU_UTIL" --chat-template /model/chat_template.jinja >/dev/null
    log "démarrage vLLM ($POLICY)"
    for _ in $(seq 1 120); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
        [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ] || { docker logs "$CONTAINER" 2>&1 | tail -30; exit 1; }
        sleep 5
    done
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { docker logs "$CONTAINER" 2>&1 | tail -30; exit 1; }
    LOG=dpo/logs/thermal_pauser.log bash docker_thermal_pauser.sh "$CONTAINER" 80 72 >/dev/null 2>&1 < /dev/null &
    PAUSER_PID=$!
    for attempt in 1 2 3; do
        python dpo/sample_policy.py --prompts "$DIR/prompts.jsonl" --out "$DIR/samples.jsonl" \
            --api-url "http://127.0.0.1:$PORT/v1" --model "$SERVED" --k "$K" --temperature "$TEMP" --concurrency 32
        n_err=$(python -c "
import json,sys
last={}
for l in open('$DIR/samples.jsonl'):
    d=json.loads(l); last[d['pid']]=d
print(sum(1 for d in last.values() if d['errors']))")
        log "tentative $attempt : $n_err prompts en erreur"
        [ "$n_err" = 0 ] && break
    done
    python -c "
import hashlib,json
h=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
json.dump({'model':'$POLICY','model_sha256':h('$POLICY/model.safetensors'),'host':open('/etc/hostname').read().strip(),
           'prompt_sha256':h('$DIR/prompts.jsonl'),'k_sampled':$K,'temperature':$TEMP,'top_p':0.95,'seed':42,
           'greedy':True,'max_tokens':1024},open('$DIR/sampling.manifest.json','w'),indent=2)"
    stop_container; PAUSER_PID=""
    log "vLLM arrêté"
fi

if [[ " $STAGES " == *" score "* ]]; then
    log "jugement des réponses"
    for attempt in 1 2; do
        python dpo/score_pilot_samples.py --prompts "$DIR/prompts.jsonl" --samples "$DIR/samples.jsonl" \
            --contexts "$DIR/context_verdicts.jsonl" --out "$DIR/scored.jsonl" --judge-key-file "$KEY" \
            --concurrency "${JUDGE_CONCURRENCY:-24}"
    done
fi

if [[ " $STAGES " == *" pairs "* ]]; then
    python dpo/build_pilot_pairs.py --prompts "$DIR/prompts.jsonl" --contexts "$DIR/context_verdicts.jsonl" \
        --scored "$DIR/scored.jsonl" --out-dir "$DIR/pairs" --model "$(basename "$(readlink -f "$POLICY")")"
fi
log "TERMINÉ"
