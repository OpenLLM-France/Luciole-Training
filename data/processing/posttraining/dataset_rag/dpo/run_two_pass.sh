#!/usr/bin/env bash
# Jugement en deux passes (Qwen3.8-27B) puis paires, pour chaque DIR donné en argument :
#   passe 1 : toutes les réponses SANS raisonnement (rapide, indulgent : ses « mauvais » sont fiables) -> scored.jsonl
#   passe 2 : AVEC raisonnement, seulement les chosen candidats, par tours (greedy d'abord) jusqu'à
#             confirmation d'un chosen par prompt -> verified.jsonl
#   paires  : build_pilot_pairs.py --verified -> pairs/
# Les verdicts de contexte (context_verdicts.jsonl) doivent exister. Reprise automatique.
# Concurrence totale : 16 (saturation mesurée de l'endpoint), les DIR passent l'un après l'autre.
#
#   cd $RAG_ROOT && setsid nohup bash dpo/run_two_pass.sh dpo/data/A dpo/data/B > dpo/logs/two_pass.log 2>&1 < /dev/null &
set -uo pipefail
RAG_ROOT="${RAG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$RAG_ROOT"
source .venv/bin/activate
KEY="$RAG_ROOT/.judge_key"
C="${JUDGE_CONCURRENCY:-16}"
ROUNDS="${ROUNDS:-3}"
log(){ echo "[two-pass $(date '+%F %H:%M:%S')] $*"; }
for DIR in "$@"; do
    log "=== $DIR"
    for f in prompts.jsonl samples.jsonl context_verdicts.jsonl; do
        [ -s "$DIR/$f" ] || { log "!! absent: $DIR/$f"; continue 2; }
    done
    log "passe 1 (sans raisonnement)"
    python dpo/score_pilot_samples.py --prompts "$DIR/prompts.jsonl" --samples "$DIR/samples.jsonl" \
        --contexts "$DIR/context_verdicts.jsonl" --out "$DIR/scored.jsonl" --judge-key-file "$KEY" \
        --no-thinking --max-tokens 800 --timeout 300 --concurrency "$C"
    for r in $(seq 1 "$ROUNDS"); do
        python dpo/select_chosen_candidates.py --prompts "$DIR/prompts.jsonl" --contexts "$DIR/context_verdicts.jsonl" \
            --scored "$DIR/scored.jsonl" --verified "$DIR/verified.jsonl" --out "$DIR/verify_round$r.jsonl"
        [ -s "$DIR/verify_round$r.jsonl" ] || break
        log "passe 2, tour $r : $(wc -l < "$DIR/verify_round$r.jsonl") chosen à vérifier (avec raisonnement)"
        python dpo/score_pilot_samples.py --prompts "$DIR/prompts.jsonl" --samples "$DIR/samples.jsonl" \
            --contexts "$DIR/context_verdicts.jsonl" --only "$DIR/verify_round$r.jsonl" --out "$DIR/verified.jsonl" \
            --judge-key-file "$KEY" --max-tokens 6000 --timeout 900 --concurrency "$C"
    done
    python dpo/build_pilot_pairs.py --prompts "$DIR/prompts.jsonl" --contexts "$DIR/context_verdicts.jsonl" \
        --scored "$DIR/scored.jsonl" --verified "$DIR/verified.jsonl" --out-dir "$DIR/pairs" \
        --model sft_luciole_1B_instruct_newrag-step_4152
done
log "TERMINÉ"
