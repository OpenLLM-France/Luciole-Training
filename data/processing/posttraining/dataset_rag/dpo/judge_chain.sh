#!/usr/bin/env bash
# Jugement du run delta, contre un juge Qwen3.8-27B OpenAI-compatible ($URL, clé dans $KEY ; le DAG
# dataset_rag_dpo_pairs fournit RAG_ROOT, R, URL, KEY, JUDGE_MODEL, CONC et SKIP_CALIB).
#   0. calibration (optionnelle, SKIP_CALIB=0) : les réponses de $R/calib/ déjà jugées avec raisonnement sont
#      rejugées ici ; arrêt si l'accord answer_correct < CALIB_MIN (0.9) (FORCE=1 pour passer outre)
#   1. chosen 35B : qwen_chosen_all (passe 1 sans raisonnement -> confirmation avec
#      raisonnement, 3 tours ; refus : judge_refusal_candidates ; greedy puis échantillons ; --prefer-quoted)
#   2. rejected 4B : TOUS les échantillons utilisables, avec raisonnement (jz_todo_rejected.py)
#   1 et 2 tournent en parallèle (3/4 de CONC chacun par défaut) pour garder le serveur plein.
# Reprise : relancer suffit (tous les scripts sautent ce qui est déjà jugé).
set -uo pipefail
cd "${RAG_ROOT:-$(dirname "$0")/..}"
URL="${URL:-http://127.0.0.1:8000/v1}"
KEY="${KEY:-dpo/.dummy_key}"
CONC="${CONC:-384}"
R="${R:-dpo/data/jz_run}"
PY="${DPO_PYTHON:-python}"
LOG_DIR="${LOG_DIR:-dpo/logs}"
mkdir -p "$LOG_DIR"
log(){ echo "[judge $(date '+%F %H:%M:%S')] $*"; }
[ -s "$KEY" ] || echo "none" > "$KEY"
# concurrence par côté (35B / 4B, qui tournent en parallèle) : 3/4 de CONC chacun ; quand un côté a
# fini, l'autre garde le serveur presque plein (run du 2026-10-06 : le 4B seul à CONC/2 laissait la moitié du serveur
# vide). La file vLLM absorbe le surplus tant que les deux tournent (timeout 1 800 s).
HALF="${SIDE_CONC:-$(( CONC * 3 / 4 ))}"; [ $HALF -ge 1 ] || HALF=1
# jugements avec raisonnement : 6 000 tokens laissaient ~1/4 des jugements de refus sans verdict (raisonnement coupé,
# finish_reason=length, test JZ du 2026-10-06) -> 16 000 (max_model_len du serveur : 24 576)
MT="${JUDGE_MAX_TOKENS:-16000}"

JM="${JUDGE_MODEL:-Qwen3.8-27B}"   # nom servi par le juge (endpoint Lucie, vLLM --served-model-name)
SP(){ "$PY" dpo/score_pilot_samples.py --judge-url "$URL" --judge-key-file "$KEY" --judge-model "$JM" "$@"; }
RJ(){ "$PY" dpo/judge_refusal_candidates.py --judge-url "$URL" --judge-key-file "$KEY" --judge-model "$JM" --max-tokens "$MT" --timeout 1800 "$@"; }
QC(){ "$PY" dpo/qwen_chosen_all.py --gen qwen35_35b_a3b --prefer-quoted "$@"; }

if [ "${SKIP_CALIB:-1}" != 1 ]; then
    for ds in hotpotqa tatqa; do
        C=$R/calib/$ds
        SP --prompts $C/prompts.jsonl --samples $C/chosen_qwen35_35b_a3b.jsonl --contexts $C/context_verdicts.jsonl \
           --only $C/only.jsonl --out $C/verified_jz.jsonl --max-tokens $MT --timeout 1800 --concurrency "$CONC"
    done
    "$PY" dpo/calib_compare.py --dir $R/calib/hotpotqa --dir $R/calib/tatqa --min "${CALIB_MIN:-0.9}" \
        | tee $R/calib/agreement.json
    [ "${PIPESTATUS[0]}" = 0 ] || [ "${FORCE:-0}" = 1 ] || { log "!! calibration en échec : arrêt (FORCE=1 pour continuer)"; exit 1; }
    log "calibration OK"
fi

chosen_35b(){
    local GEN=qwen35_35b_a3b D
    for ds in hotpotqa tatqa; do
        D=$R/$GEN/$ds
        [ -s $D/chosen_$GEN.jsonl ] || { log "!! pas de génération dans $D"; return 1; }
        for KS in -1 0,1; do
            QC todo --dir $D --stage pass1 --ks="$KS" --out $D/todo_pass1.jsonl || return 1
            if [ -s $D/todo_pass1.jsonl ]; then  # liste vide = rien à juger (ne pas échouer)
                SP --prompts $D/prompts.jsonl --samples $D/chosen_$GEN.jsonl \
                    --contexts $D/context_verdicts.jsonl --only $D/todo_pass1.jsonl --out $D/scored.jsonl \
                    --no-thinking --max-tokens 800 --timeout 300 --concurrency $HALF || return 1
            fi
            for r in 1 2 3; do
                QC todo --dir $D --stage verify --out $D/todo_verify.jsonl || return 1
                [ -s $D/todo_verify.jsonl ] || break
                SP --prompts $D/prompts.jsonl --samples $D/chosen_$GEN.jsonl --contexts $D/context_verdicts.jsonl \
                   --only $D/todo_verify.jsonl --out $D/verified.jsonl --max-tokens $MT --timeout 1800 --concurrency $HALF || return 1
            done
            QC todo --dir $D --stage refusal --ks="$KS" --out $D/todo_refusal.jsonl || return 1
            if [ -s $D/todo_refusal.jsonl ]; then
                cut -d'"' -f4 $D/todo_refusal.jsonl | sort -u > $D/todo_refusal.pids
                RJ --prompts $D/prompts.jsonl --samples $D/chosen_$GEN.jsonl --contexts $D/context_verdicts.jsonl \
                   --out $D/judged_chosen_$GEN.jsonl --pids $D/todo_refusal.pids --ks="$KS" --concurrency $HALF || return 1
            fi
            log "35B $ds : tour ks=$KS terminé"
        done
    done
    log "35B : TERMINÉ"
}

rejected_4b(){
    local GEN=qwen35_4b D
    for ds in hotpotqa tatqa; do
        D=$R/$GEN/$ds
        [ -s $D/chosen_$GEN.jsonl ] || { log "!! pas de génération dans $D"; return 1; }
        "$PY" dpo/jz_todo_rejected.py --dir $D --gen $GEN || return 1
        if [ -s $D/todo_rej_verify.jsonl ]; then  # liste vide = rien à juger (ne pas échouer)
            SP --prompts $D/prompts.jsonl --samples $D/chosen_$GEN.jsonl \
                --contexts $D/context_verdicts.jsonl --only $D/todo_rej_verify.jsonl --out $D/verified.jsonl \
                --max-tokens $MT --timeout 1800 --concurrency $HALF || return 1
        fi
        if [ -s $D/todo_rej_citfail.jsonl ]; then  # liste vide = rien à juger (ne pas échouer)
            SP --prompts $D/prompts.jsonl --samples $D/chosen_$GEN.jsonl \
                --contexts $D/context_verdicts.jsonl --only $D/todo_rej_citfail.jsonl --out $D/scored_citfail.jsonl \
                --max-tokens $MT --timeout 1800 --concurrency $HALF || return 1
        fi
        if [ -s $D/todo_rej_refusal.pids ]; then  # liste vide = rien à juger (ne pas échouer)
            RJ --prompts $D/prompts.jsonl --samples $D/chosen_$GEN.jsonl \
                --contexts $D/context_verdicts.jsonl --out $D/judged_chosen_$GEN.jsonl --pids $D/todo_rej_refusal.pids \
                --ks=-1,0,1,2,3 --concurrency $HALF || return 1
        fi
        "$PY" dpo/jz_todo_rejected.py --dir $D --gen $GEN || return 1  # reste à juger
        log "4B $ds : terminé"
    done
    log "4B : TERMINÉ"
}

TAG=$(basename "$R")
chosen_35b > "$LOG_DIR/${TAG}_judge_35b.log" 2>&1 &
P35=$!
rejected_4b > "$LOG_DIR/${TAG}_judge_4b.log" 2>&1 &
P4=$!
wait $P35; S35=$?
wait $P4; S4=$?
tail -n 3 "$LOG_DIR/${TAG}_judge_35b.log" "$LOG_DIR/${TAG}_judge_4b.log"
[ $S35 = 0 ] && [ $S4 = 0 ] || { log "!! échec (35B $S35, 4B $S4)"; exit 1; }
log "TERMINÉ"
