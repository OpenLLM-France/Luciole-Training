#!/usr/bin/env bash
# Paires à consigne stricte (phrase de refus imposée) pour heldout_strict_sft4152, sur un serveur GPU :
# attend l'échantillonnage (run_pilot.sh STAGES=sample), puis passe 1 sans raisonnement sur les
# échantillons utiles, confirmation des chosen répondables avec raisonnement (par tours), paires.
# Juge Qwen3.8-27B à 8 appels concurrents (partage de l'endpoint avec le pilote sans gold).
# D = dossier des prompts stricts (défaut : la run à 5 phrases v4).
#   cd $RAG_ROOT && setsid nohup bash dpo/run_strict.sh > dpo/logs/strict.log 2>&1 < /dev/null &
set -uo pipefail
RAG_ROOT="${RAG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$RAG_ROOT"
source .venv/bin/activate
D="${D:-dpo/data/heldout_strict_sft4152}"
KEY="$RAG_ROOT/.judge_key"
C="${JUDGE_CONCURRENCY:-8}"
log(){ echo "[strict $(date '+%F %H:%M:%S')] $*"; }
while pgrep -u "$USER" -f "[r]un_pilot.sh" >/dev/null && pgrep -u "$USER" -f "sample_policy.py --prompts $D" >/dev/null || \
      [ ! -s "$D/sampling.manifest.json" ]; do sleep 60; done
log "échantillons : $(wc -l < $D/samples.jsonl) prompts"
python dpo/build_strict_pairs.py select --dir $D --out $D/pass1_only.jsonl
log "passe 1 sans raisonnement"
python dpo/score_pilot_samples.py --prompts $D/prompts.jsonl --samples $D/samples.jsonl \
    --contexts $D/context_verdicts.jsonl --only $D/pass1_only.jsonl --out $D/scored.jsonl \
    --judge-key-file "$KEY" --no-thinking --max-tokens 800 --timeout 300 --concurrency "$C"
for r in 1 2 3 4; do
    python dpo/build_strict_pairs.py verify --dir $D --out $D/verify_round$r.jsonl
    [ -s $D/verify_round$r.jsonl ] || break
    log "tour $r : $(wc -l < $D/verify_round$r.jsonl) chosen à confirmer (avec raisonnement)"
    python dpo/score_pilot_samples.py --prompts $D/prompts.jsonl --samples $D/samples.jsonl \
        --contexts $D/context_verdicts.jsonl --only $D/verify_round$r.jsonl --out $D/verified.jsonl \
        --judge-key-file "$KEY" --max-tokens 6000 --timeout 900 --concurrency "$C"
done
python dpo/build_strict_pairs.py build --dir $D --model sft_luciole_1B_instruct_newrag-step_4152 --twin-dir "${TWIN:-dpo/data/heldout_full_sft4152}"
log "TERMINÉ"
