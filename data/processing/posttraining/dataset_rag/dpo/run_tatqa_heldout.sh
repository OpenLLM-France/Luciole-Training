#!/usr/bin/env bash
# Pool held-out TatQA sur la SFT newrag step 4152 : juge Qwen3.8-27B puis paires, avec la même
# chaîne que HotpotQA (générique : run_two_pass.sh ; strict : run_strict.sh --twin-dir générique).
# L'échantillonnage (run_pilot.sh STAGES=sample) tourne sur tous les prompts dans $ALL.
# Les verdicts de contexte démarrent tout de suite, pendant l'échantillonnage.
# Reprise : relancer suffit.
#   cd $RAG_ROOT && setsid nohup bash dpo/run_tatqa_heldout.sh > dpo/logs/tatqa_heldout.log 2>&1 < /dev/null &
set -uo pipefail
RAG_ROOT="${RAG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$RAG_ROOT"
source .venv/bin/activate
ALL=dpo/data/heldout_tatqa_sft4152
G=dpo/data/heldout_tatqa_generic_sft4152
S=dpo/data/heldout_tatqa_strict_sft4152
KEY="$RAG_ROOT/.judge_key"
C="${JUDGE_CONCURRENCY:-16}"
log(){ echo "[tatqa $(date '+%F %H:%M:%S')] $*"; }

[ -s "$G/prompts.jsonl" ] || python dpo/prep_tatqa_heldout_dirs.py prompts --all $ALL --generic $G || exit 1
log "verdicts de contexte ($(wc -l < $G/prompts.jsonl) prompts génériques)"
for attempt in 1 2; do
    python dpo/judge_pilot_contexts.py --prompts $G/prompts.jsonl --out $G/context_verdicts.jsonl \
        --judge-key-file "$KEY" --concurrency "$C"
done
python - <<EOF
import json, collections
last = {}
for l in open("$G/context_verdicts.jsonl"):
    r = json.loads(l); last[r["pid"]] = r
print("[tatqa] verdicts:", dict(collections.Counter(r["verdict"] for r in last.values())), flush=True)
EOF

while [ ! -s "$ALL/sampling.manifest.json" ]; do sleep 60; done
log "échantillonnage terminé"
python dpo/build_strict_prompts.py --pool $ALL/prompts_judge.jsonl --base-dir $G --out-dir $S || exit 1
python dpo/prep_tatqa_heldout_dirs.py samples --all $ALL --dirs $G $S || exit 1

log "=== générique"
JUDGE_CONCURRENCY="$C" bash dpo/run_two_pass.sh $G
log "=== strict"
D=$S TWIN=$G JUDGE_CONCURRENCY="$C" bash dpo/run_strict.sh
log "TERMINÉ"
