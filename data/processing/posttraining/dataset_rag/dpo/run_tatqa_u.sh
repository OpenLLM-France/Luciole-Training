#!/usr/bin/env bash
# Pool held-out TatQA, 2e passe : clones non répondables (sans gold : build_nogold_prompts_tatqa ;
# + clones « answered » du run v4) et nouvel essai des verdicts de contexte illisibles.
#   1) verdicts illisibles retirés puis rejugés (jusqu'à 3 essais), clones génériques ajoutés à $G ;
#   2) échantillonnage des clones ($U, génériques + stricts) en parallèle des verdicts de contexte ;
#   3) prompts stricts reconstruits (bases A + clones), échantillons répartis, juge et paires
#      (run_two_pass.sh sur $G, run_strict.sh sur $S). Les paires précédentes sont gardées dans pairs_v0/.
# Reprise : relancer suffit.
#   cd $RAG_ROOT && setsid nohup bash dpo/run_tatqa_u.sh > dpo/logs/tatqa_u.log 2>&1 < /dev/null &
set -uo pipefail
RAG_ROOT="${RAG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$RAG_ROOT"
source .venv/bin/activate
ALL=dpo/data/heldout_tatqa_sft4152
U=dpo/data/heldout_tatqa_u_sft4152
G=dpo/data/heldout_tatqa_generic_sft4152
S=dpo/data/heldout_tatqa_strict_sft4152
KEY="$RAG_ROOT/.judge_key"
C="${JUDGE_CONCURRENCY:-16}"
log(){ echo "[tatqa-u $(date '+%F %H:%M:%S')] $*"; }
drop_errors(){  # judge_pilot_contexts resumes by pid: unparsable verdicts must be removed to be retried
    python - "$1" <<'EOF'
import json, sys
path = sys.argv[1]
lines = open(path, encoding="utf-8").readlines()
keep = [l for l in lines if "error" not in json.loads(l)]
open(path, "w", encoding="utf-8").writelines(keep)
print(f"[tatqa-u] {len(lines) - len(keep)} verdicts illisibles retirés", flush=True)
EOF
}

[ -s "$U/prompts.jsonl" ] || { log "!! absent: $U/prompts.jsonl"; exit 1; }
[ -f "$G/context_verdicts.before_u.jsonl" ] || cp "$G/context_verdicts.jsonl" "$G/context_verdicts.before_u.jsonl"
[ -f "$G/prompts.before_u.jsonl" ] || cp "$G/prompts.jsonl" "$G/prompts.before_u.jsonl"
python dpo/prep_tatqa_heldout_dirs.py merge-u --u $U --generic $G || exit 1

(
    if [ ! -s "$U/sampling.manifest.json" ]; then
        POLICY=models/sft_newrag_step4152 DIR=$U STAGES=sample CONTAINER=vllm_tatqa_u bash dpo/run_pilot.sh
    fi
) > dpo/logs/tatqa_u_sample.log 2>&1 &
SAMPLE_PID=$!

log "verdicts de contexte ($(wc -l < $G/prompts.jsonl) prompts génériques)"
for attempt in 1 2 3; do
    drop_errors $G/context_verdicts.jsonl
    python dpo/judge_pilot_contexts.py --prompts $G/prompts.jsonl --out $G/context_verdicts.jsonl \
        --judge-key-file "$KEY" --concurrency "$C"
done
python - <<EOF
import json, collections
P = {}
for l in open("$G/prompts.jsonl"):
    p = json.loads(l); P[p["pid"]] = p
last = {}
for l in open("$G/context_verdicts.jsonl"):
    r = json.loads(l); last[r["pid"]] = r
c = collections.Counter((P[k].get("clone_source") or ("U" if P[k]["is_unanswerable"] else "A"), r["verdict"]) for k, r in last.items())
print("[tatqa-u] verdicts:", {f"{a}/{b}": n for (a, b), n in sorted(c.items())}, "| illisibles:", sum("error" in r for r in last.values()), flush=True)
EOF

wait "$SAMPLE_PID"
[ -s "$U/sampling.manifest.json" ] || { log "!! échantillonnage des clones incomplet (voir dpo/logs/tatqa_u_sample.log)"; exit 1; }
log "échantillonnage des clones terminé"
for d in $G $S; do
    [ -d "$d/pairs" ] && [ ! -d "$d/pairs_v0" ] && cp -r "$d/pairs" "$d/pairs_v0"
done
cat $ALL/prompts_judge.jsonl $U/prompts.jsonl > $U/pool_with_bases.jsonl
python dpo/build_strict_prompts.py --pool $U/pool_with_bases.jsonl --base-dir $G --out-dir $S || exit 1
python dpo/prep_tatqa_heldout_dirs.py samples --all $ALL $U --dirs $G $S || exit 1

log "=== générique"
JUDGE_CONCURRENCY="$C" bash dpo/run_two_pass.sh $G
log "=== strict"
D=$S TWIN=$G JUDGE_CONCURRENCY="$C" bash dpo/run_strict.sh
log "TERMINÉ"
