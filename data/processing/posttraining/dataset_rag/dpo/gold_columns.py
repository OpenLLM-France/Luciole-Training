#!/usr/bin/env python3
"""gold_answer / gold_scale columns of OpenLLM-France/RAG_DPO_onpolicy (commit 27e169e, every config): the original
answer looked up by qid in the HotpotQA and TAT-QA train sets (an unanswerable prompt keeps the answer of the original
question). TAT-QA: multi-span answers joined with " | ", and the scale; HotpotQA: no scale (None). Placed after qid.
Prompt set of an SFT run (e.g. hotpotqa_fr, delta run given): the answer of the converted or evaluated rows of that SFT run, no
scale."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("gold_answer", "gold_scale")


def rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def lookup(ds, run=None):
    """qid -> (gold_answer, gold_scale) for dataset ds (run: delta run of a prompt set built from an SFT run)."""
    if ds not in ("hotpotqa", "tatqa"):
        if run is None:
            raise ValueError(f"{ds} : colonnes gold depuis le run SFT des prompts, run delta requis")
        sft = {p["sft_run"] for p in rows(Path(run) / "qwen35_35b_a3b" / ds / "prompts.jsonl")}
        if len(sft) != 1:
            raise ValueError(f"{ds} : un seul run SFT attendu, {sorted(map(str, sft))}")
        source = ROOT / sft.pop()
        converted = source / f"{ds}_converted.jsonl"
        if converted.is_file():
            answer_file = converted
        else:
            # EN and translated FR DAGs read a dataset cache directly. Their
            # evaluated rows retain the original answer (including for refusals).
            evaluated = sorted(source.glob("*_augmented_evaluated.jsonl"))
            if len(evaluated) != 1:
                raise ValueError(f"{source}: un seul *_augmented_evaluated.jsonl attendu, trouvé {len(evaluated)}")
            answer_file = evaluated[0]
        answers = {r["id"]: r["answer"] for r in rows(answer_file)}
        return lambda qid: (answers[qid], None)
    if ds == "tatqa":
        tq = {q["uid"]: q for doc in json.load(open(ROOT / "tatqa_dataset_train.json", encoding="utf-8"))
              for q in doc["questions"]}

        def tatqa(qid):
            q = tq[qid.split("_", 1)[1]]
            ans = q["answer"]
            return (" | ".join(map(str, ans)) if isinstance(ans, list) else str(ans)), q.get("scale", "")
        return tatqa
    from datasets import load_from_disk
    hp = load_from_disk(str(ROOT / "hotpotqa_cache_train"))
    answers = dict(zip(hp["id"], hp["answer"]))
    return lambda qid: (answers[qid], None)


def add_to_parquet(path, ds, run=None):
    """Rewrite the parquet file at path with the gold columns (string type, even when all null) after qid: the
    bytes of the 27e169e files when applied to the files of the previous release."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    f = lookup(ds, run)
    gold = list(zip(*(f(q) for q in t.column("qid").to_pylist()))) or [(), ()]
    i = t.column_names.index("qid") + 1
    for j, (name, values) in enumerate(zip(COLUMNS, gold)):
        t = t.add_column(i + j, pa.field(name, pa.string()), pa.array(list(values), pa.string()))
    pq.write_table(t, path)
