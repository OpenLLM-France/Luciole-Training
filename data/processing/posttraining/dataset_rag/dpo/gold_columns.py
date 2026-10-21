#!/usr/bin/env python3
"""gold_answer / gold_scale columns of OpenLLM-France/RAG_DPO_onpolicy (commit 27e169e, every config): the original
answer looked up by qid in the HotpotQA and TAT-QA train sets (an unanswerable prompt keeps the answer of the original
question). TAT-QA: multi-span answers joined with " | ", and the scale; HotpotQA: no scale (None). Placed after qid."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("gold_answer", "gold_scale")


def lookup(ds):
    """qid -> (gold_answer, gold_scale) for dataset ds."""
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


def add_to_parquet(path, ds):
    """Rewrite the parquet file at path with the gold columns (string type, even when all null) after qid: the
    bytes of the 27e169e files when applied to the files of the previous release."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    f = lookup(ds)
    gold = list(zip(*(f(q) for q in t.column("qid").to_pylist()))) or [(), ()]
    i = t.column_names.index("qid") + 1
    for j, (name, values) in enumerate(zip(COLUMNS, gold)):
        t = t.add_column(i + j, pa.field(name, pa.string()), pa.array(list(values), pa.string()))
    pq.write_table(t, path)
