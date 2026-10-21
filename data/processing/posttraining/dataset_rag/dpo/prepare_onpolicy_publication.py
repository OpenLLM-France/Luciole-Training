#!/usr/bin/env python3
"""Publication folder of the on-policy DPO pairs of the SFT newrag step 4152 policy (nothing is pushed).

Two configs, one per source dataset (decision 2026-10-02); the refusal instruction is a field, not a config:
  hotpotqa   heldout_full_sft4152/pairs_v1.1 (generic, audit v1.1)
             + heldout_strictpool_sft4152/pairs_audited (strict: 30 phrases, 7 wordings)
  tatqa      heldout_tatqa_generic_sft4152/pairs_audited + heldout_tatqa_strict_sft4152/pairs_audited
Not published: hotpotqa v1 and v1.1_bal (v1.1_bal = filter on rejected_states_twin_gold), the strict run with the
5 phrases of hotpotqa_v4 (same questions as the strict pool, phrases seen in SFT).

Chosen AND rejected are sampled from the policy (strict U pairs: chosen = the policy writing the imposed phrase;
chosen_source policy_generic_twin = written on the generic twin of the strict prompt). Added fields: dataset,
refusal_instruction (generic | strict), audit (v1.1 | recheck), and for TatQA answer_type / answer_from. Pairs longer
than --max-tokens (policy tokenizer + chat template, longest side) are dropped. Checks in manifest.json: splits
disjoint by qid, lighteval refusal phrases (strict_phrases.HELD_OUT) absent, phrase chosen >= rejected per dataset,
leading whitespace symmetric, TatQA uids absent from test (lighteval source) and dev, HotpotQA ids absent from
validation, every config loadable with `datasets`. Written as Parquet, with one schema inferred from all the rows
of a config (JSON Lines cannot be read back: `datasets` infers the column types from the first block, where the
strict-only fields are still null).

Corrections of the 2026-10-02 audit (v2, all on by default; the first publication, commit a0b1519, is
`--generic-rule invite --no-quote-filter --no-split-wrong-reasoning --tatqa-split qid --no-phrase-balance
--decision-ratio 0 --no-drop-redundant`):
  --generic-rule plain    the generic refusal rule loses its "invite the user to rephrase ..." clause, which
                          96-97 % of the chosen refusals ignore (prompt edited after sampling).
  --quote-filter          drop pairs whose chosen has a quote that is not in the documents, compared on lowercase
                          alphanumerics, line by line and around ellipses (reconstructed table rows pass).
  --split-wrong-reasoning A_wrong_reasoning becomes A_wrong_answer (rejected final answer wrong) and
                          A_unsupported_reasoning (rejected answer correct and supported, reasoning unsupported),
                          from the reasoning recheck of the rejected response when there is one, else from pass 1;
                          any other verdict (answer correct but unsupported, citation-only flaw, refusal) is dropped.
  --tatqa-split table     TatQA val = ~10 % of the gold tables (hash of the table uid, seed 42); the qid split put
                          the table of 46/113 val pairs in train.
  --decision-ratio R      in each cell (config, instruction, hotpot_type | TatQA arithmetic/table/other), the
                          majority direction of the decision pairs (A_false_refusal pushes to answer,
                          U_unsupported_answer to refuse) is capped at R x the minority; otherwise the instruction
                          predicts the direction (generic ~2:1 refuse, strict ~3:1 answer). 0 = off.
  --phrase-balance        per config AND per imposed phrase, the bare phrase is rejected at most as often as it is
                          chosen (the 5 v4 phrases were 60 chosen / 109 rejected).
  --drop-redundant        the published files leave out the constant or derivable columns of REDUNDANT
                          (dataset = config, split = file, model / rejected_source constant, audit, chosen_verified,
                          rejected_is_greedy = rejected_k == -1, lengths); the policy is named in manifest.json.
Decision and phrase caps are applied until neither drops a pair. A capped pair is dropped from the end of an
order that keeps greedy rejected responses (the policy's default behaviour) first, then sha256(pid, category).

    venv/bin/python dpo/prepare_onpolicy_publication.py --out dpo/data/publish_onpolicy_sft4152_v2 \
        --tokenizer dpo/data/publish_onpolicy_sft4152/tokenizer

DAG dataset_rag_dpo_pairs (mode onpolicy): --sources replaces CONFIGS by its own pools, --gold-columns adds the columns
of hub commit 27e169e (gold_columns.py; the four published files are then reproduced byte for byte), --hotpotqa-split
qid draws the HotpotQA validation by question hash (prompts without pilot_split val).
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from add_strict_variants import GENERIC  # noqa: E402
from strict_phrases import HELD_OUT  # noqa: E402

D = HERE / "data"
SRC = D / "publish_onpolicy_sft4152" / "src"
CONFIGS = {
    "hotpotqa": [
        (D / "heldout_full_sft4152" / "pairs_v1.1", D / "heldout_full_sft4152" / "prompts.jsonl", "generic", "v1.1"),
        (SRC / "heldout_strictpool_sft4152" / "pairs_audited", SRC / "heldout_strictpool_sft4152" / "prompts.jsonl",
         "strict", "recheck"),
    ],
    "tatqa": [
        (SRC / "heldout_tatqa_generic_sft4152" / "pairs_audited", SRC / "heldout_tatqa_generic_sft4152" / "prompts.jsonl",
         "generic", "recheck"),
        (SRC / "heldout_tatqa_strict_sft4152" / "pairs_audited", SRC / "heldout_tatqa_strict_sft4152" / "prompts.jsonl",
         "strict", "recheck"),
    ],
}
FIRST = ["prompt", "chosen", "rejected", "category", "dataset", "refusal_instruction", "pid", "qid", "split", "audit"]
DROP = {"is_strict_variant"}  # replaced by refusal_instruction
REDUNDANT = ("dataset", "split", "model", "rejected_source", "audit", "chosen_verified", "rejected_is_greedy",
             "len_chosen", "len_rejected")


def rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def norm(s):
    return " ".join(str(s).lower().split())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tatqa_ids(path):
    q, t = set(), set()
    for doc in json.load(open(path, encoding="utf-8")):
        t.add(str(doc["table"]["uid"]))
        q.update(str(x["uid"]) for x in doc["questions"])
    return q, t


GENERIC_PLAIN = GENERIC.split(" and invite the user")[0] + "."
QUOTE_RE = re.compile(r"<<quote>>(.*?)<</quote>>|<quote>(.*?)</quote>|\[QUOTE\](.*?)\[/QUOTE\]"
                      r"|##begin_quote##(.*?)##end_quote##", re.S)
DOCS_MARK = "Here are the retrieved documents"
DECISION = {"A_false_refusal": "answer", "U_unsupported_answer": "refuse"}


def wrong_reasoning_kind(r):
    """New category of an A_wrong_reasoning pair, or None when its rejected flaw is neither of the two kept."""
    j = r.get("rejected_judge_recheck") or r["rejected_judge"]
    if j.get("refused") or j.get("mixed"):
        return None
    if not j["answer_correct"]:
        return "A_wrong_answer"
    if j["answer_supported"] and not j["reasoning_supported"]:
        return "A_unsupported_reasoning"
    return None


def alnum(s):
    return " ".join(re.sub(r"[^0-9a-z]+", " ", s.lower()).split())


def unfaithful_quotes(r):
    """Quotes of the chosen response not found in the documents (whole, or every line / ellipsis segment)."""
    ctx = alnum(r["prompt"][0]["content"].split(DOCS_MARK, 1)[-1])
    bad = []
    for m in QUOTE_RE.finditer(r["chosen"][0]["content"]):
        q = next(g for g in m.groups() if g is not None)
        if alnum(q) in ctx:
            continue
        segs = [alnum(s) for s in re.split(r"\n|\[\.\.\.\]|\.\.\.|…", q)]
        if not all(s in ctx for s in segs if s):
            bad.append(q.strip())
    return bad


def chunks(r):
    return {c.strip() for c in re.findall(r"\[CHUNK_START\](.*?)\[CHUNK_END\]", r["prompt"][0]["content"], re.S)}


def tatqa_table(r):
    return r["qid"].split("_", 1)[0]


def decision_cell(r):
    if r["dataset"] == "hotpotqa":
        kind = r.get("hotpot_type")
    else:
        kind = "arithmetic" if r["answer_type"] == "arithmetic" else "table" if r["answer_from"] == "table" else "other"
    return f"{r['refusal_instruction']}/{kind}"


def keep_order(r):
    """Pairs kept first when a group is capped: greedy rejected (default behaviour), then a fixed hash."""
    return (not r.get("rejected_is_greedy"), hashlib.sha256(f"{r['pid']}:{r['category']}".encode()).hexdigest())


def cap(group, n):
    """The pairs of `group` beyond the first n in keep_order."""
    return sorted(group, key=keep_order)[max(n, 0):]


def decision_drops(rs, ratio):
    cells = collections.defaultdict(lambda: {"answer": [], "refuse": []})
    for r in rs:
        if r["category"] in DECISION:
            cells[decision_cell(r)][DECISION[r["category"]]].append(r)
    out = []
    for c in cells.values():
        small, big = sorted(c.values(), key=len)
        out += cap(big, int(len(small) * ratio))
    return out


def phrase_drops(rs):
    chosen, rejected = collections.Counter(), collections.defaultdict(list)
    for r in rs:
        if r["refusal_instruction"] != "strict":
            continue
        if r["category"].startswith("U_"):
            chosen[r["strict_phrase"]] += 1
        elif r["category"] == "A_false_refusal" and r.get("rejected_is_exact_phrase"):
            rejected[r["strict_phrase"]].append(r)
    return [x for p, g in rejected.items() for x in cap(g, chosen[p])]


def correct(rs, args):
    """Audit corrections (see the docstring) on all the pairs of one config -> ({split: pairs}, report)."""
    rep = {"dropped": {}}
    tally = lambda drop: dict(sorted(collections.Counter(  # noqa: E731
        f"{r['refusal_instruction']}/{r['category']}" for r in drop).items()))
    if args.generic_rule == "plain":
        for r in rs:
            sys_msg = r["prompt"][0]
            n = sys_msg["content"].count(GENERIC)
            generic = r["refusal_instruction"] == "generic"
            # prompts of dataset_rag_dpo_prompts (jz_run) already carry the plain rule: nothing to replace
            assert n == generic or (generic and n == 0 and GENERIC_PLAIN in sys_msg["content"]), (r["pid"], n)
            sys_msg["content"] = sys_msg["content"].replace(GENERIC, GENERIC_PLAIN)
    if args.quote_filter:
        bad = {id(r): unfaithful_quotes(r) for r in rs}
        drop = [r for r in rs if bad[id(r)]]
        rep["dropped"]["quote_filter"] = tally(drop)
        rep["quote_filter_examples"] = [{"pid": r["pid"], "quote": bad[id(r)][0][:300]} for r in drop[:5]]
        rs = [r for r in rs if not bad[id(r)]]
    if args.split_wrong_reasoning:
        kind = {id(r): wrong_reasoning_kind(r) for r in rs if r["category"] == "A_wrong_reasoning"}
        rep["dropped"]["split_wrong_reasoning"] = tally(r for r in rs if id(r) in kind and kind[id(r)] is None)
        rs = [r for r in rs if kind.get(id(r), "") is not None]
        for r in rs:
            r["category"] = kind.get(id(r), r["category"])
    if args.tatqa_split == "table":
        for r in rs:
            if r["dataset"] == "tatqa":
                h = int(hashlib.sha256(f"42:{tatqa_table(r)}".encode()).hexdigest(), 16) % 100
                r["split"] = "val" if h < args.val_percent else "train"
    if args.hotpotqa_split == "qid":
        for r in rs:
            if r["dataset"] == "hotpotqa":
                h = int(hashlib.sha256(f"42:{r['qid']}".encode()).hexdigest(), 16) % 100
                r["split"] = "val" if h < args.val_percent else "train"
    dec, phr = [], []
    while True:
        d = decision_drops(rs, args.decision_ratio) if args.decision_ratio > 0 else []
        ids = {id(r) for r in d}
        rs = [r for r in rs if id(r) not in ids]
        p = phrase_drops(rs) if args.phrase_balance else []
        ids = {id(r) for r in p}
        rs = [r for r in rs if id(r) not in ids]
        dec, phr = dec + d, phr + p
        if not d and not p:
            break
    rep["dropped"]["decision_ratio"], rep["dropped"]["phrase_balance"] = tally(dec), tally(phr)
    cells = collections.defaultdict(collections.Counter)
    for r in rs:
        if r["category"] in DECISION:
            cells[decision_cell(r)][DECISION[r["category"]]] += 1
    rep["decision_cells"] = {c: dict(v) for c, v in sorted(cells.items())}
    pc = collections.defaultdict(lambda: [0, 0])
    for r in rs:
        if r["refusal_instruction"] == "strict" and r["category"].startswith("U_"):
            pc[r["strict_phrase"]][0] += 1
        elif r["category"] == "A_false_refusal" and r.get("rejected_is_exact_phrase"):
            pc[r["strict_phrase"]][1] += 1
    rep["phrases_rejected_more_than_chosen"] = sum(c < j for c, j in pc.values())
    return {s: [r for r in rs if r["split"] == s] for s in ("train", "val")}, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True, help="policy tokenizer (with its chat template)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--generic-rule", choices=("plain", "invite"), default="plain")
    ap.add_argument("--quote-filter", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--split-wrong-reasoning", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tatqa-split", choices=("table", "qid"), default="table")
    ap.add_argument("--hotpotqa-split", choices=("keep", "qid"), default="keep",
                    help="keep = split of the pairs (pilot_split of the prompts); qid = ~val-percent of the questions")
    ap.add_argument("--val-percent", type=int, default=10, help="TatQA table split (and --hotpotqa-split qid)")
    ap.add_argument("--decision-ratio", type=float, default=1.0)
    ap.add_argument("--phrase-balance", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--drop-redundant", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--sources", default=None,
                    help="JSON {config: [[pairs_dir, prompts.jsonl, generic|strict, audit], ...]} replacing CONFIGS")
    ap.add_argument("--gold-columns", action=argparse.BooleanOptionalAction, default=False,
                    help="gold_answer / gold_scale after qid, as commit 27e169e of the hub repo (gold_columns.py)")
    args = ap.parse_args()
    configs = CONFIGS if args.sources is None else {
        name: [(Path(p), Path(q), instruction, audit) for p, q, instruction, audit in parts]
        for name, parts in json.loads(Path(args.sources).read_text(encoding="utf-8")).items()}
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset, load_from_disk
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    n_tok = lambda msgs: len(tok.apply_chat_template(msgs, tokenize=True, return_dict=False))  # noqa: E731
    out = Path(args.out)

    data, keys, tok_stats, corrections = {}, set(), {}, {}
    for name, parts in configs.items():
        recs = {"train": [], "val": []}
        ntok, too_long = {}, collections.Counter()
        for pdir, prompts_path, instruction, audit in parts:
            meta = {p["pid"]: p for p in rows(prompts_path)}
            for split in ("train", "val"):
                for r in rows(pdir / f"{split}.jsonl"):
                    n = max(n_tok(r["prompt"] + r[side]) for side in ("chosen", "rejected"))
                    if n > args.max_tokens:
                        too_long[f"{instruction}/{r['category']}"] += 1
                        continue
                    ntok[id(r)] = n
                    r.update(dataset=name, refusal_instruction=instruction, audit=audit)
                    if name == "tatqa":
                        m = meta[r["pid"]]
                        r["answer_type"], r["answer_from"], r["hotpot_type"] = m["answer_type"], m["answer_from"], None
                        # refine_pairs' text containment (punctuation and sign stripped) is meaningless on numeric
                        # golds: "-1.5" matches any "15". HotpotQA-only field.
                        r["rejected_states_twin_gold"] = None
                    for k in DROP:
                        r.pop(k, None)
                    recs[split].append(r)
                    keys.update(r)
        recs, corrections[name] = correct(recs["train"] + recs["val"], args)
        lens = sorted(ntok[id(r)] for rs in recs.values() for r in rs)
        if not lens:  # every pair dropped (tiny smoke runs): config left out, said in the manifest
            tok_stats[name] = {f"dropped_over_{args.max_tokens}": dict(too_long)}
            continue
        tok_stats[name] = {"median": lens[len(lens) // 2], "p99": lens[int(len(lens) * .99)], "max": lens[-1],
                           f"dropped_over_{args.max_tokens}": dict(too_long)}
        data[name] = recs

    order = FIRST + sorted(keys - set(FIRST))
    if args.drop_redundant:
        order = [k for k in order if k not in REDUNDANT]
    held = [norm(h) for h in HELD_OUT]
    test_q, test_t = tatqa_ids(HERE.parent / "tatqa_dataset_test_gold.json")
    dev_q, dev_t = tatqa_ids(HERE.parent / "tatqa_dev.json")
    hp_val = set(load_from_disk(str(HERE.parent / "hotpotqa_cache_validation"))["id"])
    man = {"settings": {k: getattr(args, k) for k in ("generic_rule", "quote_filter", "split_wrong_reasoning",
                                                     "tatqa_split", "val_percent", "decision_ratio",
                                                     "phrase_balance", "drop_redundant", "hotpotqa_split",
                                                     "gold_columns")},
           "policy": sorted({r.get("model") for recs in data.values() for rs in recs.values() for r in rs}),
           "corrections": corrections, "configs": {}, "schema": order, "checks": {}, "tokens": tok_stats,
           "sources": {n: [f"{p[0]} ({p[2]}, audit {p[3]})" for p in parts] for n, parts in configs.items()},
           "empty_configs": sorted(set(configs) - set(data))}
    for name, recs in data.items():
        allr = recs["train"] + recs["val"]
        c = collections.Counter()
        tq, vq = {r["qid"] for r in recs["train"]}, {r["qid"] for r in recs["val"]}
        c["qid_in_train_and_val"] = len(tq & vq)
        train_chunks = set().union(*(chunks(r) for r in recs["train"]))
        c["val_pairs_sharing_a_chunk_with_train"] = sum(bool(chunks(r) & train_chunks) for r in recs["val"])
        if name == "tatqa":
            tt = {tatqa_table(r) for r in recs["train"]}
            c["val_pairs_gold_table_in_train"] = sum(tatqa_table(r) in tt for r in recs["val"])
        c["generic_rule_with_invitation"] = sum(GENERIC in r["prompt"][0]["content"] for r in allr)
        for r in allr:
            texts = [m["content"] for m in r["prompt"]] + [r["chosen"][0]["content"], r["rejected"][0]["content"]]
            c["lighteval_phrase"] += any(h in norm(t) for t in texts for h in held)
            c["empty_response"] += not r["chosen"][0]["content"].strip() or not r["rejected"][0]["content"].strip()
            c["leading_ws_asymmetric"] += r["chosen"][0]["content"][:1].isspace() != r["rejected"][0]["content"][:1].isspace()
            if name == "tatqa":
                table_uid, q_uid = r["qid"].split("_", 1)
                c["tatqa_uid_in_test_or_dev"] += q_uid in test_q | dev_q or table_uid in test_t | dev_t
            else:
                c["hotpot_id_in_validation"] += r["qid"] in hp_val
            if r["refusal_instruction"] == "strict":
                c["phrase_rejected"] += bool(r.get("rejected_is_exact_phrase")) and r["category"] == "A_false_refusal"
                c["phrase_chosen"] += r["category"].startswith("U_")
        c["phrase_balance_ok"] = c["phrase_chosen"] >= c["phrase_rejected"]
        man["checks"][name] = dict(c)
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        ordered = {s: [{k: r.get(k) for k in order} for r in rs] for s, rs in recs.items()}
        schema = pa.Table.from_pylist(ordered["train"] + ordered["val"]).schema
        for split, rs in ordered.items():
            pq.write_table(pa.Table.from_pylist(rs, schema=schema), d / f"{split}.parquet")
            if args.gold_columns:
                from gold_columns import add_to_parquet
                add_to_parquet(d / f"{split}.parquet", name)
        # an empty split (tiny smoke runs) is written but cannot be read by `datasets`
        loaded = load_dataset("parquet", data_files={s: str(d / f"{s}.parquet") for s in ("train", "val") if recs[s]})
        man["checks"][name]["loads_with_datasets"] = {s: len(loaded[s]) if s in loaded else 0 for s in ("train", "val")}
        cat = collections.Counter((r["refusal_instruction"], r["category"]) for r in allr)
        man["configs"][name] = {
            "n": len(allr), "train": len(recs["train"]), "val": len(recs["val"]), "questions": len(tq | vq),
            "by_instruction": dict(collections.Counter(r["refusal_instruction"] for r in allr)),
            "category": {f"{i}/{k}": v for (i, k), v in sorted(cat.items())},
            "chosen_source": dict(collections.Counter(r.get("chosen_source") for r in allr)),
            "unans_mode": dict(collections.Counter(r["unans_mode"] for r in allr if r.get("unans_mode"))),
            "answer_type": dict(collections.Counter(r["answer_type"] for r in allr if r.get("answer_type"))),
            "hotpot_type": dict(collections.Counter(r["hotpot_type"] for r in allr if r.get("hotpot_type"))),
            "sha256": {f"{s}.parquet": sha256(d / f"{s}.parquet") for s in ("train", "val")},
        }
    man["source_sha256"] = {"tatqa_test": sha256(HERE.parent / "tatqa_dataset_test_gold.json"),
                            "tatqa_dev": sha256(HERE.parent / "tatqa_dev.json")}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: man[k] for k in ("corrections", "configs", "checks", "tokens")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
