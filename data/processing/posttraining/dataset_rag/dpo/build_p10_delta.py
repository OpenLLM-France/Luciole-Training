#!/usr/bin/env python3
"""P10 (delta pairs): chosen = a Qwen3.5-35B-A3B response confirmed by the judge, rejected = a Qwen3.5-4B response of
the same prompt judged faulty. No text of the policy: the pair only contrasts quality, between two models of the same
family (no policy-vs-big-model provenance shortcut, the failure of the off-policy twins).

Prompts: those of the 8802410 pairs (dpo/data/qwen35_chosen_onpolicy and qwen35_4b_chosen_onpolicy, same pids), minus
the strict run with the 5 phrases of hotpotqa_v4 (left out of every release), the 8802410 validation questions (TAT-QA:
validation tables and the hash-split tables) and the P9 paired-evaluation questions (TAT-QA: their tables).
35B chosen = qwen_chosen_all.accepted (--prefer-quoted), as for the published twins. 4B rejected, one per prompt:
  A_wrong_answer / A_unsupported_reasoning   4B answer judged wrong / correct but with unsupported reasoning
                                             (reasoning verdict when there is one, else the first pass)
  A_false_refusal                            4B refusal on an answerable prompt
  A_bad_citation                             4B answer with non-conforming citations (qwen_chosen_all.citation_ok) whose
                                             content the judge finds correct and supported (`--todo-citations` lists
                                             them, run_p10_delta.sh judges them into <4B dir>/scored_citfail.jsonl)
  U_unsupported_answer / U_bad_refusal       4B answer on an unanswerable prompt / refusal that misstates what is missing
  U_strict_inexact                           strict prompt: 4B response that is not the imposed phrase
Greedy rejected first, then the closest length. Release rules of the off-policy configs, on BOTH sides (both are
written by a bigger model): published refusal rule, normalize() (opening about the documents, inviting sentence, `**`),
chosen quotes found in the documents, lighteval phrases absent, TAT-QA unanswerable leak check.

Output: dpo/data/dpo_p10_delta = P3a train + delta pairs (validation = 8802410 val), manifest.json.
--jz: the Jean Zay delta run instead (dpo/data/jz_run/{qwen35_35b_a3b,qwen35_4b}/{hotpotqa,tatqa}, build_jz_run.py, every
4B sample judged there, A_bad_citation verdicts already in scored_citfail.jsonl) -> dpo/data/jz_delta/delta_pairs.jsonl +
manifest.json, delta pairs only (no P3a base, no split), plus the DPOFilter checks of the colleague on the chosen
("Qwen" mention, CJK absent from the documents) and delta_pairs_datatrove.jsonl (his document format; --tokenizer PATH for token counts).
    cd $RAG_ROOT && .venv/bin/python dpo/build_p10_delta.py [--todo-citations] [--jz]
"""
import collections
import hashlib
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qwen_chosen_all as Q  # noqa: E402
from build_pilot_pairs import is_good  # noqa: E402
from release_text import TatqaLeak, normalize, published_prompt, quotes_found  # noqa: E402
from strict_phrases import HELD_OUT  # noqa: E402

D = HERE / "data"
SUBS = ("tatqa", "hotpotqa", "hotpotqa_strictpool")
D35, D4 = D / "qwen35_chosen_onpolicy", D / "qwen35_4b_chosen_onpolicy"
BASE, P3A, P9E = D / "rag_dpo_onpolicy_8802410", D / "dpo_p3a_Ux2", D / "p9_paired_eval"
PRIORITY = ("A_wrong_answer", "A_unsupported_reasoning", "A_false_refusal", "A_bad_citation",
            "U_unsupported_answer", "U_bad_refusal", "U_strict_inexact")
KEEP = ("prompt", "chosen", "rejected", "category", "pid", "qid", "refusal_instruction")


def rows(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def table(qid):
    return qid.split("_", 1)[0]


def export_datatrove(pairs, path, man):
    """Same pairs in the document format of the colleague's create_preference_data.py (datatrove JsonlWriter: text, id,
    metadata.{context, chosen, rejected, chosen_size, rejected_size, token_diff, token_ratio}), so that his scripts
    (score_preference_pairs.py, judge_summary.py, lenbias.py) run on them unchanged. Tokens: --tokenizer PATH (Qwen3.5),
    else whitespace words (flagged in the manifest)."""
    tok = None
    if "--tokenizer" in sys.argv:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(sys.argv[sys.argv.index("--tokenizer") + 1])
    n = (lambda t: len(tok(t, add_special_tokens=False)["input_ids"])) if tok else (lambda t: len(t.split()))
    with open(path, "w", encoding="utf-8") as f:
        for r in pairs:
            c, j = r["chosen"][0]["content"], r["rejected"][0]["content"]
            nc, nr = n(c), n(j)
            chosen = r["prompt"] + r["chosen"]
            f.write(json.dumps({"text": "\n".join(m["content"] for m in chosen), "id": r["pid"], "metadata": {
                "context": r["prompt"], "chosen": chosen, "rejected": r["prompt"] + r["rejected"],
                "chosen_size": "qwen35_35b_a3b", "rejected_size": "qwen35_4b", "token_diff": nc - nr,
                "token_ratio": nc / max(nr, 1), **{k: r[k] for k in ("category", "dataset", "qid", "unans_mode")}}},
                ensure_ascii=False) + "\n")
    man["datatrove_export"] = {"file": path.name, "token_count": "tokenizer" if tok else "whitespace_words"}


def main():
    global SUBS, D35, D4
    todo_mode = "--todo-citations" in sys.argv
    jz = "--jz" in sys.argv
    if jz:
        run = Path(os.environ.get("JZ_RUN", D / "jz_run"))  # JZ_RUN=dpo/data/jz_run_test : test
        SUBS, D35, D4 = ("hotpotqa", "tatqa"), run / "qwen35_35b_a3b", run / "qwen35_4b"
    Q.PREFER_QUOTED = True
    val_q = {r["qid"] for r in rows(BASE / "val.jsonl")}
    ev_q = {p["qid"] for p in rows(P9E / "prompts.jsonl")} if (P9E / "prompts.jsonl").exists() else set()
    bad_tables = {table(q) for q in val_q | ev_q if "_" in q}
    held = [" ".join(h.lower().split()) for h in HELD_OUT]
    leak_file = HERE.parent / "tatqa_dataset_train.json"
    leak = TatqaLeak(leak_file) if leak_file.exists() else None
    pairs, st, lens = [], collections.Counter(), collections.defaultdict(list)
    for sub in SUBS:
        Q.GEN = "qwen35_35b_a3b"
        P, S35, sc35, ve35, rj35 = Q.state(D35 / sub)
        Q.GEN = "qwen35_4b"
        _, S4, sc4, ve4, rj4 = Q.state(D4 / sub)
        cit = Q.last_by_key(D4 / sub / "scored_citfail.jsonl")
        todo = []
        dataset = "tatqa" if sub == "tatqa" else "hotpotqa"
        for pid, p in P.items():
            q = p["qid"]
            if str(p.get("pair_source", "")).endswith("heldout_strict_sft4152"):
                st["excluded_strict_v4_phrase_run"] += 1
                continue
            if q in val_q or q in ev_q or ("_" in q and (table(q) in bad_tables or
                                                         int(hashlib.sha256(f"42:{table(q)}".encode()).hexdigest(), 16) % 100 < 10)):
                st["excluded_val_or_paired_eval"] += 1
                continue
            Q.GEN = "qwen35_35b_a3b"
            acc = Q.accepted(p, S35.get(pid, {}), sc35, ve35, rj35)
            if not acc and p["chosen_kind"] == "phrase":
                # strict unanswerable prompt where the 35B never output the exact imposed phrase: the chosen is the
                # phrase itself (what the instruction asks for, checked lexically, no generator text)
                acc = [{"k": None, "text": p["strict_phrase"], "finish_reason": "stop"}]
                st[f"{dataset}/phrase_chosen_fallback"] += 1
            if not acc:
                continue
            ch, kind = acc[0], p["chosen_kind"]
            bad = collections.defaultdict(list)
            for k, s in S4.get(pid, {}).items():
                if not Q.usable(s):
                    continue
                if kind == "answer":
                    j = (ve4.get((pid, k)) or sc4.get((pid, k)) or {}).get("judge")
                    if j is None:
                        if Q.citation_ok(p, s["text"]):
                            continue
                        c = cit.get((pid, k), {}).get("judge")
                        if c is None:
                            todo.append({"pid": pid, "k": k})
                        elif not c["refused"] and not c["mixed"] and c["answer_correct"] and c["answer_supported"] \
                                and c["reasoning_supported"]:
                            bad["A_bad_citation"].append(s)
                    elif j["refused"]:
                        bad["A_false_refusal"].append(s)
                    elif not j["mixed"] and not is_good(j, False):
                        if not j["answer_correct"]:
                            bad["A_wrong_answer"].append(s)
                        elif j["answer_supported"] and not j["reasoning_supported"]:
                            bad["A_unsupported_reasoning"].append(s)
                elif kind == "refusal":
                    r = rj4.get((pid, k))
                    if r and r["judge"]:
                        if not r["judge"]["refused"]:
                            bad["U_unsupported_answer"].append(s)
                        elif not r["judge"]["good_chosen"]:
                            bad["U_bad_refusal"].append(s)
                elif not Q.is_exact(s["text"], p["strict_phrase"]):
                    bad["U_strict_inexact"].append(s)
            cat = next((c for c in PRIORITY if bad[c]), None)
            if cat is None:
                continue
            rej = min(bad[cat], key=lambda s: (s["k"] != -1, abs(len(s["text"]) - len(ch["text"]))))
            ph = p.get("strict_phrase")

            def norm(text):
                # the style normalisation strips "Based on the retrieved context, ..." openings and inviting sentences,
                # i.e. parts of 6 of the 30 imposed phrases: a text that IS the imposed phrase is kept as is
                if ph and Q.is_exact(text, ph):
                    return text.strip().replace("**", "")
                return normalize(text, True)
            chosen, rejected = norm(ch["text"]), norm(rej["text"])
            tag = f"{dataset}/{cat}"
            if cat == "U_strict_inexact" and Q.is_exact(rejected, ph):
                st[f"{tag}/dropped_rejected_exact_after_normalisation"] += 1
                continue
            prompt = published_prompt(p["messages"])
            if not chosen or not rejected or chosen == rejected:
                st[f"{tag}/dropped_empty_or_identical"] += 1
                continue
            if cat.startswith("A_") and not quotes_found(chosen, p["citation_style"], prompt[0]["content"]):
                st[f"{tag}/dropped_chosen_quote_not_in_documents"] += 1
                continue
            if any(h in " ".join(t.lower().split()) for t in (chosen, rejected) for h in held):
                st[f"{tag}/dropped_lighteval_phrase"] += 1
                continue
            if leak and dataset == "tatqa" and p["is_unanswerable"] and leak.leaks(prompt[0]["content"], q, p["gold_answer"]):
                st[f"{tag}/dropped_answer_leak_same_report"] += 1
                continue
            # filtres DPOFilter de create_preference_data.py (collègue) sur le chosen : mention de Qwen, caractères CJK
            if jz and "Qwen" in chosen:
                st[f"{tag}/dropped_chosen_mentions_qwen"] += 1
                continue
            # CJK : seulement s'il ne vient pas des documents (HotpotQA : 40 sorties du 35B recopient des noms chinois)
            ctx = " ".join(m["content"] for m in prompt)
            if jz and any(c not in ctx for c in re.findall(r"[\u4e00-\u9fff]+", chosen)):
                st[f"{tag}/dropped_chosen_cjk_not_in_documents"] += 1
                continue
            st[f"{tag}/kept"] += 1
            lens[cat].append((len(chosen), len(rejected)))
            pairs.append({"prompt": prompt, "chosen": [{"role": "assistant", "content": chosen}],
                          "rejected": [{"role": "assistant", "content": rejected}], "category": cat, "pid": pid,
                          "qid": q, "refusal_instruction": "strict" if p.get("strict_refusal_instruction") else "generic",
                          "dataset": dataset, "pair_source": "jz_delta_35b_vs_4b" if jz else "p10_delta_35b_vs_4b",
                          "chosen_k": ch["k"], "unans_mode": p.get("unans_mode"), "clone_source": p.get("clone_source"),
                          "strict_phrase": ph, "chosen_is_imposed_phrase": bool(ph and Q.is_exact(chosen, ph)),
                          "rejected_is_exact_phrase": bool(ph and Q.is_exact(rejected, ph)),
                          "rejected_k": rej["k"]})
        if todo_mode:
            with open(D4 / sub / "p10_todo_citfail.jsonl", "w") as f:
                for t in todo:
                    f.write(json.dumps(t) + "\n")
            st[f"{dataset}/citations_to_judge/{sub}"] = len(todo)
    if todo_mode:
        print(json.dumps(dict(sorted(st.items())), indent=1))
        return
    report = {"stats": dict(sorted(st.items())),
              "added": dict(sorted(collections.Counter(f'{r["dataset"]}/{r["refusal_instruction"]}/{r["category"]}'
                                                       for r in pairs).items())),
              "median_len_chosen_rejected": {c: (statistics.median(a for a, _ in v), statistics.median(b for _, b in v))
                                             for c, v in sorted(lens.items())}}
    if jz:
        out = Path(os.environ.get("JZ_OUT", D / "jz_delta"))
        out.mkdir(exist_ok=True)
        with open(out / "delta_pairs.jsonl", "w", encoding="utf-8") as f:
            for r in pairs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        man = {"description": f"{len(pairs)} delta pairs of the Jean Zay run (35B chosen, 4B rejected; build_p10_delta.py --jz)",
               **report, "sha256": hashlib.sha256((out / "delta_pairs.jsonl").read_bytes()).hexdigest()}
        export_datatrove(pairs, out / "delta_pairs_datatrove.jsonl", man)
        (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps(man, indent=1, ensure_ascii=False))
        return
    base = [{**{k: r.get(k) for k in KEEP}, "dataset": r.get("dataset"), "pair_source": r.get("pair_source")}
            for r in rows(P3A / "train.jsonl")]
    val = [{**{k: r.get(k) for k in KEEP}, "dataset": "tatqa" if "_" in r["qid"] else "hotpotqa",
            "pair_source": "onpolicy_8802410"} for r in rows(BASE / "val.jsonl")]
    rng = random.Random(42)
    train = base + pairs
    rng.shuffle(train)
    out = D / "dpo_p10_delta"
    out.mkdir(exist_ok=True)
    man = {"description": f"P3a train ({len(base)}) + {len(pairs)} delta pairs (35B chosen, 4B rejected; build_p10_delta.py)",
           **report, "files": {}}
    for s, rs in (("train", train), ("val", val)):
        p = out / f"{s}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        man["files"][s] = {"n": len(rs), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    with open(out / "delta_pairs.jsonl", "w", encoding="utf-8") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in man.items() if k != "files"}, indent=1, ensure_ascii=False), man["files"])


if __name__ == "__main__":
    main()
