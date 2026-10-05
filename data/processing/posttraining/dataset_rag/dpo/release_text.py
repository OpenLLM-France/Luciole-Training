#!/usr/bin/env python3
"""Text helpers shared by the DPO release scripts (build_p10_delta.py, build_jz_prompts.py, build_unans_dag.py).

  published_prompt  published generic refusal rule: the clause " and invite the user to rephrase the query or add
                    relevant documents" removed from the system prompt.
  normalize         style normalisation of a pair side (below).
  quotes_found      every <quote> excerpt of a chosen found in the documents.
  TatqaLeak         TAT-QA unanswerable prompt whose reference answer appears in a document of the gold report.

Style normalisation (decision 2026-10-02, option b): the big models open with "Based on the provided documents, ..."
(53-97 %) and use Markdown bold (96-100 %), the policy rarely: left as is, DPO would learn the style. On every pair:
the opening clause is removed from the chosen, and `**` is removed from chosen AND rejected. Leading whitespace is
stripped on both sides.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dpo_common import STYLE_MARKERS  # noqa: E402

CLAUSE = " and invite the user to rephrase the query or add relevant documents"
QUOTE_END = {"default": "##end_quote##", "xml": "</quote>", "bracket": "[/QUOTE]", "double_angle": "<</quote>>"}


def norm_txt(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def published_prompt(prompt):
    return [{**m, "content": m["content"].replace(CLAUSE, "")} if m["role"] == "system" else m for m in prompt]


INTRO = re.compile(r"^(?:based on|according to|from|using|looking at|referring to)\s+(?P<body>[^,:\n]{0,90}?)\s*[,:]\s*", re.I)
INTRO_WORDS = re.compile(r"document|context|data|information|report|table|paragraph|statement|source|figure|excerpt", re.I)
SPECIFIC = re.compile(r"^specifically,?\s+(?:in\s+)?(?P<src>(?:Table|Paragraph) \d+)(?:\s+(?:in|of) the (?:context|documents?))?\s*,\s*", re.I)
INVITE = re.compile(r"rephrase|reformulat|add (?:more |additional )?(?:relevant )?documents|provid(?:e|ing) (?:additional|more|further) "
                    r"(?:documents|context|information)|additional (?:relevant )?documents", re.I)
INVITE_SENT = re.compile(r"[^.!?\n]*(?:" + INVITE.pattern + r")[^.!?\n]*[.!?]?[ \t]*", re.I)


def normalize(text, chosen_side):
    """`**` removed on both sides; on the chosen (bigger model) side also the opening that refers to the documents
    ("Based on the provided documents, ...", "Based on the data in Table 2, ...": removed, even when it names a table,
    since the policy never opens that way and the citation markers name the source) and the sentences inviting the
    user to rephrase or add documents (no longer asked by the published prompts)."""
    t = text.strip().replace("**", "")
    if not chosen_side:
        return t
    m = INTRO.match(t)
    if m and INTRO_WORDS.search(m.group("body")):
        t = t[m.end():]
    m = SPECIFIC.match(t)
    if m:  # "specifically Table 4, ..." left by the first cut; the citation markers already name the source
        t = t[m.end():]
    # drop the inviting sentence only, inside its line (lists, tables and quotes keep their line breaks)
    t = INVITE_SENT.sub(" ", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = t.strip()
    return t[:1].upper() + t[1:]


def quotes_found(text, style, context):
    """8802410 rule: every <quote> excerpt found in the documents (lowercase alphanumerics, per line / ellipsis part)."""
    ctx = norm_txt(context)
    for q in re.findall(re.escape(STYLE_MARKERS[style]) + r"(.*?)" + re.escape(QUOTE_END[style]), text, re.S):
        parts = [p for line in q.splitlines() for p in re.split(r"\.\.\.|…", line) if norm_txt(p)]
        if not all(norm_txt(p) in ctx for p in parts):
            return False
    return True


class TatqaLeak:
    """Reference answer inside a document of the gold report (same rule as the on-policy audit)."""

    def __init__(self, train_json):
        docs = json.load(open(train_json, encoding="utf-8"))
        n = lambda s: " ".join(str(s).split()).lower()  # noqa: E731
        self.n = n
        self.paras = [(n(p["text"])[:80], d["table"]["uid"]) for d in docs for p in d["paragraphs"] if len(n(p["text"])) >= 25]
        self.tables = [(d["table"]["uid"], {n(c) for row in d["table"]["table"] for c in row if n(c)}) for d in docs]

    def owner(self, title, body):
        b = self.n(body)
        if title.startswith("Paragraph"):
            hits = {u for k, u in self.paras if k in b}
            return hits.pop() if len(hits) == 1 else None
        cells = {self.n(c) for line in body.splitlines() for c in line.split("|") if self.n(c) and not set(self.n(c)) <= set("-: ")}
        best = max(self.tables, key=lambda t: len(cells & t[1]) / max(1, len(cells | t[1])))
        return best[0] if len(cells & best[1]) / max(1, len(cells | best[1])) > 0.5 else None

    def leaks(self, prompt_system, qid, gold):
        gold = (gold or "").strip()
        if not gold or re.fullmatch(r"(19|20)\d\d(, (19|20)\d\d)*|\d", gold):
            return False
        nums = lambda s: {x.replace(",", "") for x in re.findall(r"\d[\d,]*\.?\d*", s)}  # noqa: E731
        g, uid = nums(gold), qid.split("_")[0]
        for title, body in re.findall(r"\* filename: (.+?)\n\n\[CHUNK_START\]\n(.*?)\n\[CHUNK_END\]", prompt_system, re.S):
            if self.owner(title, body) == uid and ((g and g <= nums(body)) or (not g and self.n(gold) in self.n(body))):
                return True
        return False
