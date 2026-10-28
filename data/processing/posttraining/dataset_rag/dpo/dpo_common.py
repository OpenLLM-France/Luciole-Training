"""Shared helpers for the on-policy RAG DPO pipeline (self-contained, no repo imports).

Everything here must run on the DGX with only the stdlib, so the pipeline does not
depend on the dataset_rag package layout.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── IO ──────────────────────────────────────────────────────────────────────


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def delta_datasets(run, gen: str = "qwen35_35b_a3b") -> list[str]:
    """Datasets of a delta run (dirs of gen holding prompts), hotpotqa / tatqa first: both for the Jean Zay run, the
    set of its SFT run for a prompt set built from one (e.g. hotpotqa_fr)."""
    found = {d.name for d in (Path(run) / gen).glob("*") if (d / "prompts.jsonl").is_file()}
    return [ds for ds in ("hotpotqa", "tatqa") if ds in found] + sorted(found - {"hotpotqa", "tatqa"})


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


# ── Lexical refusal (EN/FR/DE patterns of dataset_rag/utils.py REFUSAL_PATTERNS)

REFUSAL_PATTERNS = [
    r"(?:do not|don't|cannot|can't|unable to) (?:allow me to |)(?:answer|provide|respond)",
    r"(?:not enough|insufficient) information",
    r"does not (?:contain|allow|provide)",
    r"ne (?:me )?permettent? pas",
    r"pas en mesure de répondre",
    r"ne (?:peux|puis) pas (?:fournir|répondre)",
    r"(?:nicht in der Lage|ermöglich(?:t|en) es mir nicht|kann .* nicht beantworten)",
    r"nicht genügend Informationen",
]
_REFUSAL_RE = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]


def lexical_refusal(text: str) -> bool:
    return any(p.search(text) for p in _REFUSAL_RE)


# ── Context & citations ─────────────────────────────────────────────────────

_FILENAME_RE = re.compile(r"^\* filename: (.+?)\s*$", re.M)
# Numbered documents. OpenRAG layout (formatting_sft.format_context): every document opens with its "[Source N]" line
# (the first one right after the opening backtick of the context), then its chunk with its "* filename:" header; chunk
# texts cannot contain "[Source" (neutralized). Prompts built before that layout number the documents of the ref
# format only, as "[N] filename: <title>". ref citation format (formatting_sft.CITATION_FORMATS): cited inline as
# <ref id="N">verbatim excerpt</ref>, or listed in a References section as <ref id="N"/> (end-of-completion placement).
_SOURCE_RE = re.compile(r"(?:^|(?<=`))\[Source (\d+)\]$", re.M)
_NUMBERED_RE = re.compile(r"(?:^|(?<=`))\[(\d+)\] filename: (.+?)\s*$", re.M)
_REF_TAG_RE = re.compile(r'<ref id="(\d+)">(.*?)</ref>', re.S)
_REF_DOC_RE = re.compile(r'<ref id="(\d+)"/>')
# left over once the valid tags are removed: a malformed tag or a citation in another format (formatting_sft)
_STRAY_CITATION_RE = re.compile(
    r"</?ref\b|##Cite|##begin_quote##|##end_quote##|<<quote>>|<<cite:|<quote>|<cite>|\[QUOTE\]|\[Source:")

# The 4 training citation styles of formatting_sft.CITATION_STYLES.
STYLE_MARKERS = {
    "default": "##begin_quote##",
    "xml": "<quote>",
    "bracket": "[QUOTE]",
    "double_angle": "<<quote>>",
}
_CITE_PATTERNS = {
    "default": re.compile(r'##Cite\s*"([^"]+)"\s*##'),
    "xml": re.compile(r'<cite>\s*"([^"]+)"\s*</cite>'),
    "bracket": re.compile(r'\[Source:\s*"([^"]+)"\s*\]'),
    "double_angle": re.compile(r'<<cite:\s*"([^"]+)"\s*>>'),
}


def context_block(system_prompt: str) -> str:
    """Only the retrieved documents part of the system prompt (what the judge gets in lighteval)."""
    for marker in ("Here are the retrieved documents", "Voici les documents récupérés"):
        i = system_prompt.find(marker)
        if i >= 0:
            return system_prompt[i:]
    return system_prompt


def context_titles(system_prompt: str) -> list[str]:
    """Document titles in context order; for numbered documents, index N-1 is document N."""
    numbered = numbered_documents(system_prompt)
    if numbered:
        return [numbered[n][0] for n in sorted(numbered)]
    return _FILENAME_RE.findall(system_prompt)


def numbered_documents(system_prompt: str) -> dict[int, tuple[str, str]]:
    """{N: (title, text)} of the "[Source N]" documents, or of the "[N] filename: ..." ones of a ref prompt built before
    the OpenRAG layout (a document runs to the next header)."""
    heads = list(_SOURCE_RE.finditer(system_prompt))
    if heads:
        docs = {}
        for i, m in enumerate(heads):
            text = system_prompt[m.end(): heads[i + 1].start() if i + 1 < len(heads) else len(system_prompt)]
            title = _FILENAME_RE.search(text)
            docs[int(m.group(1))] = (title.group(1).strip() if title else "", text)
        return docs
    heads = list(_NUMBERED_RE.finditer(system_prompt))
    return {int(m.group(1)): (m.group(2).strip(), system_prompt[m.end(): heads[i + 1].start() if i + 1 < len(heads)
                                                                  else len(system_prompt)])
            for i, m in enumerate(heads)}


def ref_citations(text: str, system_prompt: str, end_of_completion: bool) -> tuple[list[str], int, int]:
    """(cited titles, ref tags, invalid ones) as formatting_sft.parse_ref_citations (eval_ref_invalid of the SFT
    filter): a tag is invalid when its id is not a document number or, inline, when its excerpt is empty or not verbatim
    (up to whitespace) in that document; every stray tag or citation of another format counts as invalid."""
    ws = lambda t: re.sub(r"\s+", " ", t).strip()  # noqa: E731
    docs = numbered_documents(system_prompt)
    tag_re = _REF_DOC_RE if end_of_completion else _REF_TAG_RE
    titles, invalid = [], 0
    refs = list(tag_re.finditer(text))
    for ref in refs:
        n = int(ref.group(1))
        if n not in docs or (not end_of_completion and (not ws(ref.group(2)) or ws(ref.group(2)) not in ws(docs[n][1]))):
            invalid += 1
            continue
        if docs[n][0] not in titles:
            titles.append(docs[n][0])
    invalid += len(_STRAY_CITATION_RE.findall(tag_re.sub("", text)))
    return titles, len(refs), invalid


def requested_citation_style(system_prompt: str) -> tuple[str | None, str]:
    """(style name, placement) requested by the system prompt's citation rule ("ref" for the ref format)."""
    if '<ref id="N"' in system_prompt:
        return "ref", ("eoc" if '<ref id="N"/>' in system_prompt else "inline")
    rule = next((l for l in system_prompt.splitlines()
                 if "When quoting from the context" in l or "Lorsque vous citez" in l), "")
    style = None
    # double_angle must be tested before xml ("<<quote>>" contains "<quote>").
    for name in ("double_angle", "default", "bracket", "xml"):
        if f"`{STYLE_MARKERS[name]}`" in rule:
            style = name
            break
    placement = "eoc" if ("Do **not** add citation" in rule or "N'ajoutez **pas**" in rule) else "inline"
    return style, placement


def extract_citations(text: str) -> dict[str, list[str]]:
    """Cited titles per style found in the text (all styles are scanned)."""
    out = {}
    for name, pat in _CITE_PATTERNS.items():
        found = [t.strip() for t in pat.findall(text) if t.strip()]
        if found:
            out[name] = list(dict.fromkeys(found))
    return out


def _norm_title(t: str) -> str:
    return " ".join(t.strip().lower().split())


def _title_match(a: str, b: str) -> bool:
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    return longer.startswith(shorter) and len(shorter.split()) >= 2


def citation_metrics(text: str, gold_titles: list[str], ctx_titles: list[str],
                     style: str | None, system_prompt: str | None = None) -> dict:
    """Citation precision / recall / F1 against the gold titles. ref format (style "ref", system_prompt needed for the
    numbered documents): cited = documents of the valid tags, n_invalid_cited = invalid tags (eval_ref_invalid)."""
    ref = None
    if style == "ref":
        if system_prompt is None:
            raise ValueError("citation_metrics: the ref format needs the system prompt")
        ref = ref_citations(text, system_prompt, requested_citation_style(system_prompt)[1] == "eoc")
        by_style = {"ref": ref[0]} if ref[0] else {}
        by_style.update(extract_citations(text))
    else:
        by_style = extract_citations(text)
    cited = list(dict.fromkeys(t for ts in by_style.values() for t in ts))
    cited_n = [_norm_title(t) for t in cited]
    gold_n = [_norm_title(t) for t in gold_titles]
    ctx_n = [_norm_title(t) for t in ctx_titles]
    in_gold = [c for c in cited_n if any(_title_match(c, g) for g in gold_n)]
    in_ctx = [c for c in cited_n if any(_title_match(c, x) for x in ctx_n)]
    gold_hit = [g for g in gold_n if any(_title_match(c, g) for c in cited_n)]
    p = len(in_gold) / len(cited_n) if cited_n else None
    r = len(gold_hit) / len(gold_n) if gold_n else None
    f1 = (2 * p * r / (p + r)) if (p is not None and r is not None and p + r > 0) else (0.0 if cited_n or gold_n else None)
    out = {
        "n_cited": len(cited_n),
        "cit_precision": p,
        "cit_recall": r,
        "cit_f1": f1,
        "n_distractor_cited": len([c for c in in_ctx if c not in in_gold]),
        "n_invalid_cited": len(cited_n) - len(in_ctx),
        "styles_used": sorted(by_style),
        "wrong_style": bool(style and by_style and set(by_style) != {style}),
    }
    if ref is not None:
        out.update(n_ref_tags=ref[1], n_ref_invalid=ref[2], n_invalid_cited=ref[2],
                   # another format, or only stray / wrong-variant tags
                   wrong_style=bool(set(by_style) - {"ref"}) or (ref[2] > 0 and not ref[1]))
    return out


# ── Judge (Lucie OpenAI-compatible endpoint) ────────────────────────────────


def factual_judge_messages(prompts, question, answer, gold, supporting_facts, context):
    """Same layout as luciole_rag.build_factual_judge_messages."""
    sf_text = "\n".join(f"- {t}" for t in supporting_facts) if supporting_facts else "(none available)"
    user = (
        f"**Question:**\n{question}\n\n"
        f"**Correct answer:**\n{gold or ''}\n\n"
        f"**Supporting facts (document titles):**\n{sf_text}\n\n"
        f"**Context:**\n{context}\n\n"
        f"**Reasoning trace:**\n{answer}"
    )
    return [{"role": "system", "content": prompts["JUDGE_FACTUAL_SYSTEM_PROMPT"]},
            {"role": "user", "content": user}]


def refusal_judge_messages(prompts, question, answer):
    """Same layout as luciole_rag.build_refusal_judge_messages."""
    user = f"**Question:**\n{question}\n\n**Assistant response:**\n{answer}"
    return [{"role": "system", "content": prompts["JUDGE_REFUSAL_SYSTEM_PROMPT"]},
            {"role": "user", "content": user}]


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "oui")
    return None


def parse_factual(text):
    try:
        d = _extract_json(text)
        score = int(d["score"])
        if not 1 <= score <= 5:
            return None
        return {"score": score, "refused": bool(_as_bool(d.get("refused", False)))}
    except Exception:
        return None


def parse_refusal(text):
    try:
        v = _as_bool(_extract_json(text)["refused"])
        return None if v is None else {"refused": v}
    except Exception:
        return None


def chat_call(url: str, key: str, model: str, messages, max_tokens=512, retries=4, timeout=180, thinking=None):
    payload = {"model": model, "temperature": 0.0, "max_tokens": max_tokens, "messages": messages}
    if thinking is not None:  # Qwen-style reasoning switch
        payload["chat_template_kwargs"] = {"enable_thinking": thinking}
    body = json.dumps(payload).encode()
    for k in range(retries):
        try:
            rq = urllib.request.Request(url, data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(rq, timeout=timeout) as r:
                d = json.load(r)
            ch = d["choices"][0]
            content = ch["message"]["content"]
            if content is None:  # e.g. the reasoning used up max_tokens: say so instead of a silent null verdict
                print(f"[chat_call] empty content: finish_reason={ch.get('finish_reason')} usage={d.get('usage')}",
                      file=sys.stderr, flush=True)
            return content
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, OSError) as e:
            if k == retries - 1:
                print(f"[chat_call] failed after {retries} tries: {e!r}"[:500], file=sys.stderr, flush=True)
                return None
            time.sleep(3 * (k + 1))
    return None


class JudgePool:
    """The same judge on several servers: --judge-url "URL" or "URL,URL2@32" (an entry without @n gets --concurrency).
    Run with `size` threads: each call takes a free slot on a live server; a server whose /models stops answering is
    left aside for 5 minutes and the call goes to another one (a GPU host killed overnight costs no verdict)."""

    def __init__(self, spec: str, concurrency: int):
        self.servers = []
        for part in spec.split(","):
            u, _, n = part.strip().partition("@")
            self.servers.append({"url": u.rstrip("/"), "slots": int(n) if n else concurrency, "busy": 0, "down": 0.0})
        self.size = sum(s["slots"] for s in self.servers)
        self.cv = threading.Condition()

    def _take(self):
        with self.cv:
            while True:
                live = [s for s in self.servers if s["down"] < time.time() and s["busy"] < s["slots"]]
                if live:
                    s = max(live, key=lambda s: s["slots"] - s["busy"])
                    s["busy"] += 1
                    return s
                self.cv.wait(30)

    def _alive(self, s, key):
        try:
            rq = urllib.request.Request(s["url"] + "/models", headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(rq, timeout=10):
                return True
        except Exception:
            return False

    def call(self, key: str, model: str, messages, **kw):
        for _ in range(len(self.servers) + 1):
            s = self._take()
            try:
                raw = chat_call(s["url"] + "/chat/completions", key, model, messages, **kw)
            finally:
                with self.cv:
                    s["busy"] -= 1
                    self.cv.notify_all()
            if raw is not None or len(self.servers) == 1 or self._alive(s, key):
                return raw
            print(f"[judge-pool] {s['url']} ne répond plus : écarté 5 min", file=sys.stderr, flush=True)
            with self.cv:
                s["down"] = time.time() + 300
        return None
