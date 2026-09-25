"""Shared helpers for the on-policy RAG DPO pipeline (self-contained, no repo imports).

Everything here must run on the DGX with only the stdlib, so the pipeline does not
depend on the dataset_rag package layout.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request

# ── IO ──────────────────────────────────────────────────────────────────────


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


# ── Context & citations ─────────────────────────────────────────────────────

_FILENAME_RE = re.compile(r"^\* filename: (.+?)\s*$", re.M)

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
    marker = "Here are the retrieved documents"
    i = system_prompt.find(marker)
    return system_prompt[i:] if i >= 0 else system_prompt


def context_titles(system_prompt: str) -> list[str]:
    return _FILENAME_RE.findall(system_prompt)


def requested_citation_style(system_prompt: str) -> tuple[str | None, str]:
    """(style name, placement) requested by the system prompt's citation rule."""
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
                     style: str | None) -> dict:
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
    return {
        "n_cited": len(cited_n),
        "cit_precision": p,
        "cit_recall": r,
        "cit_f1": f1,
        "n_distractor_cited": len([c for c in in_ctx if c not in in_gold]),
        "n_invalid_cited": len(cited_n) - len(in_ctx),
        "styles_used": sorted(by_style),
        "wrong_style": bool(style and by_style and set(by_style) != {style}),
    }


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
