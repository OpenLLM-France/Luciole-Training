#!/usr/bin/env python3
"""Shared utilities for dataset_rag scripts."""

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp


def load_env_key(name: str, base_dir: Path | None = None) -> str | None:
    """Load an environment variable from `<base_dir>/.env` first, then process env."""
    env_path = (base_dir or Path(__file__).parent) / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get(name)


_DEFAULT_LLM_API_URL = "http://localhost:8000/v1/chat/completions"
_DEFAULT_LLM_MODEL = "Mistral-Small-3.1-24B-Instruct-2503"

LLM_API_URL = load_env_key("LLM_BASE_URL") or _DEFAULT_LLM_API_URL
LLM_MODEL = load_env_key("LLM_MODEL") or _DEFAULT_LLM_MODEL
LLM_MAX_CONCURRENT = 10


def log(message: str) -> None:
    print(message, flush=True)


@dataclass
class Metrics:
    total_rows: int = 0
    processed: int = 0
    failed: int = 0
    unanswerable_count: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def speed(self) -> float:
        return self.processed / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def done(self) -> int:
        return self.processed + self.failed

    @property
    def progress_pct(self) -> float:
        return (self.done / self.total_rows * 100.0) if self.total_rows > 0 else 0.0


def jsonl_read(path: str, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            row = line.strip()
            if not row:
                continue
            rows.append(json.loads(row))
    return rows


def jsonl_write(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_checkpoint_ids(path: str) -> set[str]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return set()
    with open(checkpoint_path, "r", encoding="utf-8") as handle:
        return set(json.load(handle))


def save_checkpoint_ids(
    processed_ids: set[str],
    path: str,
    *,
    sort_ids: bool = False,
    indent: int | None = None,
) -> None:
    payload = sorted(processed_ids) if sort_ids else list(processed_ids)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)


def format_context_chunks(
    *,
    context: dict,
    relevant_titles: set[str],
    shuffle: bool = True,
    max_remove_background_ratio: float = 0.7,
    min_chunks: int = 2,
    protected_background_titles: set[str] | None = None,
    contextual_summaries: dict[str, str] | None = None,
) -> tuple[str, dict]:
    protected_background_titles = protected_background_titles or set()
    titles = context["title"]
    sentences = context["sentences"]

    relevant_chunks: list[dict] = []
    background_chunks: list[dict] = []

    for title, sents in zip(titles, sentences):
        text = " ".join(sents).strip() if isinstance(sents, list) else str(sents).strip()
        chunk = {"title": title, "text": text}
        if title in relevant_titles:
            relevant_chunks.append(chunk)
        else:
            background_chunks.append(chunk)

    num_relevant = len(relevant_chunks)
    num_background_initial = len(background_chunks)
    num_background_removed = 0

    removable_background = [
        c for c in background_chunks if c["title"] not in protected_background_titles
    ]
    protected_background = [
        c for c in background_chunks if c["title"] in protected_background_titles
    ]

    if removable_background and max_remove_background_ratio > 0:
        actual_ratio = random.uniform(0, max_remove_background_ratio)
        num_to_remove = int(len(removable_background) * actual_ratio)
        max_removable = max(
            0, len(relevant_chunks) + len(background_chunks) - min_chunks
        )
        num_to_remove = min(num_to_remove, max_removable)
        if num_to_remove > 0:
            random.shuffle(removable_background)
            removable_background = removable_background[num_to_remove:]
            num_background_removed = num_to_remove

    final_chunks = relevant_chunks + protected_background + removable_background

    if shuffle:
        random.shuffle(final_chunks)

    # Renumber paragraph chunks sequentially so displayed order matches numbering.
    # Non-paragraph titles (e.g. "Table") are kept as-is.
    title_mapping: dict[str, str] = {}
    para_counter = 0
    for chunk in final_chunks:
        old_title = chunk["title"]
        if old_title.startswith("Paragraph"):
            para_counter += 1
            new_title = f"Paragraph {para_counter}"
        else:
            new_title = old_title
        title_mapping[old_title] = new_title
        chunk["title"] = new_title

    if contextual_summaries:
        # Map summaries to the new titles after renumbering
        mapped_summaries: dict[str, str] = {}
        for old_title, new_title in title_mapping.items():
            if old_title in contextual_summaries:
                mapped_summaries[new_title] = contextual_summaries[old_title]
        parts = []
        for c in final_chunks:
            summary = mapped_summaries.get(c["title"], "")
            parts.append(format_chunk_openrag(c["title"], c["text"], summary))
        formatted = "\n\n".join(parts)
    else:
        formatted = "\n\n".join(
            format_chunk_base(c["title"], c["text"]) for c in final_chunks
        )

    stats = {
        "chunks_relevant": num_relevant,
        "chunks_background_initial": num_background_initial,
        "chunks_background_removed": num_background_removed,
        "chunks_background_final": num_background_initial - num_background_removed,
        "chunks_total": len(final_chunks),
        "title_mapping": title_mapping,
    }
    return formatted, stats


async def post_chat_completion(
    session: aiohttp.ClientSession,
    *,
    api_url: str,
    api_key: str,
    payload: dict,
    max_retries: int = 3,
    timeout_seconds: int = 120,
    debug_http: bool = False,
) -> dict | None:
    """Post a chat completion request with retries and exponential backoff."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = aiohttp.ClientTimeout(
        total=timeout_seconds,
        connect=timeout_seconds,
        sock_connect=timeout_seconds,
        sock_read=timeout_seconds,
    )
    for attempt in range(max_retries):
        try:
            async with session.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    print(
                        f"HTTP attempt {attempt + 1}/{max_retries} failed with "
                        f"status {response.status}: {body[:500]}"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return None
                return await response.json()
        except Exception as exc:
            details = f"{type(exc).__name__}: {exc}"
            if debug_http:
                print(
                    f"HTTP attempt {attempt + 1}/{max_retries} failed: {details}"
                )
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Generation == inference prompt
# ---------------------------------------------------------------------------
#
# The reasoning traces are generated with the *same* OpenRAG system prompt the
# fine-tuned model sees at inference time, built by
# formatting_sft.build_chat_system_prompt (single source of truth). The augment
# scripts draw the citation placement (inline vs trailing References section) per
# row, build the matching prompt with the canonical markup style, and pass it
# here with ``context_in_system=True``. This removes the generation/inference
# prompt mismatch our colleagues flagged; quality is then enforced by the
# verification/filter step (factual judge + citation F1).


async def call_llm(
    session: aiohttp.ClientSession,
    question: str,
    context: str,
    semaphore: asyncio.Semaphore,
    *,
    api_key: str,
    system_prompt: str,
    model: str = LLM_MODEL,
    max_retries: int = 3,
    timeout_seconds: int = 120,
    max_tokens: int = 1024,
    debug_http: bool = False,
    context_in_system: bool = False,
) -> str | None:
    if not api_key:
        raise ValueError("api_key is required for LLM calls (set LLM_API_KEY in .env or environment)")
    if context_in_system:
        # Match inference message layout: `system_prompt` already contains the
        # retrieved documents (built by the caller, e.g. via
        # formatting_sft.build_chat_system_prompt); only the bare user query goes
        # in the user turn, exactly what the fine-tuned model sees at inference.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
        ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    async with semaphore:
        data = await post_chat_completion(
            session,
            api_url=LLM_API_URL,
            api_key=api_key,
            payload=payload,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            debug_http=debug_http,
        )
        if data is not None:
            return data["choices"][0]["message"]["content"]
    return None


# ---------------------------------------------------------------------------
# Query-grounded refusal generation
# ---------------------------------------------------------------------------
#
# For unanswerable rows we want the refusal target to be *adapted to the query*
# rather than a single canned sentence. A constant refusal makes the training
# target trivial: the model learns to map "looks unanswerable" -> a memorized
# string without attending to the actual question/documents, which encourages
# systematic over-refusal after fine-tuning. Here the LLM writes a short,
# grounded explanation (what the documents actually cover + the specific gap vs
# the question): the *body* of the trace. The canonical refusal phrase is
# appended later (at formatting time) as the closing sentence, so refusal rows
# stay structurally symmetric with answerable rows (reasoning -> conclusion).

REFUSAL_BODY_SYSTEM_PROMPT = {
    "en": (
        "The documents provided below do NOT contain the information needed to "
        "answer the user's question. Write a short, grounded explanation (2 to 4 "
        "sentences) that:\n"
        "1. States concisely what the provided documents are actually about.\n"
        "2. Explains specifically what the question asks that the documents do not cover.\n"
        "\nStrict rules:\n"
        "- Do NOT invent, guess, or state any fact as if it answered the question.\n"
        "- Do NOT write a 'Final Answer' / 'Réponse finale' line — output the explanation only.\n"
        "- Refer to the documents naturally (e.g. by their topic), keep it under 80 words.\n"
        "- {lang_directive}"
    ),
    "fr": (
        "Les documents fournis ci-dessous NE contiennent PAS les informations "
        "nécessaires pour répondre à la question de l'utilisateur. Rédige une courte "
        "explication ancrée dans les documents (2 à 4 phrases) qui :\n"
        "1. Indique de façon concise de quoi parlent réellement les documents fournis.\n"
        "2. Explique précisément ce que demande la question et que les documents ne couvrent pas.\n"
        "\nRègles strictes :\n"
        "- N'invente jamais, ne devine pas et n'énonce aucun fait comme s'il répondait à la question.\n"
        "- N'écris PAS de ligne « Réponse finale » / « Final Answer » — produis uniquement l'explication.\n"
        "- Référence les documents naturellement (par leur sujet), reste sous 80 mots.\n"
        "- {lang_directive}"
    ),
}

# Explicit output-language directive: the model ignores "same language as the
# question", so we name the target language outright (data language per config).
REFUSAL_LANG_DIRECTIVE = {
    "en": "Write your entire explanation in English.",
    "fr": "Rédige toute ton explication en français.",
    "de": "Schreibe deine gesamte Erklärung auf Deutsch.",
    "es": "Escribe toda tu explicación en español.",
    "ru": "Напиши всё объяснение на русском языке.",
}

_FINAL_ANSWER_LINE_RE = re.compile(
    r"\n*\s*\*{0,2}\s*(?:final answer|réponse finale)\s*\*{0,2}\s*:.*\Z",
    re.IGNORECASE | re.DOTALL,
)


def _strip_final_answer_line(text: str) -> str:
    """Remove a trailing 'Final Answer:' / 'Réponse finale :' block if the model added one."""
    return _FINAL_ANSWER_LINE_RE.sub("", text).strip()


async def generate_refusal_body(
    session: aiohttp.ClientSession,
    question: str,
    context: str,
    semaphore: asyncio.Semaphore,
    *,
    api_key: str,
    language: str = "en",
    model: str = LLM_MODEL,
    max_retries: int = 3,
    timeout_seconds: int = 120,
    max_tokens: int = 256,
    debug_http: bool = False,
) -> str | None:
    """Generate a query-grounded refusal reasoning body.

    Returns the explanation body (no Final Answer line) on success, or None if the
    call failed or the output was empty. Callers should fall back to a canned
    refusal phrase when None is returned.
    """
    template = REFUSAL_BODY_SYSTEM_PROMPT.get(language, REFUSAL_BODY_SYSTEM_PROMPT["en"])
    lang_directive = REFUSAL_LANG_DIRECTIVE.get(language, REFUSAL_LANG_DIRECTIVE["en"])
    system_prompt = template.format(lang_directive=lang_directive)
    body = await call_llm(
        session, question, context, semaphore,
        api_key=api_key,
        system_prompt=system_prompt,
        model=model,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        debug_http=debug_http,
    )
    if body is None:
        return None
    body = _strip_final_answer_line(body.strip())
    if not body:
        return None
    return body


# ---------------------------------------------------------------------------
# Contextual Retrieval — OpenRAG chunk contextualisation
# ---------------------------------------------------------------------------

CONTEXTUALIZER_SYSTEM_PROMPT = """\
You are an AI assistant and your task is to write a short, contextual \
summary (1-2 sentences) for the current chunk using the following elements.

- Filename: Document name and type identifying the source
- First Chunks: Opening content establishing the document's main subject/purpose
- Previous Chunks: Immediately preceding content showing continuation or transition
- Current Chunk: The content being contextualized

## Core Principles
- Analyze continuity carefully: Determine if the current chunk continues, \
concludes, or starts a new section relative to previous content
- Write the contextual summary in the same language as the current chunk
- For tables or image descriptions, focus on contextual significance \
rather than repeating descriptive details

## Output Format
- Plain text only—no markdown, bullet points, or explanatory preamble
- 1-2 concise sentences maximum
- Standalone and independently comprehensible context

## Examples
- This part of the research paper presents the methodology following \
the literature review on climate modeling.
- This section includes performance charts and KPI metrics that quantify \
the operational improvements.
- The health study continues with demographic tables and numerical outcomes \
from the clinical trial.
- This section continues the listing of recommended measures to enhance sovereignty."""


CONTEXTUALIZER_BATCH_SYSTEM_PROMPT = """\
You are an AI assistant and your task is to write a short, contextual \
summary (1-2 sentences) for EVERY chunk of a document, in a single pass.

You are given the filename and the full ordered list of chunks. Each chunk is \
introduced by a `[CHUNK n]` marker. Contextualize each chunk relative to the \
document as a whole and to the chunks that precede it.

## Core Principles
- Analyze continuity carefully: Determine if a chunk continues, concludes, \
or starts a new section relative to the preceding chunks
- Write each contextual summary in the same language as its own chunk
- For tables or image descriptions, focus on contextual significance \
rather than repeating descriptive details

## Output Format
- Reply with a single JSON object and nothing else—no markdown fences, \
no preamble, no trailing commentary
- One key per chunk: the chunk number as a string, in order
- Each value is 1-2 concise sentences of plain text, standalone and \
independently comprehensible
- Every chunk must appear exactly once, even if its content is short or \
unrelated to the rest

## Output Example
{"1": "This part of the research paper presents the methodology following the \
literature review on climate modeling.", "2": "This section includes performance \
charts and KPI metrics that quantify the operational improvements."}"""


_LANG_NAMES = {"en": "English", "fr": "French", "de": "German", "es": "Spanish", "ru": "Russian"}


def _build_contextualizer_user_prompt(
    filename: str,
    first_chunks: list[dict],
    prev_chunks: list[dict],
    current_chunk: dict,
    language: str,
) -> str:
    first_text = "\n---\n".join(c["text"] for c in first_chunks)
    prev_text = "\n---\n".join(c["text"] for c in prev_chunks) if prev_chunks else ""
    lang_name = _LANG_NAMES.get(language, language)
    return (
        f"Here is the context to consider for generating the context:\n"
        f"- Filename: {filename}\n"
        f"- First chunks:\n{first_text}\n\n"
        f"- Previous chunks:\n{prev_text}\n\n"
        f"Here is the current chunk to contextualize strictly in this {lang_name} language:\n"
        f"- Current chunk:\n\n{current_chunk['text']}"
    )


def _build_contextualizer_batch_user_prompt(
    filename: str,
    chunks: list[dict],
    language: str,
) -> str:
    lang_name = _LANG_NAMES.get(language, language)
    body = "\n\n".join(
        f"[CHUNK {i}]\n{c['text']}" for i, c in enumerate(chunks, start=1)
    )
    return (
        f"Here is the context to consider for generating the contexts:\n"
        f"- Filename: {filename}\n"
        f"- Number of chunks: {len(chunks)}\n\n"
        f"Here are the chunks to contextualize strictly in this {lang_name} language:\n\n"
        f"{body}\n\n"
        f'Reply with a JSON object holding exactly {len(chunks)} keys, "1" through '
        f'"{len(chunks)}".'
    )


async def _contextualize_chunks_batched(
    session: aiohttp.ClientSession,
    chunks: list[dict],
    semaphore: asyncio.Semaphore,
    *,
    api_key: str,
    language: str,
    filename: str,
    max_retries: int,
    timeout_seconds: int,
    debug_http: bool,
) -> dict[int, str]:
    """Summarize every chunk in one call. Returns {1-based index: summary}.

    Missing or malformed entries are simply absent from the mapping; the caller
    falls back to a per-chunk call for those.
    """
    user_prompt = _build_contextualizer_batch_user_prompt(filename, chunks, language)
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": CONTEXTUALIZER_BATCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        # ~120 tokens per summary, plus room for the JSON scaffolding.
        "max_tokens": min(4096, 120 * len(chunks) + 256),
        "response_format": {"type": "json_object"},
    }
    async with semaphore:
        data = await post_chat_completion(
            session,
            api_url=LLM_API_URL,
            api_key=api_key,
            payload=payload,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            debug_http=debug_http,
        )
    if data is None:
        return {}
    try:
        content = data["choices"][0]["message"]["content"]
        parsed = _extract_json(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return {}

    summaries: dict[int, str] = {}
    for key, value in parsed.items():
        try:
            idx = int(str(key).strip())
        except ValueError:
            continue
        if 1 <= idx <= len(chunks) and isinstance(value, str) and value.strip():
            summaries[idx] = value.strip()
    return summaries


async def contextualize_chunks(
    session: aiohttp.ClientSession,
    chunks: list[dict],
    semaphore: asyncio.Semaphore,
    *,
    api_key: str,
    language: str = "en",
    filename: str = "Document",
    max_retries: int = 2,
    timeout_seconds: int = 120,
    debug_http: bool = False,
    batched: bool = False,
) -> dict[str, str]:
    """Generate contextual summaries for ordered chunks following OpenRAG logic.

    *chunks* must be in document order: [{"title": str, "text": str}, ...].
    Returns a mapping title → contextual summary (1-2 sentences).

    By default, each chunk gets its own request containing the first two chunks,
    the two immediately preceding chunks and the current chunk, matching the
    production OpenRAG chunker. ``batched=True`` remains available as an
    explicit cheaper approximation; omitted batch entries fall back to the
    production per-chunk request.
    """
    if len(chunks) < 2:
        return {}

    first_chunks = chunks[:2]
    batch_summaries: dict[int, str] = {}
    if batched:
        batch_summaries = await _contextualize_chunks_batched(
            session, chunks, semaphore,
            api_key=api_key,
            language=language,
            filename=filename,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            debug_http=debug_http,
        )
        if len(batch_summaries) == len(chunks):
            return {c["title"]: batch_summaries[i] for i, c in enumerate(chunks, start=1)}

    async def _one(i: int) -> tuple[str, str]:
        chunk = chunks[i]
        prev = chunks[max(0, i - 2):i]
        user_prompt = _build_contextualizer_user_prompt(
            filename=filename,
            first_chunks=first_chunks,
            prev_chunks=prev,
            current_chunk=chunk,
            language=language,
        )
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": CONTEXTUALIZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 150,
        }
        try:
            async with semaphore:
                data = await post_chat_completion(
                    session,
                    api_url=LLM_API_URL,
                    api_key=api_key,
                    payload=payload,
                    max_retries=max_retries,
                    timeout_seconds=timeout_seconds,
                    debug_http=debug_http,
                )
            if data is not None:
                summary = data["choices"][0]["message"]["content"]
                if isinstance(summary, str):
                    return chunk["title"], summary.strip()
        except Exception as exc:
            if debug_http:
                print(f"  Contextualizer failed for {chunk['title']}: {exc}")
        return chunk["title"], ""

    # 0-based positions still needing an individual call (all of them when the
    # batched pass is disabled or came back empty).
    missing = [i for i in range(len(chunks)) if (i + 1) not in batch_summaries]
    results = await asyncio.gather(*[_one(i) for i in missing])

    summaries = {c["title"]: batch_summaries[i] for i, c in enumerate(chunks, start=1)
                 if i in batch_summaries}
    summaries.update(dict(results))
    return summaries


def format_chunk_base(title: str, text: str) -> str:
    """Format a single chunk in the OpenRAG base style (no contextual summary)."""
    return (
        f"* filename: {title}\n\n"
        f"[CHUNK_START]\n{text}\n[CHUNK_END]"
    )


def format_chunk_openrag(title: str, text: str, summary: str) -> str:
    """Format a single chunk in the OpenRAG contextualised style."""
    return (
        f"[CONTEXT]\n{summary}\n"
        f"* filename: {title}\n\n"
        f"[CHUNK_START]\n{text}\n[CHUNK_END]"
    )


def _extract_json(text: str) -> dict:
    """Extract a JSON object from text that may contain markdown fences or extra prose."""
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


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison (lowercase, no articles/punctuation)."""
    answer = answer.lower()
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    answer = re.sub(r"[^\w\s]", "", answer)
    return " ".join(answer.split()).strip()


def extract_answer_from_reasoning(reasoning: str) -> str | None:
    """Extract the final answer from a reasoning trace (supports EN/FR/DE/IT/ES)."""
    patterns = [
        r"\*\*(?:Final\s+)?Answer[:\*]*\**[:\s]*(.+?)(?:\n|$)",
        r"(?:Final\s+)?Answer[:\s]+(.+?)(?:\n|$)",
        r"[Tt]he (?:final )?answer is[:\s]+(.+?)(?:\.|$)",
        r"\*\*Réponse\s+finale\s*[:\*]*\**[:\s]*(.+?)(?:\n|$)",
        r"\*\*(.+?)\*\*\s*(?:\.|$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, reasoning, re.IGNORECASE | re.MULTILINE)
        if matches:
            answer = matches[-1].strip()
            answer = re.sub(r"\*+", "", answer).strip(".")
            if 0 < len(answer) < 300:
                return answer
    return None


def extract_cited_titles(reasoning: str) -> list[str]:
    """Extract chunk titles cited via ##Cite "title" ## markers."""
    titles = re.findall(r'##Cite\s*"([^"]+)"\s*##', reasoning)
    titles += re.findall(r'##Cite\s*[«»]\s*([^«»]+?)\s*[«»]\s*##', reasoning)
    if not titles:
        titles = re.findall(r'##Cite\s+([^#"«»]+?)\s*##', reasoning)
    seen: set[str] = set()
    unique: list[str] = []
    for t in titles:
        norm = t.strip().lower()
        if norm not in seen:
            seen.add(norm)
            unique.append(t.strip())
    return unique


_CHUNK_HASH_SUFFIX_RE = re.compile(r"\s*\[[0-9a-f]{12}\]\s*$")
_CHUNK_WS_RE = re.compile(r"\s+")


def _normalize_chunk_title(title: str) -> str:
    """Strip trailing ``[hash]`` suffix, lowercase, collapse whitespace."""
    t = _CHUNK_HASH_SUFFIX_RE.sub("", title).strip().lower()
    return _CHUNK_WS_RE.sub(" ", t)


def evaluate_chunk_citations(
    cited: list[str], expected: list[str]
) -> tuple[float | None, float | None]:
    """Compute precision and recall of cited chunks vs ground-truth supporting facts.

    A citation matches an expected title when one is a prefix of the other
    (after stripping the trailing ``[hash]`` suffix and normalizing). This
    accommodates models that complete the truncated filename with text from
    the chunk body instead of quoting it verbatim.
    """
    cited_norm = {_normalize_chunk_title(t) for t in cited if t.strip()}
    expected_norm = {_normalize_chunk_title(t) for t in expected if t.strip()}
    cited_norm.discard("")
    expected_norm.discard("")
    if not cited_norm and not expected_norm:
        return None, None

    def _match(a: str, b: str) -> bool:
        if a == b:
            return True
        shorter, longer = (a, b) if len(a) < len(b) else (b, a)
        return longer.startswith(shorter) and len(shorter.split()) >= 2

    matched_cited = sum(1 for c in cited_norm if any(_match(c, e) for e in expected_norm))
    matched_expected = sum(1 for e in expected_norm if any(_match(c, e) for c in cited_norm))

    precision = matched_cited / len(cited_norm) if cited_norm else None
    recall = matched_expected / len(expected_norm) if expected_norm else None
    return precision, recall


# Refusal detection patterns (multilingual)
REFUSAL_PATTERNS = [
    # English
    r"(?:do not|don't|cannot|can't|unable to) (?:allow me to |)(?:answer|provide|respond)",
    r"(?:not enough|insufficient) information",
    r"does not (?:contain|allow|provide)",
    # French
    r"ne (?:me )?permettent? pas",
    r"pas en mesure de répondre",
    r"ne (?:peux|puis) pas (?:fournir|répondre)",
    # German
    r"(?:nicht in der Lage|ermöglich(?:t|en) es mir nicht|kann .* nicht beantworten)",
    r"nicht genügend Informationen",
    # Spanish
    r"no (?:me )?permit(?:e|en) responder",
    r"no puedo (?:responder|proporcionar)",
    # Russian
    r"не позволяют .* ответить",
    r"не могу ответить",
]

_REFUSAL_RE = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]


def detect_refusal(response: str) -> bool:
    """Detect if a model response is a refusal to answer."""
    for pattern in _REFUSAL_RE:
        if pattern.search(response):
            return True
    return False


JUDGE_SYSTEM_PROMPT = """\
You are an impartial evaluator. You will be given:
1. A **question**.
2. A **context** (retrieved documents).
3. A **reasoning trace** produced by an AI assistant.

Your task is to rate the **relevance and quality of the reasoning trace** on a scale from 1 to 5:

- **1**: The reasoning is completely irrelevant, ignores the context, or is nonsensical.
- **2**: The reasoning mentions the topic but largely misuses the context or follows a flawed logic.
- **3**: The reasoning is partially relevant — some steps are correct but important information is missed or misinterpreted.
- **4**: The reasoning is mostly relevant and logical, with only minor omissions or imprecisions.
- **5**: The reasoning is fully relevant, logically sound, and makes excellent use of the provided context with perfect citation.

You MUST reply with ONLY a JSON object in this exact format (no other text):
{"score": <int>, "justification": "<one sentence>"}
"""

JUDGE_FRENCH_QUALITY_SYSTEM_PROMPT = """\
You are an expert linguist evaluating the quality of French text that was translated from English.
You will be given a **French text** (a question, a reasoning trace, or both).

Rate the **quality of the French** on a scale from 1 to 5. The scores are grouped into three quality levels:

### Bon (score 5) — Natural French, no issues
The text reads like it was written by a native French speaker. Correct grammar, natural phrasing, appropriate vocabulary, no anglicisms or calques.

### Acceptable (score 3 or 4) — Minor issues only
The text is understandable and mostly natural, with one or two MINOR issues such as:
- A missing article (e.g. "pour Union Deportiva" instead of "pour l'Union Deportiva")
- A single minor anglicism where the meaning is still clear (e.g. "score musical" instead of "bande originale")
- A single false friend that remains understandable (e.g. "incorporé" instead of "constituée en société")
- Slightly awkward but grammatically correct phrasing
Use **4** if there is only one very minor issue, **3** if there are two minor issues or one more noticeable one.

### Non acceptable (score 1 or 2) — Serious errors
The text has at least one SERIOUS error that makes it sound like bad machine translation:
- Syntactic calques: word order copied from English (e.g. "Sont-ils X et Y des exemples de..." instead of "X et Y sont-ils des exemples de...")
- Untranslated English words (e.g. "planter" instead of "planteur/propriétaire terrien")
- Multiple gender/agreement errors in the same sentence (e.g. "les plus gros pièces utilisés" instead of "les plus grandes pièces utilisées")
- Broken sentence structure requiring a complete rewrite
- Multiple anglicisms in the same sentence (e.g. "concours de talent de réalité" instead of "concours de talents de télé-réalité")
Use **2** if the text is still understandable despite the serious error(s), **1** if the text is incomprehensible or needs a complete rewrite.

CRITICAL rules — read carefully:
- Words like "réalisateur", "producteur", "société de production", "documentaire", "activiste", "universitaire" are STANDARD French words. Do NOT treat them as anglicisms.
- Proper nouns, band names, movie titles, and university names in English (e.g. "Carnegie Mellon University", "Queens of the Stone Age") are expected and must NOT be penalized.
- A question like "Quelle université est située à San Antonio ?" is CORRECT French — do not penalize it.
- Only flag issues that a native French speaker would actually notice as unnatural.

Calibration examples:

**Score 5** (bon):
"Quel artiste est originaire d'Afrique du Sud — la chanteuse Carina Round ou le groupe de rock Seether ?"
→ Natural syntax, correct grammar, no issues.

**Score 5** (bon):
"Quelle université est située à San Antonio, au Texas - Carnegie Mellon University ou Trinity University ?"
→ Perfectly correct. English university names are expected.

**Score 5** (bon):
"En quelle année le grand-père de Maria Clémentine Sobieska est-il devenu roi de Pologne et grand duc de Lituanie ?"
→ Natural phrasing, proper inversion.

**Score 4** (acceptable):
"Dans quelle ligue Borja Herrera Gonzalez joue-t-il quand il joue comme latéral gauche ou ailier pour Union Deportiva Las Palmas ?"
→ Missing article "l'" before "Union Deportiva". Otherwise natural.

**Score 3** (acceptable):
"Qui a écrit le score musical pour la première production de Horizon Pictures ?"
→ "score musical" is a calque of "musical score" (correct: "bande originale"). Otherwise fine.

**Score 2** (non acceptable):
"Sont-ils Armand Schaefer et Frederick Wiseman des exemples de réalisateurs canadiens ?"
→ "Sont-ils X et Y..." calques English "Are X and Y...". Should be "X et Y sont-ils..."

**Score 2** (non acceptable):
"Carroll, New York, est nommé d'après quel riche planter de Maryland ?"
→ "planter" is an untranslated English word.

**Score 2** (non acceptable):
"Quelle entreprise a produit les plus gros pièces d'artillerie utilisés pendant la Première Guerre mondiale ?"
→ Multiple gender errors: "gros" → "grandes", "utilisés" → "utilisées".

**Score 1** (non acceptable):
"ZeniMax Media Inc. est une entreprise médiatique américaine, développeuse duquel jeu vidéo de rôle multijoueur en ligne massivement (MMORPG), publié par Bethesda Softworks ?"
→ Broken structure, incomprehensible.

You MUST reply with ONLY a JSON object (no other text):
{"score": <int>, "justification": "<one sentence in French>"}
"""

JUDGE_FACTUAL_SYSTEM_PROMPT = """\
You are an impartial factual evaluator. You will be given:
1. A **question**.
2. A **correct answer** (ground truth).
3. The **supporting facts** (the specific document titles that contain the evidence needed to answer the question).
4. A **context** (retrieved documents).
5. A **reasoning trace** produced by an AI assistant.

Your task is to rate the **factual correctness and faithfulness** of the reasoning trace on a scale from 1 to 5:

- **1**: The final answer is wrong AND the reasoning does not use the correct supporting facts at all.
- **2**: The final answer is wrong, but the reasoning references some of the correct supporting facts; OR the answer is partially right but the reasoning is based on wrong evidence.
- **3**: The final answer is approximately correct but imprecise, or the reasoning misses one of the key supporting facts, or the reasoning contains a factual error despite reaching the right answer.
- **4**: The final answer is correct and the reasoning uses most of the supporting facts properly, with only minor omissions or imprecisions.
- **5**: The final answer is correct, the reasoning correctly identifies and uses all the supporting facts, and the logical chain from evidence to answer is flawless.

You MUST reply with ONLY a JSON object in this exact format (no other text):
{"score": <int>, "justification": "<one sentence>"}
"""

JUDGE_REFUSAL_SYSTEM_PROMPT = """\
You decide whether an AI assistant REFUSED to answer or ATTEMPTED an answer.

The assistant was given a question and a set of retrieved documents that do NOT contain the information needed to answer it. The correct behaviour is to refuse: to state that the retrieved documents do not allow it to answer (typically inviting the user to rephrase the query or add relevant documents).

Classify the assistant's response:
- "refused": true  — the response declines to answer on the grounds that the documents are insufficient / do not contain the needed information. It does NOT provide a substantive factual answer to the question. (It may still describe what the documents DO cover, as long as it does not answer the actual question.)
- "refused": false — the response provides or attempts a substantive answer to the question, asserts a fact as the answer, or otherwise behaves as if the documents were sufficient.

Judge only the behaviour (refuse vs answer), not the wording, politeness or formatting.

You MUST reply with ONLY a JSON object in this exact format (no other text):
{"refused": <true|false>, "justification": "<one short sentence>"}
"""


def _build_judge_user_prompt(question: str, context: str, reasoning: str) -> str:
    return (
        f"**Question:**\n{question}\n\n"
        f"**Context:**\n{context}\n\n"
        f"**Reasoning trace:**\n{reasoning}"
    )


def _build_factual_judge_user_prompt(
    question: str, correct_answer: str, supporting_facts: list[str],
    context: str, reasoning: str,
) -> str:
    sf_text = "\n".join(f"- {t}" for t in supporting_facts) if supporting_facts else "(none available)"
    return (
        f"**Question:**\n{question}\n\n"
        f"**Correct answer:**\n{correct_answer}\n\n"
        f"**Supporting facts (document titles):**\n{sf_text}\n\n"
        f"**Context:**\n{context}\n\n"
        f"**Reasoning trace:**\n{reasoning}"
    )


async def _call_judge(
    session: aiohttp.ClientSession,
    system_prompt: str,
    user_content: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
    *,
    api_key: str,
    api_url: str = LLM_API_URL,
    model: str = LLM_MODEL,
    max_retries: int = 3,
) -> dict | None:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": 512,
    }
    async with semaphore:
        result = await post_chat_completion(
            session,
            api_url=api_url,
            api_key=api_key,
            payload=payload,
            max_retries=max_retries,
            timeout_seconds=120,
        )
        if result is None:
            return None
        try:
            content = result["choices"][0]["message"]["content"].strip()
            parsed = _extract_json(content)
            score = int(parsed["score"])
            if score < 1 or score > 5:
                raise ValueError(f"Score {score} out of range 1-5")
            return {"score": score, "justification": parsed.get("justification", "")}
        except Exception as exc:
            print(f"  Judge response parsing failed: {exc}")
    return None


def _build_french_quality_judge_user_prompt(reasoning: str) -> str:
    return f"**Reasoning trace (in French):**\n{reasoning}"


async def run_french_quality_judge_ordered(
    rows: list[dict],
    *,
    api_key: str,
    api_url: str = LLM_API_URL,
    model: str = LLM_MODEL,
    desc: str = "LLM French quality judge",
) -> list[dict | None]:
    semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENT)
    async with aiohttp.ClientSession() as session:
        from tqdm.asyncio import tqdm as atqdm
        tasks = [
            _call_judge(
                session,
                JUDGE_FRENCH_QUALITY_SYSTEM_PROMPT,
                _build_french_quality_judge_user_prompt(
                    row.get("reasoning_trace", ""),
                ),
                0.0,
                semaphore,
                api_key=api_key,
                api_url=api_url,
                model=model,
            )
            for row in rows
        ]
        return await atqdm.gather(*tasks, desc=desc)


async def run_judge_ordered(
    rows: list[dict],
    *,
    api_key: str,
    api_url: str = LLM_API_URL,
    model: str = LLM_MODEL,
    desc: str = "LLM judge",
) -> list[dict | None]:
    semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENT)
    async with aiohttp.ClientSession() as session:
        from tqdm.asyncio import tqdm as atqdm
        tasks = [
            _call_judge(
                session,
                JUDGE_SYSTEM_PROMPT,
                _build_judge_user_prompt(
                    row.get("question", ""),
                    row.get("context", ""),
                    row.get("reasoning_trace", ""),
                ),
                0.0,
                semaphore,
                api_key=api_key,
                api_url=api_url,
                model=model,
            )
            for row in rows
        ]
        return await atqdm.gather(*tasks, desc=desc)


async def run_factual_judge_ordered(
    rows: list[dict],
    sf_lookup: dict[str, set[str]],
    *,
    api_key: str,
    api_url: str = LLM_API_URL,
    model: str = LLM_MODEL,
    desc: str = "LLM factual judge",
) -> list[dict | None]:
    semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENT)
    async with aiohttp.ClientSession() as session:
        from tqdm.asyncio import tqdm as atqdm
        tasks = []
        for row in rows:
            base_id = row.get("id", "").removesuffix("_unanswerable")
            sf_titles = row["supporting_facts_titles"] if "supporting_facts_titles" in row \
                else list(sf_lookup.get(base_id, []))
            tasks.append(
                _call_judge(
                    session,
                    JUDGE_FACTUAL_SYSTEM_PROMPT,
                    _build_factual_judge_user_prompt(
                        row.get("question", ""),
                        row.get("answer", ""),
                        sf_titles,
                        row.get("context", ""),
                        row.get("reasoning_trace", ""),
                    ),
                    0.1,
                    semaphore,
                    api_key=api_key,
                    api_url=api_url,
                    model=model,
                )
            )
        return await atqdm.gather(*tasks, desc=desc)


def run_llm_judge(
    rows: list[dict],
    *,
    api_key: str,
    api_url: str = LLM_API_URL,
    model: str = LLM_MODEL,
    desc: str = "LLM judge",
) -> list[dict | None]:
    """Synchronous wrapper: run the relevance LLM judge on a list of rows."""
    if not api_key:
        raise ValueError("api_key is required for LLM judge (set LLM_API_KEY in .env or environment)")
    print(f"\nRunning LLM-as-a-judge ({model}) on {len(rows)} rows...")
    return asyncio.run(run_judge_ordered(rows, api_key=api_key, api_url=api_url, model=model, desc=desc))


def run_factual_judge(
    rows: list[dict],
    sf_lookup: dict[str, set[str]],
    *,
    api_key: str,
    api_url: str = LLM_API_URL,
    model: str = LLM_MODEL,
    desc: str = "LLM factual judge",
) -> list[dict | None]:
    """Synchronous wrapper: run the factual LLM judge on a list of rows."""
    if not api_key:
        raise ValueError("api_key is required for LLM judge (set LLM_API_KEY in .env or environment)")
    print(f"\nRunning factual LLM-as-a-judge ({model}) on {len(rows)} rows...")
    return asyncio.run(run_factual_judge_ordered(rows, sf_lookup, api_key=api_key, api_url=api_url, model=model, desc=desc))


def run_french_quality_judge(
    rows: list[dict],
    *,
    api_key: str,
    api_url: str = LLM_API_URL,
    model: str = LLM_MODEL,
    desc: str = "LLM French quality judge",
) -> list[dict | None]:
    """Synchronous wrapper: run the French quality LLM judge on a list of rows."""
    if not api_key:
        raise ValueError("api_key is required for LLM judge (set LLM_API_KEY in .env or environment)")
    print(f"\nRunning French quality LLM-as-a-judge ({model}) on {len(rows)} rows...")
    return asyncio.run(run_french_quality_judge_ordered(rows, api_key=api_key, api_url=api_url, model=model, desc=desc))


def _build_refusal_judge_user_prompt(question: str, context: str, reasoning: str) -> str:
    return (
        f"**Question:**\n{question}\n\n"
        f"**Retrieved documents:**\n{context}\n\n"
        f"**Assistant response:**\n{reasoning}"
    )


async def _call_refusal_judge(
    session: aiohttp.ClientSession,
    user_content: str,
    semaphore: asyncio.Semaphore,
    *,
    api_key: str,
    api_url: str = LLM_API_URL,
    model: str = LLM_MODEL,
    max_retries: int = 3,
) -> dict | None:
    """Binary refusal judge: returns {"refused": bool, "justification": str} or None."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_REFUSAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    async with semaphore:
        result = await post_chat_completion(
            session, api_url=api_url, api_key=api_key, payload=payload,
            max_retries=max_retries, timeout_seconds=120,
        )
        if result is None:
            return None
        try:
            content = result["choices"][0]["message"]["content"].strip()
            parsed = _extract_json(content)
            refused = parsed["refused"]
            if isinstance(refused, str):
                refused = refused.strip().lower() in ("true", "yes", "1")
            return {"refused": bool(refused), "justification": parsed.get("justification", "")}
        except Exception as exc:
            print(f"  Refusal judge response parsing failed: {exc}")
    return None


async def run_refusal_judge_ordered(
    rows: list[dict],
    *,
    api_key: str,
    api_url: str = LLM_API_URL,
    model: str = LLM_MODEL,
    desc: str = "LLM refusal judge",
    concurrency: int = LLM_MAX_CONCURRENT,
) -> list[dict | None]:
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        from tqdm.asyncio import tqdm as atqdm
        tasks = [
            _call_refusal_judge(
                session,
                _build_refusal_judge_user_prompt(
                    row.get("question", ""),
                    row.get("context", ""),
                    row.get("reasoning_trace", ""),
                ),
                semaphore,
                api_key=api_key,
                api_url=api_url,
                model=model,
            )
            for row in rows
        ]
        return await atqdm.gather(*tasks, desc=desc)


def run_refusal_judge(
    rows: list[dict],
    *,
    api_key: str,
    api_url: str = LLM_API_URL,
    model: str = LLM_MODEL,
    desc: str = "LLM refusal judge",
    concurrency: int = LLM_MAX_CONCURRENT,
) -> list[dict | None]:
    """Synchronous wrapper: run the binary refusal judge on a list of (unanswerable) rows."""
    if not api_key:
        raise ValueError("api_key is required for LLM judge (set LLM_API_KEY in .env or environment)")
    print(f"\nRunning refusal LLM-as-a-judge ({model}) on {len(rows)} rows...")
    return asyncio.run(run_refusal_judge_ordered(rows, api_key=api_key, api_url=api_url, model=model, desc=desc, concurrency=concurrency))


def filter_row(row: dict, *, french_quality_threshold: int = 5, require_answer_match: bool = True) -> bool:
    """Keep unanswerable rows and rows that pass all evaluated quality thresholds.

    Always required (when answerable), unless ``require_answer_match`` is False:
      - eval_answer_correct must be True

    ``require_answer_match=False`` is used when the traces carry no "Final Answer"
    line to extract (the unified-prompt format): correctness is then gated purely
    by the factual judge (+ citation F1) below, not by exact string match.

    Applied only if the metric was evaluated (field present in row):
      - eval_chunk_f1 must be 1.0   (requires --eval-chunks)
      - eval_factual_judge_score must be 5  (requires --llm-judge-factual or --llm-judge-factual-openrouter)
      - eval_french_quality_judge_score must be >= french_quality_threshold, default 5 (requires --llm-judge-french-quality)

    Unanswerable rows are kept only if the model actually refused
    (``eval_model_refused`` is True); rows where it hallucinated an answer on the
    insufficient context are dropped. Legacy rows without the field are kept.
    """
    if row.get("is_unanswerable", False):
        return row.get("eval_model_refused", True) is not False
    if require_answer_match and not row.get("eval_answer_correct"):
        return False
    if "eval_chunk_f1" in row and row["eval_chunk_f1"] != 1.0:
        return False
    if "eval_factual_judge_score" in row and row["eval_factual_judge_score"] != 5:
        return False
    if "eval_french_quality_judge_score" in row and (row["eval_french_quality_judge_score"] or 0) < french_quality_threshold:
        return False
    return True


UNANSWERABLE_ID_SUFFIX = "_unanswerable"


def pair_base_id(row_id: object) -> str:
    """Question id shared by an answerable row and its unanswerable twin."""
    text = str(row_id)
    if text.endswith(UNANSWERABLE_ID_SUFFIX):
        return text[: -len(UNANSWERABLE_ID_SUFFIX)]
    return text


def filter_rows(rows: list[dict], *, keep_pairs: bool = True, **kwargs) -> tuple[list[dict], dict]:
    """Apply ``filter_row`` to every row, optionally keeping twins together.

    An unanswerable row is a clone of an answerable one with the gold evidence
    removed: same question, same distractors. The two form a minimal pair whose
    only difference is what the refusal is supposed to depend on. Filtering them
    independently breaks most pairs: the answerable side faces a far stricter
    gate (judge score, citation F1) than the refusal side, so the dataset pays
    the cost of the duplication while keeping only a minority of the contrast.

    With ``keep_pairs`` a pair survives whole or not at all. Rows whose twin is
    absent from the input are unaffected: pairing is only enforced where it was
    actually built.
    """
    kept = [row for row in rows if filter_row(row, **kwargs)]
    stats = {"kept_independently": len(kept), "pairs_dropped": 0, "rows_dropped_by_pairing": 0}
    if not keep_pairs:
        return kept, stats

    def sides(subset: list[dict]) -> dict[str, set[bool]]:
        seen: dict[str, set[bool]] = {}
        for row in subset:
            seen.setdefault(pair_base_id(row.get("id", "")), set()).add(
                bool(row.get("is_unanswerable", False))
            )
        return seen

    paired = sides(rows)
    survived = sides(kept)
    result = [
        row
        for row in kept
        if len(paired[pair_base_id(row.get("id", ""))]) < 2
        or len(survived[pair_base_id(row.get("id", ""))]) == 2
    ]
    stats["rows_dropped_by_pairing"] = len(kept) - len(result)
    stats["pairs_dropped"] = sum(
        1
        for base, present in paired.items()
        if len(present) == 2 and len(survived.get(base, set())) == 1
    )
    return result, stats
