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


_DEFAULT_LLM_API_URL = "https://chat.lucie.ovh.linagora.com/v1/chat/completions"
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

    formatted = "\n\n".join(f"[{c['title']}]\n{c['text']}" for c in final_chunks)
    stats = {
        "chunks_relevant": num_relevant,
        "chunks_background_initial": num_background_initial,
        "chunks_background_removed": num_background_removed,
        "chunks_background_final": num_background_initial - num_background_removed,
        "chunks_total": len(final_chunks),
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
) -> str | None:
    if not api_key:
        raise ValueError("api_key is required for LLM calls (set LLM_API_KEY in .env or environment)")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
        ],
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


def evaluate_chunk_citations(
    cited: list[str], expected: list[str]
) -> tuple[float | None, float | None]:
    """Compute precision and recall of cited chunks vs ground-truth supporting facts."""
    cited_norm = {t.strip().lower() for t in cited}
    expected_norm = {t.strip().lower() for t in expected}
    if not cited_norm and not expected_norm:
        return None, None
    correct = cited_norm & expected_norm
    precision = len(correct) / len(cited_norm) if cited_norm else None
    recall = len(correct) / len(expected_norm) if expected_norm else None
    return precision, recall


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


def filter_row(row: dict) -> bool:
    """Keep unanswerable rows and rows that pass all evaluated quality thresholds.

    Always required (when answerable):
      - eval_answer_correct must be True

    Applied only if the metric was evaluated (field present in row):
      - eval_chunk_f1 must be 1.0   (requires --eval-chunks)
      - eval_factual_judge_score must be 5  (requires --llm-judge-factual or --llm-judge-factual-openrouter)
    """
    if row.get("is_unanswerable", False):
        return True
    if not row.get("eval_answer_correct"):
        return False
    if "eval_chunk_f1" in row and row["eval_chunk_f1"] != 1.0:
        return False
    if "eval_factual_judge_score" in row and row["eval_factual_judge_score"] != 5:
        return False
    return True
