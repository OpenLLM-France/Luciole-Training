#!/usr/bin/env python3
"""Shared BM25 distractor selection helpers.

Dataset-specific converters are still responsible for building gold context and
candidate pools. This module only owns the common retrieval policy:
dedup/exclude known gold chunks, optionally exclude source documents, filter
exact answer leaks, and select ranked BM25 candidates.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, Callable

from rank_bm25 import BM25Okapi

LEAK_MIN_LENGTH = 4
_WS_RE = re.compile(r"\s+")


def content_hash(text: str) -> str:
    """Stable short hash used in generated chunk titles and deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def normalize_for_leak_check(text: object) -> str:
    """Normalize text for conservative exact-substring leakage checks."""
    return _WS_RE.sub(" ", str(text).lower().strip())


def build_bm25_index(pool: Sequence[dict[str, Any]]) -> BM25Okapi:
    """Build a BM25 index over a pool of dict candidates with a ``text`` field."""
    tokenized = [str(candidate.get("text", "")).lower().split() for candidate in pool]
    return BM25Okapi(tokenized)


def _normalized_answer_strings(
    answer_strings: Iterable[object] | None,
    *,
    min_length: int = LEAK_MIN_LENGTH,
) -> list[str]:
    if not answer_strings:
        return []
    normalized: list[str] = []
    for raw in answer_strings:
        norm = normalize_for_leak_check(raw)
        if len(norm) >= min_length:
            normalized.append(norm)
    return normalized


def select_bm25_distractors(
    *,
    question: str,
    pool: Sequence[dict[str, Any]],
    bm25: Any,
    num_distractors: int,
    exclude_hashes: set[str] | None = None,
    exclude_doc_ids: set[str] | None = None,
    start_rank: int = 1,
    answer_strings: Iterable[object] | None = None,
    stats: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Select top BM25 distractors from a candidate pool.

    ``start_rank`` is 1-based and is applied after hard exclusions and answer
    leakage filtering. For example, ``start_rank=10`` skips the first nine valid
    candidates and starts from the tenth.
    """
    if num_distractors <= 0 or not pool:
        return []
    if start_rank < 1:
        raise ValueError("start_rank must be >= 1")

    exclude_hashes = {str(value) for value in (exclude_hashes or set())}
    exclude_doc_ids = {str(value) for value in (exclude_doc_ids or set())}
    norm_answers = _normalized_answer_strings(answer_strings)

    scores = bm25.get_scores(str(question).lower().split())
    ranked = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)

    distractors: list[dict[str, Any]] = []
    valid_rank = 0
    for idx in ranked:
        candidate = pool[idx]

        if str(candidate.get("hash", "")) in exclude_hashes:
            continue
        if str(candidate.get("doc_id", "")) in exclude_doc_ids:
            continue

        if norm_answers:
            norm_text = normalize_for_leak_check(candidate.get("text", ""))
            if any(answer in norm_text for answer in norm_answers):
                if stats is not None:
                    stats["leak_filtered"] = stats.get("leak_filtered", 0) + 1
                continue

        valid_rank += 1
        if valid_rank < start_rank:
            continue

        distractors.append(candidate)
        if len(distractors) >= num_distractors:
            break

    return distractors


# ---------------------------------------------------------------------------
# Mixed-difficulty selection (hard / medium / random buckets)
# ---------------------------------------------------------------------------
#
# BM25-from-rank-N distractors are lexically similar but trivially rejectable
# by a strong model. The mixed selector instead samples three buckets per row:
#   hard: top of the fused ranking (ranks 1..hard_pool), where the old
#            start_rank skip used to throw them away. Label safety comes from
#            an LLM verification pass ("does this passage answer the
#            question?") instead of the rank skip; candidates the LLM says
#            answer the question are rejected and backfilled from medium.
#   medium: next ranks, protected by the lexical answer-leak filter.
#   random: uniform draws from the remaining pool, so the model still sees
#            plainly irrelevant passages (matches real top-k retrieval noise).
# Every selected candidate is tagged with a `difficulty` field for bucketed
# evaluation and mix control downstream.

DIFFICULTY_HARD = "hard"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_RANDOM = "random"

_MIX_SPEC_RE = re.compile(r"^\s*(hard|medium|random)\s*=\s*(\d+)\s*$")


def parse_mix_spec(spec: str) -> dict[str, int]:
    """Parse a distractor mix spec like ``"hard=2,medium=5,random=2"``."""
    mix = {DIFFICULTY_HARD: 0, DIFFICULTY_MEDIUM: 0, DIFFICULTY_RANDOM: 0}
    for part in spec.split(","):
        if not part.strip():
            continue
        m = _MIX_SPEC_RE.match(part)
        if not m:
            raise ValueError(
                f"Invalid mix component {part!r} (expected e.g. 'hard=2,medium=5,random=2')"
            )
        mix[m.group(1)] = int(m.group(2))
    if sum(mix.values()) <= 0:
        raise ValueError("Distractor mix must select at least one passage")
    return mix


class DenseRetriever:
    """Optional dense scorer over the candidate pool (sentence-transformers).

    E5-family models get the required ``query:`` / ``passage:`` prefixes.
    Instantiating this class requires the ``sentence-transformers`` package.
    """

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 64):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Dense retrieval requires sentence-transformers "
                "(pip install sentence-transformers)"
            ) from exc
        self._is_e5 = "e5" in model_name.lower()
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self._pool_embeddings = None

    def index_pool(self, pool: Sequence[dict[str, Any]]) -> None:
        texts = [str(candidate.get("text", "")) for candidate in pool]
        if self._is_e5:
            texts = [f"passage: {t}" for t in texts]
        self._pool_embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def get_scores(self, query: str) -> Any:
        if self._pool_embeddings is None:
            raise RuntimeError("index_pool() must be called before get_scores()")
        if self._is_e5:
            query = f"query: {query}"
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        return self._pool_embeddings @ query_embedding


def _top_indices(scores: Sequence[float], top_n: int) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return order[:top_n]


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = 60) -> list[int]:
    """Reciprocal-rank fusion of several ranked index lists."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] += 1.0 / (k + rank + 1)
    return sorted(fused, key=lambda i: (-fused[i], i))


class MixedDistractorSelector:
    """Two-phase mixed-difficulty distractor selection over a shared pool.

    Usage: call :meth:`prepare` once per row, optionally run the LLM
    verification pass over :meth:`pending_verifications` (feeding results back
    via :meth:`set_verdicts`), then call :meth:`finalize` per row.
    """

    def __init__(
        self,
        *,
        pool: Sequence[dict[str, Any]],
        bm25: Any,
        mix: dict[str, int],
        dense: DenseRetriever | None = None,
        use_evidence_query: bool = False,
        verify_hard: bool = True,
        hard_pool_size: int | None = None,
        seed: int | None = None,
        top_n: int = 300,
        rrf_k: int = 60,
        evidence_max_chars: int = 800,
    ):
        self.pool = pool
        self.bm25 = bm25
        self.mix = {
            DIFFICULTY_HARD: int(mix.get(DIFFICULTY_HARD, 0)),
            DIFFICULTY_MEDIUM: int(mix.get(DIFFICULTY_MEDIUM, 0)),
            DIFFICULTY_RANDOM: int(mix.get(DIFFICULTY_RANDOM, 0)),
        }
        self.num_distractors = sum(self.mix.values())
        self.dense = dense
        self.use_evidence_query = use_evidence_query
        self.verify_hard = verify_hard
        self.hard_pool_size = hard_pool_size or max(2 * self.mix[DIFFICULTY_HARD], 6)
        self.rng = random.Random(seed)
        self.top_n = top_n
        self.rrf_k = rrf_k
        self.evidence_max_chars = evidence_max_chars
        self._rows: dict[Any, dict[str, Any]] = {}
        self.stats: dict[str, int] = defaultdict(int)

    # -- phase 1 -----------------------------------------------------------

    def _ranked_indices(self, question: str, evidence: str | None) -> list[int]:
        rankings: list[list[int]] = []
        question_tokens = str(question).lower().split()
        rankings.append(_top_indices(self.bm25.get_scores(question_tokens), self.top_n))
        queries: list[str] = []
        if self.use_evidence_query and evidence:
            queries.append(str(evidence)[: self.evidence_max_chars])
        for extra_query in queries:
            rankings.append(
                _top_indices(self.bm25.get_scores(extra_query.lower().split()), self.top_n)
            )
        if self.dense is not None:
            rankings.append(_top_indices(self.dense.get_scores(str(question)), self.top_n))
            for extra_query in queries:
                rankings.append(_top_indices(self.dense.get_scores(extra_query), self.top_n))
        if len(rankings) == 1:
            return rankings[0]
        return rrf_fuse(rankings, k=self.rrf_k)

    def prepare(
        self,
        key: Any,
        *,
        question: str,
        evidence: str | None = None,
        exclude_hashes: set[str] | None = None,
        exclude_doc_ids: set[str] | None = None,
        answer_strings: Iterable[object] | None = None,
    ) -> None:
        """Rank candidates for one row and stage the three buckets."""
        exclude_hashes = {str(v) for v in (exclude_hashes or set())}
        exclude_doc_ids = {str(v) for v in (exclude_doc_ids or set())}
        norm_answers = _normalized_answer_strings(answer_strings)

        def excluded(candidate: dict[str, Any]) -> bool:
            return (
                str(candidate.get("hash", "")) in exclude_hashes
                or str(candidate.get("doc_id", "")) in exclude_doc_ids
            )

        def lexical_leak(candidate: dict[str, Any]) -> bool:
            if not norm_answers:
                return False
            norm_text = normalize_for_leak_check(candidate.get("text", ""))
            return any(answer in norm_text for answer in norm_answers)

        valid = [
            self.pool[i]
            for i in self._ranked_indices(question, evidence)
            if not excluded(self.pool[i])
        ]

        # Hard bucket: top of the ranking, NO lexical leak filter when the LLM
        # verifies (the lexical filter over-drops short/numeric answers, which
        # are exactly the hardest confusable candidates). Without verification
        # the lexical filter is the only guard, so it stays on.
        hard_candidates: list[dict[str, Any]] = []
        if self.mix[DIFFICULTY_HARD] > 0:
            for candidate in valid[: self.hard_pool_size]:
                if not self.verify_hard and lexical_leak(candidate):
                    self.stats["hard_lexical_leak_filtered"] += 1
                    continue
                hard_candidates.append(candidate)

        medium_candidates: list[dict[str, Any]] = []
        random_pool: list[dict[str, Any]] = []
        for candidate in valid[self.hard_pool_size :]:
            if lexical_leak(candidate):
                self.stats["leak_filtered"] += 1
                continue
            if len(medium_candidates) < self.mix[DIFFICULTY_MEDIUM] + self.mix[DIFFICULTY_HARD]:
                medium_candidates.append(candidate)
            else:
                random_pool.append(candidate)

        self._rows[key] = {
            "question": str(question),
            "hard": hard_candidates,
            "medium": medium_candidates,
            "random_pool": random_pool,
            "verdicts": None,
            "verified": False,
        }

    # -- phase 2 -----------------------------------------------------------

    def pending_verifications(self) -> list[tuple[Any, str, list[str]]]:
        """Rows awaiting LLM verification: (key, question, candidate texts)."""
        if not self.verify_hard:
            return []
        return [
            (key, state["question"], [str(c.get("text", "")) for c in state["hard"]])
            for key, state in self._rows.items()
            if state["hard"] and not state["verified"]
        ]

    def set_verdicts(self, key: Any, verdicts: list[bool] | None) -> None:
        """Record verification results. ``True`` means "answers the question"
        (candidate rejected). ``None`` marks a failed call: all hard candidates
        for the row are conservatively rejected and backfilled from medium."""
        state = self._rows[key]
        state["verdicts"] = verdicts
        state["verified"] = True

    # -- phase 3 -----------------------------------------------------------

    def finalize(self, key: Any) -> list[dict[str, Any]]:
        """Assemble the final tagged selection for one row."""
        state = self._rows.pop(key)
        selected: list[dict[str, Any]] = []

        hard_quota = self.mix[DIFFICULTY_HARD]
        if hard_quota > 0 and state["hard"]:
            if self.verify_hard:
                verdicts = state["verdicts"]
                if verdicts is None:
                    self.stats["hard_rows_unverified"] += 1
                    survivors: list[dict[str, Any]] = []
                else:
                    survivors = [
                        candidate
                        for candidate, answers in zip(state["hard"], verdicts)
                        if not answers
                    ]
                    self.stats["hard_verified_rejected"] += len(state["hard"]) - len(survivors)
            else:
                survivors = state["hard"]
            for candidate in survivors[:hard_quota]:
                selected.append({**candidate, "difficulty": DIFFICULTY_HARD})

        self.stats["hard_selected"] += len(selected)
        # Medium bucket absorbs any hard deficit.
        medium_quota = self.mix[DIFFICULTY_MEDIUM] + (hard_quota - len(selected))
        medium_taken = state["medium"][:medium_quota]
        selected.extend({**c, "difficulty": DIFFICULTY_MEDIUM} for c in medium_taken)
        self.stats["medium_selected"] += len(medium_taken)

        random_quota = self.mix[DIFFICULTY_RANDOM] + (self.num_distractors - len(selected) - self.mix[DIFFICULTY_RANDOM])
        random_quota = max(0, min(random_quota, self.num_distractors - len(selected)))
        random_pool = state["medium"][len(medium_taken):] if not state["random_pool"] else state["random_pool"]
        if random_quota > 0 and random_pool:
            taken = self.rng.sample(random_pool, min(random_quota, len(random_pool)))
            selected.extend({**c, "difficulty": DIFFICULTY_RANDOM} for c in taken)
            self.stats["random_selected"] += len(taken)

        if len(selected) < self.num_distractors:
            self.stats["short_rows"] += 1
        self.stats["added"] += len(selected)
        return selected


# ---------------------------------------------------------------------------
# LLM verification of hard candidates
# ---------------------------------------------------------------------------
#
# One call per row: the model sees the question and the numbered hard
# candidates, and returns a JSON array of booleans (true = the passage answers
# the question or reveals the answer). Parsing is conservative: any failure
# rejects the whole hard bucket for that row (backfilled from medium), so LLM
# flakiness can never inject an answer-leaking "distractor".

VERIFICATION_SYSTEM_PROMPT = (
    "You are a strict annotator building retrieval training data. You are given "
    "a question and numbered passages. For EACH passage decide whether it "
    "contains enough information to answer the question, or directly reveals "
    "the answer (even partially rephrased). Passages that are merely on the "
    "same topic but do not answer the question are false.\n"
    "Reply with ONLY a JSON array of booleans, one per passage, in order. "
    "No explanation."
)

_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


def build_verification_messages(
    question: str, texts: Sequence[str], *, max_chars: int = 1500
) -> list[dict[str, str]]:
    numbered = "\n\n".join(f"[{i + 1}] {str(t)[:max_chars]}" for i, t in enumerate(texts))
    user = (
        f"Question: {question}\n\nPassages:\n{numbered}\n\n"
        f"JSON array of exactly {len(texts)} booleans:"
    )
    return [
        {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_verification_verdicts(content: str | None, expected: int) -> list[bool] | None:
    """Extract a boolean array from the model reply; None on any mismatch."""
    if not content:
        return None
    match = _JSON_ARRAY_RE.search(content)
    if not match:
        return None
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, list) or len(raw) != expected:
        return None
    verdicts: list[bool] = []
    for value in raw:
        if isinstance(value, bool):
            verdicts.append(value)
        elif isinstance(value, (int, float)) and value in (0, 1):
            verdicts.append(bool(value))
        elif isinstance(value, str) and value.strip().lower() in ("true", "false", "yes", "no"):
            verdicts.append(value.strip().lower() in ("true", "yes"))
        else:
            return None
    return verdicts


def verification_cache_key(question: str, texts: Sequence[str]) -> str:
    """Content hash of one verification input (question + hard candidate texts).

    Keys the resumable checkpoint on the *content* verified, not the row id, so a
    cached verdict is reused only when the exact same candidates come back; the
    checkpoint stays correct across seed/config changes that reshuffle candidates.
    """
    h = hashlib.md5()
    h.update(str(question).encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(str(t).encode("utf-8"))
    return h.hexdigest()


def _load_verification_checkpoint(path: str) -> dict[str, list[bool] | None]:
    """Read a JSONL verdict checkpoint into {content_hash: verdicts}."""
    cache: dict[str, list[bool] | None] = {}
    if not path or not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from a hard kill
            if isinstance(rec, dict) and "h" in rec:
                cache[rec["h"]] = rec.get("v")
    return cache


async def verify_hard_candidates(
    selector: MixedDistractorSelector,
    *,
    api_key: str,
    model: str | None = None,
    api_url: str | None = None,
    concurrency: int = 10,
    max_retries: int = 3,
    timeout_seconds: int = 60,
    max_tokens: int = 128,
    debug_http: bool = False,
    checkpoint_path: str | None = None,
    log: Callable[[str], None] = print,
) -> None:
    """Run the LLM verification pass over every pending row of the selector.

    When *checkpoint_path* is given, each verdict is appended to a JSONL file as
    soon as it is computed and reused on a later run, so a killed conversion (the
    verify pass is otherwise one unresumable multi-hour block) restarts only the
    rows it never reached.
    """
    import asyncio

    import aiohttp

    from utils import LLM_API_URL, LLM_MODEL, post_chat_completion

    pending = selector.pending_verifications()
    if not pending:
        return
    if not api_key:
        raise ValueError("api_key is required for --verify-hard (set LLM_API_KEY in .env)")
    model = model or LLM_MODEL
    api_url = api_url or LLM_API_URL

    # Resume: apply cached verdicts, keep only rows never verified before.
    cache = _load_verification_checkpoint(checkpoint_path) if checkpoint_path else {}
    todo: list[tuple[Any, str, list[str], str]] = []
    reused = 0
    for key, question, texts in pending:
        h = verification_cache_key(question, texts)
        if h in cache:
            selector.set_verdicts(key, cache[h])
            reused += 1
        else:
            todo.append((key, question, texts, h))

    if reused:
        log(f"Resuming verify-hard: {reused} verdict(s) reused from checkpoint, "
            f"{len(todo)} row(s) left to verify.")
    if not todo:
        log(f"Verification done (all {reused} rows served from checkpoint).")
        return

    semaphore = asyncio.Semaphore(concurrency)
    done = 0
    # Append mode so a resumed run keeps prior verdicts; each write is a single
    # synchronous line between awaits, so no lock is needed under asyncio.
    ckpt_file = open(checkpoint_path, "a", encoding="utf-8") if checkpoint_path else None

    log(f"Verifying hard distractor candidates for {len(todo)} rows (LLM: {model})...")

    try:
        async with aiohttp.ClientSession() as session:
            async def verify_one(key: Any, question: str, texts: list[str], h: str) -> None:
                nonlocal done
                payload = {
                    "model": model,
                    "messages": build_verification_messages(question, texts),
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                }
                async with semaphore:
                    data = await post_chat_completion(
                        session,
                        api_url=api_url,
                        api_key=api_key,
                        payload=payload,
                        max_retries=max_retries,
                        timeout_seconds=timeout_seconds,
                        debug_http=debug_http,
                    )
                content = None
                if data is not None:
                    try:
                        content = data["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, TypeError):
                        content = None
                verdicts = parse_verification_verdicts(content, len(texts))
                selector.set_verdicts(key, verdicts)
                if ckpt_file is not None:
                    ckpt_file.write(json.dumps({"h": h, "v": verdicts}) + "\n")
                    ckpt_file.flush()
                done += 1
                if done % 500 == 0:
                    log(f"  verified {done}/{len(todo)} rows")

            await asyncio.gather(*(verify_one(*item) for item in todo))
    finally:
        if ckpt_file is not None:
            ckpt_file.close()
    log(f"Verification done ({done} rows{f', {reused} from checkpoint' if reused else ''}).")
