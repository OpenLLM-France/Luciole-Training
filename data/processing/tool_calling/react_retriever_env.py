"""Dense-retrieval environment backing the `wikipedia_retriever` tool.

The retriever counterpart of react_wiki_env.py: where WikiEnv browses Wikipedia
by title over HTTP/ZIM, RetrieverEnv does semantic search over a pre-built
embedding index (build_wiki_index.py, Qwen3-Embedding-0.6B). One natural-language
query returns the top-k most relevant passages -- no exact title, no separate
lookup step.

`submit_answer` and its F1 grading are inherited from GradedEnv (react_wiki_env.py),
so this module only implements the retrieval half. The tool schemas live in
react_tools.py; the pipeline wiring in react_hotpot.py.

The heavy embedding / faiss dependencies are imported lazily so this module (and
anything importing it) stays cheap unless a query is actually embedded.
"""

import json
import os
import threading
import time

import numpy as np
from loguru import logger

from react_wiki_env import GradedEnv

# The query encoder is not configurable here: it must match the model the index
# was built with, so it is read from the index's meta.json. Index directories
# (keyed by language) are defined in react_hotpot.WIKI_INDEX_PATHS.
DEFAULT_TOP_K = 10  # passages returned by wikipedia_retriever() when the model omits k
MAX_TOP_K = 20      # hard cap so a model cannot flood the episode context with k
# Max characters shown per source in an observation; -1 (or <=0) = no cap -- show
# the whole chunk, whose size is already bounded at index-build time (e.g.
# FineWiki's --max_length tokens per section).
MAX_PASSAGE_CHARS = -1

# How often (in queries) to emit an embedding-profiling line; 0 disables it.
EMBED_LOG_EVERY = int(os.environ.get("REACT_EMBED_LOG_EVERY", "50"))
# Cores actually usable by this process -- the reference the "avg concurrent
# encodes" figure below is judged against (>= cores means CPU-saturated).
try:
    _N_CORES = len(os.sched_getaffinity(0))
except AttributeError:  # not Linux
    _N_CORES = os.cpu_count() or 1

# FAISS search threads. Default 8, tuned for the real (32B) generation workload:
# there, conversations spend most of their time generating, so only ~1-2
# retrievals run at once -- 8 threads each stays well under the core count (no
# oversubscription) while cutting a single 5.2M-vector FlatIP scan from ~6 s
# (1 thread) toward ~1 s, shrinking the per-conversation off-GPU stall.
#
# Set REACT_FAISS_THREADS=1 for the 0.6B *test* model instead: there generation
# is near-instant, dozens of retrievals run concurrently, and 8 threads each
# would oversubscribe the cores (N_calls x threads >> cores) and slow every
# search. (FAISS threads are fixed at index load, before runtime concurrency is
# known, so this can't adapt automatically.) NB: FlatIP is FLOP-bound regardless
# -- an approximate index (HNSW/IVF) is the real fix if search ever dominates.
_FAISS_THREADS = int(os.environ.get("REACT_FAISS_THREADS", "8"))


class _EmbedStats:
    """Process-wide profiler answering "is the CPU embedder the bottleneck?".

    Every EMBED_LOG_EVERY queries it logs (into datatrove's own loguru stream,
    so it lands in the job log next to the inference metrics):

      * avg encode / search latency,
      * query throughput,
      * current & peak in-flight encodes, and
      * **avg concurrent encodes** = throughput x avg latency (Little's law) --
        the number of encodes running at once on average. If that meets or
        exceeds the usable core count, encoding is CPU-saturated and is a genuine
        bottleneck suspect; if it stays well below, embedding has spare capacity
        and something else (usually vLLM generation) is the long pole.
    """

    def __init__(self, report_every=EMBED_LOG_EVERY):
        self.lock = threading.Lock()
        self.report_every = report_every
        self.n = 0
        self.encode_s = 0.0
        self.search_s = 0.0
        self.inflight = 0
        self.peak = 0
        self.first_t = None

    def begin(self):
        """Mark an encode as started (bumps the in-flight gauge)."""
        if not self.report_every:
            return
        with self.lock:
            if self.first_t is None:
                self.first_t = time.perf_counter()
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)

    def end(self, encode_s, search_s):
        """Mark an encode as finished; log a summary every report_every queries."""
        if not self.report_every:
            return
        with self.lock:
            self.inflight -= 1
            self.n += 1
            self.encode_s += encode_s
            self.search_s += search_s
            if self.n % self.report_every:
                return
            n, es, ss = self.n, self.encode_s, self.search_s
            inflight, peak = self.inflight, self.peak
            wall = time.perf_counter() - self.first_t
        thru = n / wall if wall else 0.0
        # Little's law over the WHOLE tool call (encode + search): mean number of
        # retrievals doing CPU work at once. Counting encode alone hid that search
        # is the real consumer. (With _FAISS_THREADS=1 the search wall-time equals
        # its CPU-time, so this is accurate; multi-threaded search understates it.)
        busy = (es + ss) / wall if wall else 0.0
        verdict = "CPU-SATURATED" if busy >= 0.9 * _N_CORES else "has headroom"
        logger.info(
            "retriever embed [{}]: {} queries | encode {:.0f} ms, search {:.0f} ms avg "
            "| {:.1f} q/s | in-flight {} (peak {}) | avg concurrent {:.1f} vs {} cores "
            "| cum encode {:.0f}s + search {:.0f}s / wall {:.0f}s",
            verdict, n, 1000 * es / n, 1000 * ss / n, thru, inflight, peak,
            busy, _N_CORES, es, ss, wall,
        )


_STATS = _EmbedStats()


# ---------------------------------------------------------------------------
# Shared, process-wide index + embedder (loaded once, reused across all
# concurrent episodes in the process -- like WikiEnv's shared ZIM Archive).
# ---------------------------------------------------------------------------
class _Index:
    """A loaded dense-retrieval index: passages + a top-k search over them.

    Prefers a FAISS FlatIP index (index.faiss) when present and faiss is
    importable -- fast exact search. Otherwise falls back to a brute-force dot
    product over the memory-mapped float16 embedding matrix (correct, but reads
    the whole matrix per query, so only practical for small indexes).
    """

    def __init__(self, index_dir):
        with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        # Keep only what the observation needs (title + text); the row order in
        # passages.jsonl matches the embedding-matrix row order.
        self.titles, self.texts = [], []
        with open(os.path.join(index_dir, "passages.jsonl"), encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                self.titles.append(p.get("title") or "")
                self.texts.append(p.get("text") or "")

        self._faiss = None
        self._emb = None
        faiss_path = os.path.join(index_dir, "index.faiss")
        if os.path.exists(faiss_path):
            try:
                import faiss

                # Decouple FAISS threading from the OMP=1 embedder cap (see
                # _FAISS_THREADS): restores multi-threaded search.
                faiss.omp_set_num_threads(_FAISS_THREADS)
                self._faiss = faiss.read_index(faiss_path)
            except ImportError:
                pass
        if self._faiss is not None:
            logger.info(
                "retriever index: FAISS backend, {} vectors, {} search threads",
                self._faiss.ntotal, _FAISS_THREADS,
            )
        else:
            # mmap so we don't pull ~10 GB of float16 into RAM at load time.
            self._emb = np.load(
                os.path.join(index_dir, "embeddings.f16.npy"), mmap_mode="r"
            )
            logger.warning(
                "retriever index: no FAISS index -- brute-force numpy over {} vectors, "
                "single-threaded under OMP=1 (expect ~seconds/query). Rebuild with "
                "--build_faiss.", self._emb.shape[0],
            )

    def search(self, qvec, k):
        """Return up to k (score, row_index) pairs, best first."""
        k = min(k, len(self.titles))
        if k == 0:
            return []
        if self._faiss is not None:
            scores, idxs = self._faiss.search(qvec.astype(np.float32)[None, :], k)
            scores, idxs = scores[0], idxs[0]
        else:
            # Rows are L2-normalized (see build_wiki_index): dot == cosine.
            sims = np.asarray(self._emb @ qvec.astype(np.float16), dtype=np.float32)
            idxs = np.argpartition(-sims, k - 1)[:k]
            idxs = idxs[np.argsort(-sims[idxs])]
            scores = sims[idxs]
        return [(float(s), int(i)) for s, i in zip(scores, idxs) if i >= 0]


_index_cache = {}
_index_lock = threading.Lock()


def _open_index(index_dir):
    with _index_lock:
        index = _index_cache.get(index_dir)
        if index is None:
            index = _Index(index_dir)
            _index_cache[index_dir] = index
    return index


_embedder_cache = {}
_embedder_lock = threading.Lock()


def _open_embedder(model_name):
    """Load (once) the SentenceTransformer query encoder, on CPU.

    Always CPU: the tool executor runs in the same process as the vLLM server,
    which reserves most of the GPU, so a second CUDA model would likely OOM.
    Query encoding is a single short string, fast enough on CPU.
    """
    with _embedder_lock:
        model = _embedder_cache.get(model_name)
        if model is None:
            import torch
            from sentence_transformers import SentenceTransformer  # lazy: heavy import

            # Single-threaded intra-op: hundreds of these encoders run concurrently
            # (one per tool call), so a multi-threaded BLAS pool per encode exhausts
            # OpenBLAS's buffer count. See the env-var note in react_hotpot.py.
            torch.set_num_threads(1)
            model = SentenceTransformer(model_name, device="cpu")
            _embedder_cache[model_name] = model
    return model


def _request_label(n):
    """Spreadsheet-style letter for the n-th request (1->A, 26->Z, 27->AA)."""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _clean_passage(text):
    """Tidy a passage while preserving its line structure (for multi-line passages).

    Collapses intra-line whitespace and strips each line, drops leading/trailing
    blank lines, and squeezes runs of blank lines to a single one -- so paragraph
    breaks survive but stray indentation and blank runs do not bloat the context.
    """
    out = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if not line and (not out or not out[-1]):
            continue  # skip leading blank / collapse blank runs
        out.append(line)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out)


def _attr(value):
    """Make a string safe to sit inside a double-quoted XML attribute.

    Only the quote can break the tag structure a reader relies on, so neutralise
    it (titles rarely contain one); everything else is left readable rather than
    entity-escaped, since the observation is shown to the model, not parsed.
    """
    return " ".join(value.split()).replace('"', "'")


class RetrieverEnv(GradedEnv):
    """Dense-retrieval env backing the `wikipedia_retriever` tool.

    Embeds each query with the same Qwen3 model used to build the index, applying
    the Qwen3 query-side instruction ("Instruct: {task}\\nQuery: {q}") read back
    from the index's meta.json so query and passage encodings stay in sync, then
    returns the top-k passages. `submit_answer` grading is inherited from
    GradedEnv (identical to the wiki_api tool set).
    """

    def __init__(self, ground_truth=None, index_dir=None,
                 accept_threshold=None, scorer=None):
        super().__init__(ground_truth=ground_truth, accept_threshold=accept_threshold,
                         scorer=scorer)
        if not index_dir:
            raise ValueError("RetrieverEnv requires an index_dir")
        self.index_dir = index_dir
        # Pagination state for next_results: the most recent query's embedding
        # and how many of its ranked results have already been shown. Per-episode
        # (each conversation gets a fresh env), so no cross-conversation leakage.
        self._last_qvec = None
        self._last_offset = 0
        # Per-request label so passage ids are globally unique within an episode:
        # the Nth wikipedia_retriever call is request A, B, C, ... and its passages
        # are [A1], [A2], ...; next_results keeps the same letter and continues the
        # numbering ([A6], [A7]). Reset each episode with the fresh env.
        self._request_no = 0
        self._request_label = ""

    def wikipedia_retriever(self, query, k=DEFAULT_TOP_K):
        """wikipedia_retriever[query, k]: the top-k Wikipedia passages by semantic similarity.

        Starts a fresh ranked list for `query`; `next_results` then walks further
        down it. `k` (optional, default DEFAULT_TOP_K) is clamped to [1, MAX_TOP_K]
        so a stray large value cannot flood the episode context.
        """
        k = max(1, min(int(k), MAX_TOP_K))
        index = _open_index(self.index_dir)
        # The query encoder must match the model the index was built with, so it
        # is a property of the index -- read it from meta.json rather than
        # letting the caller pick a (possibly mismatched) model.
        model = _open_embedder(index.meta["model"])
        # Qwen3 asymmetry: passages were embedded raw, queries get the instruct
        # prefix. `prompt=` is prepended verbatim by SentenceTransformer, giving
        # exactly "Instruct: {task}\nQuery: {query}".
        instruction = index.meta.get("query_instruction")
        prompt = f"Instruct: {instruction}\nQuery: " if instruction else None
        # Time the encode (the CPU-heavy step) and search under the in-flight
        # gauge so _STATS can report whether embedding is CPU-bound.
        _STATS.begin()
        encode_s = search_s = 0.0
        try:
            t0 = time.perf_counter()
            qvec = model.encode(
                [query], prompt=prompt, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )[0]
            encode_s = time.perf_counter() - t0
            self._last_qvec = qvec  # remembered so next_results need not re-embed
            self._last_offset = 0

            t0 = time.perf_counter()
            hits = index.search(qvec, k)
            search_s = time.perf_counter() - t0
        finally:
            _STATS.end(encode_s, search_s)
        if not hits:
            self.obs = "No passages found for that query."
            return self.obs
        # New request -> next letter; next_results will keep this letter.
        self._request_no += 1
        self._request_label = _request_label(self._request_no)
        self._last_offset = len(hits)
        self.obs = self._render(index, hits, start_rank=1)
        return self.obs

    def next_results(self, k=DEFAULT_TOP_K):
        """next_results[k]: the next k passages for the most recent wikipedia_retriever query.

        Continues further down the same ranked list (like scrolling past the first
        page of results). Must follow a `wikipedia_retriever`. `k` (optional, default
        DEFAULT_TOP_K) is clamped to [1, MAX_TOP_K].
        """
        if self._last_qvec is None:
            return "Error: call `wikipedia_retriever` before `next_results`."
        k = max(1, min(int(k), MAX_TOP_K))
        index = _open_index(self.index_dir)
        # Re-rank the top (offset + k) for the stored query and take the new tail.
        # Cheap for FAISS, unbounded in depth, and avoids re-embedding the query.
        hits = index.search(self._last_qvec, self._last_offset + k)
        new_hits = hits[self._last_offset:]
        if not new_hits:
            self.obs = "No more results for this query."
            return self.obs
        start_rank = self._last_offset + 1
        self._last_offset += len(new_hits)
        self.obs = self._render(index, new_hits, start_rank=start_rank)
        return self.obs

    def _render(self, index, hits, start_rank):
        """Format ranked (score, row) hits as an observation.

        Each passage is wrapped in a <source id="..." title="..."> tag. The id is
        the request letter plus its rank within that request's ranked list (e.g.
        A1, A2 for the first query; B1 for the next query; and, after a
        next_results, A6), so ids stay unique across every query in the episode and
        submit_answer can cite them. The explicit open/close tag delimits passages
        unambiguously even when a passage spans multiple lines -- a bare
        line-leading id could otherwise be confused with the passage's own text.
        """
        label = self._request_label
        blocks = []
        for offset, (score, i) in enumerate(hits):
            text = _clean_passage(index.texts[i])
            if MAX_PASSAGE_CHARS > 0 and len(text) > MAX_PASSAGE_CHARS:
                text = text[:MAX_PASSAGE_CHARS].rstrip() + "..."
            pid = f"{label}{start_rank + offset}"
            title = index.titles[i]
            open_tag = (
                f'<source id="{pid}" title="{_attr(title)}">' if title
                else f'<source id="{pid}">'
            )
            blocks.append(f"{open_tag}\n{text}\n</source>")
        return "\n\n".join(blocks)


def new_retriever_env(doc, index_dir=None, accept_threshold=None, scorer=None):
    """Create a fresh RetrieverEnv for one conversation and alias its scores.

    Mirrors react_hotpot.new_env (the wiki_api factory). The ground-truth answer
    is taken from the document so `submit_answer` can grade the agent; the env's
    score list is aliased into the doc metadata so each submit_answer score lands
    in the output (same list object). `scorer` overrides the default token-F1
    grading (e.g. FEVER exact match). How much of each source is shown is capped by
    the module-level MAX_PASSAGE_CHARS.
    """
    env = RetrieverEnv(
        ground_truth=doc.metadata["answer"], index_dir=index_dir,
        accept_threshold=accept_threshold, scorer=scorer,
    )
    doc.metadata["submit_answer_scores"] = env.scores
    return env
