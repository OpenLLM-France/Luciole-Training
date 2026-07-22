"""Build a dense-retrieval index over Wikipedia lead paragraphs (the HotpotQA
"abstracts" corpus) with Qwen3-Embedding-0.6B.

One-time, offline job. Writes under --out:
  passages.jsonl      one {"id","title","text"} per passage; row i in this file
                      corresponds to row i in the embedding matrix.
  embeddings.f16.npy  float16 array [N, dim], each row L2-normalized so that a
                      dot product == cosine similarity.
  meta.json           model / dim / count / the query instruction / provenance.
  index.faiss         (optional) FAISS index (--index_type flat|ivfpq), only if
                      faiss is installed and --build_faiss is passed.

Corpus sources (--source):
  beir     HotpotQA fullwiki corpus from HF (BeIR/hotpotqa, English, ~5.2M
           passages) -- the exact corpus the HotpotQA leaderboard retrieves over.
  finewiki Full articles from HuggingFaceFW/finewiki (--finewiki_config, e.g.
           en). Articles are split into <=max_length-token chunks with a
           RecursiveCharacterTextSplitter; continuation chunks are prefixed with
           the article title so they stay self-identifying. Defaults --max_length
           to 512. Tens of millions of chunks -> pair with --index_type ivfpq so
           the FAISS build does not OOM.

beir keeps its old defaults (one passage per article, --index_type flat), so
existing invocations are unchanged.

IMPORTANT (Qwen3 asymmetry): passages are embedded with NO instruction; queries
must be embedded as "Instruct: {task}\nQuery: {q}". The query side reads the
task string back from meta.json["query_instruction"] so the two stay in sync.

Depends on sentence-transformers + datasets; faiss is optional. Run on one H100:

  python build_wiki_index.py --source beir --out "$OpenLLM_OUTPUT/data/wiki_index/en"
  python build_wiki_index.py --source finewiki --finewiki_config en \
    --index_type ivfpq --build_faiss --out "$OpenLLM_OUTPUT/data/finewiki_index/en"
"""

import argparse
import json
import os
import time

# Jean Zay compute nodes have no internet: read models/datasets from the HF
# cache and never hit the network (a cached HEAD check would otherwise fail with
# "Network is unreachable"). setdefault so an online node can still override by
# exporting HF_HUB_OFFLINE=0 before running. Must precede the HF imports below.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# Chunking is parallelized with datasets.map(num_proc=...); disable the fast
# tokenizer's own thread pool so it doesn't oversubscribe / deadlock on fork.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Let the CUDA caching allocator grow segments instead of fragmenting: the OOM
# traces showed ~18 GB "reserved but unallocated" (fragmentation) on top of a
# transient attention spike. Must be set before torch is imported (below, via
# sentence_transformers).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
# sentence_transformers is imported lazily inside load_embedder so the chunking
# helpers (chunk_text / _chunk_batch) can be imported without the embedding stack.

# Retrieval task fed to Qwen3-Embedding as the query-side instruction. Stored in
# meta.json so the query encoder uses the exact same string.
DEFAULT_TASK = "Given a question, retrieve Wikipedia passages that answer it"


# ---------------------------------------------------------------------------
# Corpus sources: each yields (id, title, text) tuples.
# ---------------------------------------------------------------------------
def iter_beir(limit=None):
    """Iterate the BeIR HotpotQA corpus (English abstracts, ~5.2M passages).

    Non-streaming so it works offline from the HF cache (streaming reads from the
    Hub over HTTP even when cached). The dataset must be pre-downloaded on a node
    with internet -- see the header for the pre-cache command.
    """
    from datasets import load_dataset

    ds = load_dataset("BeIR/hotpotqa", "corpus", split="corpus")
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        text = (row.get("text") or "").strip()
        if not text:
            continue
        yield str(row.get("_id", i)), (row.get("title") or "").strip(), text


# ---------------------------------------------------------------------------
# FineWiki (HuggingFaceFW/finewiki): full articles chunked into <=max_tokens
# pieces with LangChain's RecursiveCharacterTextSplitter -- the standard recursive
# splitter used across the RAG literature. Pass max_tokens below the model's
# max_seq_length to leave room for the EOS Qwen3-Embedding appends (last-token
# pooling).
# ---------------------------------------------------------------------------


def chunk_text(text, title, tokenizer, max_tokens=512):
    """FineWiki article -> chunks via LangChain's RecursiveCharacterTextSplitter
    (default separators). The splitter caps *content* at max_tokens; every
    continuation chunk (all but the first, which already opens the article) is
    then prefixed with the article title so it stays self-identifying for
    retrieval. Those title tokens are NOT counted against max_tokens, so a chunk
    can exceed it by the title length -- load_embedder adds a matching margin
    (TITLE_MARGIN) so the embedder never truncates the appended EOS."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=max_tokens, chunk_overlap=0)

    for i, piece in enumerate(splitter.split_text(text), 0):
        if i > 0:
            piece = f"# {title}\n{piece}"
        piece = piece.strip()
        if piece:
            yield piece


def _chunk_batch(batch, tokenizer, max_tokens):
    """datasets.map worker (batched, one-to-many): turn a batch of FineWiki
    articles into <=max_tokens chunks. Returns aligned id/title/text lists; chunk
    id is '<article-id>#<k>'. Module-level so it pickles to the map workers, which
    run in parallel (num_proc) -- this is what keeps the GPUs fed rather than
    starved behind a single-threaded chunker."""
    ids, titles, texts = [], [], []
    for i in range(len(batch["text"])):
        text = (batch["text"][i] or "").strip()
        if not text:
            continue
        title = (batch["title"][i] or "").strip()
        base = batch["id"][i] if batch["id"][i] is not None else batch["page_id"][i]
        for k, chunk in enumerate(chunk_text(text, title, tokenizer, max_tokens)):
            ids.append(f"{base}#{k}")
            titles.append(title)
            texts.append(chunk)
    return {"id": ids, "title": titles, "text": texts}


SOURCES = ("beir", "finewiki")


# Extra tokens the embedder tolerates on top of the chunk budget, so the title
# prefix chunk_text prepends to continuation chunks (plus the appended EOS) is
# never truncated -- truncating the EOS would corrupt Qwen3's last-token pooling.
# Sized to cover a Wikipedia title; long-title chunks beyond this get clipped.
TITLE_MARGIN = 32


# ---------------------------------------------------------------------------
# Qwen3 embedding. SentenceTransformer handles the last-token pooling, L2
# normalization, batching and device placement; passages are encoded with no
# prompt and queries with the "Instruct: ...\nQuery: " prompt (Qwen3 asymmetry).
# ---------------------------------------------------------------------------
def load_embedder(model_name, max_length):
    from sentence_transformers import SentenceTransformer
    # Attention backend matters for memory: the naive "eager" kernel materializes
    # the full [batch, heads, seq, seq] score matrix (O(seq^2) memory -> the OOM
    # on long-passage batches). Prefer flash-attn (fastest), else fall back to
    # PyTorch SDPA, which also avoids materializing that matrix (same O(seq)
    # memory for this inference workload) -- NOT to eager.
    for attn in ("flash_attention_2", "sdpa"):
        try:
            model = SentenceTransformer(
                model_name,
                model_kwargs={"torch_dtype": "float16", "attn_implementation": attn},
            )
            break
        except Exception:
            continue
    else:  # neither backend loaded -- let transformers pick its default
        model = SentenceTransformer(model_name, model_kwargs={"torch_dtype": "float16"})
    # Room for the title prefix + EOS on top of the chunk budget (see chunk_text).
    model.max_seq_length = max_length + TITLE_MARGIN
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=SOURCES, default="beir",
                    help="'beir' = HotpotQA corpus (English); 'finewiki' = HuggingFaceFW/finewiki")
    ap.add_argument("--out", required=True, help="Output directory for the index artifacts")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--max_length", type=int, default=None,
                    help="Passage truncation length in tokens (default: 512 for finewiki, else 1024)")
    ap.add_argument("--batch_size", type=int, default=256, help="Per-GPU micro-batch for the encoder")
    ap.add_argument("--num_gpus", type=int, default=0,
                    help="GPUs for encoding (0=auto: all visible). >1 uses a sentence-transformers "
                         "multi-process pool, one worker per GPU, splitting the corpus across them")
    ap.add_argument("--shard_size", type=int, default=100_000,
                    help="Texts buffered before each encode call (dispatch granularity + RAM cap)")
    ap.add_argument("--mp_chunk_size", type=int, default=0,
                    help="[multi-gpu] sentences dispatched per worker call (0=auto)")
    ap.add_argument("--task", default=DEFAULT_TASK, help="Query-side instruction (stored in meta.json)")
    ap.add_argument("--limit", type=int, default=None, help="Cap passages (smoke tests)")
    ap.add_argument("--build_faiss", action="store_true", help="Also build a FAISS index (needs faiss)")
    ap.add_argument("--finewiki_config", default="en",
                    help="[finewiki] HuggingFaceFW/finewiki language config (e.g. en, fr, de)")
    ap.add_argument("--chunk_procs", type=int, default=0,
                    help="[finewiki] parallel processes for chunking (0=auto: SLURM_CPUS_PER_TASK)")
    ap.add_argument("--limit_articles", type=int, default=None,
                    help="[finewiki] cap source articles before chunking (smoke tests)")
    ap.add_argument("--index_type", choices=("flat", "ivfpq"), default="flat",
                    help="FAISS index kind: 'flat' exact (default; fine <~5M vectors) or "
                         "'ivfpq' compressed (use for chunked finewiki, ~tens of M vectors)")
    ap.add_argument("--ivf_nlist", type=int, default=0, help="[ivfpq] #coarse cells (0=auto: 4*sqrt(N))")
    ap.add_argument("--pq_m", type=int, default=128, help="[ivfpq] PQ subquantizers (must divide dim)")
    ap.add_argument("--pq_nbits", type=int, default=8, help="[ivfpq] bits per subquantizer")
    ap.add_argument("--nprobe", type=int, default=32, help="[ivfpq] cells probed at search time (recall knob)")
    ap.add_argument("--train_size", type=int, default=1_000_000, help="[ivfpq] #vectors sampled to train the index")
    ap.add_argument("--add_batch", type=int, default=1_000_000, help="[ivfpq] rows per add() call (caps peak RAM)")
    args = ap.parse_args()

    if args.max_length is None:
        args.max_length = 512 if args.source == "finewiki" else 1024

    os.makedirs(args.out, exist_ok=True)
    passages_path = os.path.join(args.out, "passages.jsonl")
    emb_path = os.path.join(args.out, "embeddings.f16.npy")
    meta_path = os.path.join(args.out, "meta.json")

    # Build the (id, title, text) corpus FIRST. For finewiki this runs the CPU
    # chunking in parallel via datasets.map(num_proc=...) and MUST happen before
    # any CUDA is initialized (load_embedder below): map forks its workers, and
    # forking after CUDA init is unsafe. Parallel chunking is what keeps the GPUs
    # fed instead of starved behind a single-threaded chunker.
    if args.source == "beir":
        corpus = iter_beir(limit=args.limit)
    else:  # finewiki: parallel-chunk full articles -> <=max_length-token chunks
        import itertools
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)   # CPU only, no CUDA yet
        ds = load_dataset("HuggingFaceFW/finewiki", args.finewiki_config, split="train")
        if args.limit_articles is not None:
            ds = ds.select(range(min(len(ds), args.limit_articles)))
        procs = args.chunk_procs or int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
        # Chunk content is capped at max_length tokens; the title prefix and EOS
        # are absorbed by TITLE_MARGIN on the embedder side (see chunk_text).
        chunked = ds.map(
            _chunk_batch, batched=True, batch_size=200, num_proc=procs,
            remove_columns=ds.column_names,
            fn_kwargs=dict(tokenizer=tok, max_tokens=args.max_length),
            desc="chunking finewiki")
        print(f"chunked {len(ds):,} articles -> {len(chunked):,} chunks "
              f"across {procs} procs", flush=True)
        corpus = ((r["id"], r["title"], r["text"]) for r in chunked)
        if args.limit is not None:
            corpus = itertools.islice(corpus, args.limit)

    embedder = load_embedder(args.model, max_length=args.max_length)
    # get_embedding_dimension is the current name; fall back for older ST.
    dim = (embedder.get_embedding_dimension() if hasattr(embedder, "get_embedding_dimension")
           else embedder.get_sentence_embedding_dimension())
    print(f"model={args.model} dim={dim} device={embedder.device} "
          f"source={args.source} out={args.out} max_length={args.max_length}", flush=True)

    # Encoder fan-out. With >1 GPU, sentence-transformers spawns one worker
    # process per device and splits each shard across them (near-linear speedup
    # on the embedding step, which dominates runtime). Output order is preserved,
    # so passages.jsonl stays row-aligned with the embedding matrix.
    import torch
    n_gpus = args.num_gpus or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    pool = None
    if n_gpus > 1:
        targets = [f"cuda:{i}" for i in range(n_gpus)]
        pool = embedder.start_multi_process_pool(target_devices=targets)
        print(f"multi-GPU encode across {targets}", flush=True)

    def encode(texts):
        # No prompt -> passage-side encoding (Qwen3 asymmetry). Normalize here
        # (not via the encoder kwarg) so behaviour is identical on the single- and
        # multi-process paths regardless of sentence-transformers version.
        if pool is not None:
            emb = embedder.encode_multi_process(
                texts, pool, batch_size=args.batch_size,
                chunk_size=args.mp_chunk_size or None)
        else:
            emb = embedder.encode(
                texts, batch_size=args.batch_size, convert_to_numpy=True,
                show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        np.divide(emb, norms, out=emb, where=norms > 0)  # -> dot == cosine
        return emb.astype(np.float16)                    # cast now to cap RAM

    def flush(texts, metas, pf, chunks):
        """Embed one shard, append fp16 rows, and write the aligned passages."""
        chunks.append(encode(texts))
        for (id_, title, text) in metas:
            pf.write(json.dumps({"id": id_, "title": title, "text": text}, ensure_ascii=False) + "\n")

    buf_texts, buf_metas, chunks, n = [], [], [], 0
    t0 = time.time()
    with open(passages_path, "w", encoding="utf-8") as pf:
        for id_, title, text in corpus:
            buf_texts.append(text)
            buf_metas.append((id_, title, text))
            if len(buf_texts) >= args.shard_size:
                flush(buf_texts, buf_metas, pf, chunks)
                n += len(buf_texts)
                buf_texts, buf_metas = [], []
                rate = n / (time.time() - t0)
                print(f"  {n:>12,} passages  ({rate:,.0f}/s)", flush=True)
        if buf_texts:
            flush(buf_texts, buf_metas, pf, chunks)
            n += len(buf_texts)

    if pool is not None:
        embedder.stop_multi_process_pool(pool)

    if not chunks:
        raise SystemExit("No passages produced -- check the source / paths.")

    embeddings = np.concatenate(chunks, axis=0)
    assert embeddings.shape[0] == n, (embeddings.shape[0], n)
    np.save(emb_path, embeddings)

    meta = {
        "model": args.model,
        "dim": int(dim),
        "count": int(n),
        "normalized": True,
        "dtype": "float16",
        "metric": "inner_product",  # == cosine, rows are L2-normalized
        "query_instruction": args.task,
        "passage_instruction": None,  # Qwen3: passages embedded raw
        "max_length": args.max_length,
        "source": args.source,
        "lang": args.finewiki_config if args.source == "finewiki" else "en",
        "index_type": args.index_type if args.build_faiss else None,
        "nprobe": args.nprobe if (args.build_faiss and args.index_type == "ivfpq") else None,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"wrote {n:,} passages -> {emb_path} ({embeddings.nbytes/1e9:.1f} GB) "
          f"in {time.time()-t0:.0f}s", flush=True)

    if args.build_faiss:
        try:
            import faiss
        except ImportError:
            print("faiss not installed; skipping index.faiss "
                  "(pip install faiss-gpu-cu12 or faiss-cpu). "
                  "The .npy alone is enough for torch top-k search.", flush=True)
        else:
            if args.index_type == "flat":
                # Exact inner-product search. Holds all vectors as float32 in RAM
                # (~4 bytes * N * dim) -- fine up to a few M vectors.
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings.astype(np.float32))
            else:
                # IVF-PQ: partitioned + compressed, for tens of millions of
                # vectors (chunked finewiki) where a flat f32 copy would OOM.
                if dim % args.pq_m:
                    raise SystemExit(f"--pq_m {args.pq_m} must divide dim {dim}")
                nlist = args.ivf_nlist or max(1, int(4 * (n ** 0.5)))
                nlist = min(nlist, n)
                quant = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFPQ(quant, dim, nlist, args.pq_m,
                                         args.pq_nbits, faiss.METRIC_INNER_PRODUCT)
                ts = min(args.train_size, n)
                rng = np.random.default_rng(0)  # seeded -> reproducible offline
                sample = np.sort(rng.choice(n, size=ts, replace=False))
                print(f"training IVFPQ (nlist={nlist}, m={args.pq_m}, "
                      f"nbits={args.pq_nbits}) on {ts:,} vectors...", flush=True)
                index.train(embeddings[sample].astype(np.float32))
                for i in range(0, n, args.add_batch):  # batched -> caps peak RAM
                    index.add(embeddings[i:i + args.add_batch].astype(np.float32))
                index.nprobe = args.nprobe
            faiss.write_index(index, os.path.join(args.out, "index.faiss"))
            print(f"wrote index.faiss ({index.ntotal:,} vectors, "
                  f"type={args.index_type})", flush=True)


if __name__ == "__main__":
    main()
