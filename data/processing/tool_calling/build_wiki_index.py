"""Build a dense-retrieval index over Wikipedia lead paragraphs (the HotpotQA
"abstracts" corpus) with Qwen3-Embedding-0.6B.

One-time, offline job. Writes under --out:
  passages.jsonl      one {"id","title","text"} per passage; row i in this file
                      corresponds to row i in the embedding matrix.
  embeddings.f16.npy  float16 array [N, dim], each row L2-normalized so that a
                      dot product == cosine similarity.
  meta.json           model / dim / count / the query instruction / provenance.
  index.faiss         (optional) FlatIP index, only if faiss is installed and
                      --build_faiss is passed.

Two corpus sources (--source):
  beir  HotpotQA fullwiki corpus from HF (BeIR/hotpotqa, English, ~5.2M passages)
        -- the exact corpus the HotpotQA leaderboard retrieves over.
  zim   The lead paragraph of every article in a local Kiwix ZIM. Works for en
        and fr, and reuses the mirrors already used by react_hotpot.py.

IMPORTANT (Qwen3 asymmetry): passages are embedded with NO instruction; queries
must be embedded as "Instruct: {task}\nQuery: {q}". The query side reads the
task string back from meta.json["query_instruction"] so the two stay in sync.

Depends on sentence-transformers + datasets (+ libzim for --source zim); faiss
is optional. Run on one H100:

  python build_wiki_index.py --source beir --out "$OpenLLM_OUTPUT/data/wiki_index/en"
  python build_wiki_index.py --source zim --lang fr --out "$OpenLLM_OUTPUT/data/wiki_index/fr"
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
# Let the CUDA caching allocator grow segments instead of fragmenting: the OOM
# traces showed ~18 GB "reserved but unallocated" (fragmentation) on top of a
# transient attention spike. Must be set before torch is imported (below, via
# sentence_transformers).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from sentence_transformers import SentenceTransformer

# Mirrors react_hotpot.WIKI_ZIM_PATHS (kept local so this one-time builder does
# not import the heavy datatrove inference pipeline just to read two paths).
_WIKI_ZIM_DIR = os.path.expandvars("$OpenLLM_OUTPUT/data/wikipedia")
WIKI_ZIM_PATHS = {
    "en": os.path.join(_WIKI_ZIM_DIR, "wikipedia_en_all_nopic.zim"),
    "fr": os.path.join(_WIKI_ZIM_DIR, "wikipedia_fr_all_nopic_2026-05.zim"),
}

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


def _lead_paragraph(html, max_chars):
    """Extract an article's lead paragraph(s) from its rendered HTML.

    Concatenates the first content <p> blocks (each longer than two words) up to
    max_chars, mirroring the "abstract" granularity HotpotQA is built on. Returns
    "" for pages with no real prose (disambiguation / index pages), which the
    caller then skips.
    """
    from bs4 import BeautifulSoup  # available via react_wiki_env's deps

    soup = BeautifulSoup(html, "html.parser")
    content = soup.find(class_="mw-parser-output") or soup
    parts, total = [], 0
    for p in content.find_all("p"):
        t = " ".join(p.get_text().split())
        if len(t.split()) <= 2:
            continue
        parts.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return " ".join(parts)[:max_chars]


def iter_zim(zim_path, limit=None, max_chars=1200, min_chars=40):
    """Yield the lead paragraph of every article in a Kiwix ZIM.

    Iterates all entries by id, keeping only non-redirect text/html items (the
    articles). Media and metadata entries are skipped by mimetype.
    """
    from libzim.reader import Archive

    archive = Archive(zim_path)
    n = 0
    for eid in range(archive.all_entry_count):
        if limit is not None and n >= limit:
            break
        entry = archive._get_entry_by_id(eid)
        if entry.is_redirect:
            continue
        item = entry.get_item()
        if item.mimetype != "text/html":
            continue
        html = bytes(item.content).decode("utf-8", errors="ignore")
        text = _lead_paragraph(html, max_chars)
        if len(text) < min_chars:  # disambiguation / stub / index page
            continue
        n += 1
        yield entry.path, entry.title, text


SOURCES = {"beir": iter_beir, "zim": iter_zim}


# ---------------------------------------------------------------------------
# Qwen3 embedding. SentenceTransformer handles the last-token pooling, L2
# normalization, batching and device placement; passages are encoded with no
# prompt and queries with the "Instruct: ...\nQuery: " prompt (Qwen3 asymmetry).
# ---------------------------------------------------------------------------
def load_embedder(model_name, max_length):
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
    model.max_seq_length = max_length
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=SOURCES, default="beir",
                    help="'beir' = HotpotQA corpus (English); 'zim' = local Kiwix mirror (en/fr)")
    ap.add_argument("--lang", choices=WIKI_ZIM_PATHS, default="en",
                    help="ZIM language for --source zim (selects the mirror)")
    ap.add_argument("--zim_path", default=None, help="Override the ZIM path (default: WIKI_ZIM_PATHS[lang])")
    ap.add_argument("--out", required=True, help="Output directory for the index artifacts")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--max_length", type=int, default=1024, help="Passage truncation length in tokens")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--task", default=DEFAULT_TASK, help="Query-side instruction (stored in meta.json)")
    ap.add_argument("--max_chars", type=int, default=1200, help="[zim] lead-paragraph char budget per article")
    ap.add_argument("--limit", type=int, default=None, help="Cap passages (smoke tests)")
    ap.add_argument("--build_faiss", action="store_true", help="Also build a FAISS FlatIP index (needs faiss)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    passages_path = os.path.join(args.out, "passages.jsonl")
    emb_path = os.path.join(args.out, "embeddings.f16.npy")
    meta_path = os.path.join(args.out, "meta.json")

    if args.source == "beir":
        corpus = iter_beir(limit=args.limit)
    else:
        zim_path = args.zim_path or WIKI_ZIM_PATHS[args.lang]
        if not os.path.exists(zim_path):
            raise SystemExit(f"ZIM not found: {zim_path}")
        corpus = iter_zim(zim_path, limit=args.limit, max_chars=args.max_chars)

    embedder = load_embedder(args.model, max_length=args.max_length)
    dim = embedder.get_sentence_embedding_dimension()
    print(f"model={args.model} dim={dim} device={embedder.device} "
          f"source={args.source} out={args.out}", flush=True)

    def flush(texts, metas, pf, chunks):
        """Embed one buffer, append fp16 rows, and write the aligned passages."""
        # normalize -> dot product == cosine; no prompt -> passage-side encoding.
        emb = embedder.encode(
            texts, batch_size=args.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float16)  # cast now to cap RAM
        chunks.append(emb)
        for (id_, title, text) in metas:
            pf.write(json.dumps({"id": id_, "title": title, "text": text}, ensure_ascii=False) + "\n")

    buf_texts, buf_metas, chunks, n = [], [], [], 0
    t0 = time.time()
    with open(passages_path, "w", encoding="utf-8") as pf:
        for id_, title, text in corpus:
            buf_texts.append(text)
            buf_metas.append((id_, title, text))
            if len(buf_texts) >= args.batch_size:
                flush(buf_texts, buf_metas, pf, chunks)
                n += len(buf_texts)
                buf_texts, buf_metas = [], []
                if n % (args.batch_size * 40) == 0:
                    rate = n / (time.time() - t0)
                    print(f"  {n:>9,} passages  ({rate:,.0f}/s)", flush=True)
        if buf_texts:
            flush(buf_texts, buf_metas, pf, chunks)
            n += len(buf_texts)

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
        "lang": args.lang if args.source == "zim" else "en",
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
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings.astype(np.float32))
            faiss.write_index(index, os.path.join(args.out, "index.faiss"))
            print(f"wrote index.faiss ({index.ntotal:,} vectors)", flush=True)


if __name__ == "__main__":
    main()
