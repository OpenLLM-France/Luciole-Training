#!/usr/bin/env python3
"""Remove a redundant output weight matrix from tied-embedding models.

When a model ties its input and output embeddings (``tie_word_embeddings: true``
in ``config.json``), the language-modelling head is reconstructed from the input
embedding matrix at load time, so any ``lm_head.weight`` stored in the checkpoint
is **ignored** when the model is loaded — whatever it contains. It is dead weight
that just wastes ``vocab_size * hidden_size * dtype_bytes`` on disk (~0.5 GB for a
128k-vocab / 2048-hidden bf16 model).

In practice that stored head shows up in two flavours:
  * all zeros (e.g. the released Luciole-1B-Base), or
  * a stale, partially-diverged copy of the input embeddings (e.g. intermediate
    training checkpoints, where it stays highly correlated with the embeddings).

Both are safe to delete because tying overrides them at load. This script removes
the head **in place**, but only when, for each model:

  1. ``config.json`` has ``tie_word_embeddings == true``; and
  2. the stored head is either **all zeros** OR **nearly equal to the input
     embeddings** (flattened cosine >= --cosine-threshold).

If the head is tied but is *neither* all-zero nor close to the embeddings, the
script **raises an exception** for that model rather than guessing — that is an
unexpected state worth inspecting by hand (the head is still ignored at load, but
the mismatch is surprising enough that we refuse to delete silently).

It accepts either:
  * a single model folder (containing ``config.json`` + safetensors), or
  * a parent folder holding several model subfolders (each processed in turn).

Both single-file (``model.safetensors``) and sharded
(``model.safetensors.index.json`` + shards) layouts are supported.

Requires: torch, safetensors. Run, for example, with::

    python remove_useless_tied_weight_matrix.py /path/to/models --dry-run
    python remove_useless_tied_weight_matrix.py /path/to/model
"""
import argparse
import json
import os
import struct
import sys

# Common names used for the (tied) output projection across architectures.
HEAD_KEY_CANDIDATES = ["lm_head.weight", "output.weight", "embed_out.weight"]

# Common names used for the input embedding matrix across architectures.
EMBED_KEY_CANDIDATES = [
    "model.embed_tokens.weight",
    "transformer.wte.weight",
    "gpt_neox.embed_in.weight",
    "model.embed_in.weight",
    "tok_embeddings.weight",
    "embed_tokens.weight",
]

DEFAULT_COSINE_THRESHOLD = 0.9


# --------------------------------------------------------------------------- #
# Filesystem / discovery helpers
# --------------------------------------------------------------------------- #
def is_model_dir(path):
    """A model folder has a config.json and at least one safetensors artifact."""
    if not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    if os.path.isfile(os.path.join(path, "model.safetensors")):
        return True
    if os.path.isfile(os.path.join(path, "model.safetensors.index.json")):
        return True
    # Any *.safetensors shard also qualifies.
    return any(f.endswith(".safetensors") for f in os.listdir(path))


def discover_model_dirs(root):
    """Return the list of model folders to process under ``root``.

    If ``root`` is itself a model folder, return just ``[root]``. Otherwise treat
    it as a parent and return its immediate model subfolders (sorted).
    """
    root = os.path.abspath(root)
    if is_model_dir(root):
        return [root]
    subdirs = [
        os.path.join(root, name)
        for name in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, name))
    ]
    return [d for d in subdirs if is_model_dir(d)]


def read_safetensors_header(path):
    """Read a .safetensors header (the leading JSON dict) without loading data."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def human(nbytes):
    n = float(nbytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024 or unit == "TB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


def find_key(keys, candidates):
    for cand in candidates:
        if cand in keys:
            return cand
    return None


# --------------------------------------------------------------------------- #
# Removal decision
# --------------------------------------------------------------------------- #
def flat_cosine(a, b):
    """Cosine similarity between two tensors, flattened, computed in float32."""
    import torch

    a = a.flatten().float()
    b = b.flatten().float()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 0.0
    return float(torch.dot(a, b) / denom)


def removal_decision(head, embed, threshold):
    """Decide whether a tied output head may be deleted.

    Returns a human-readable reason string when removal is allowed.
    Raises RuntimeError when the head is tied but neither all-zero nor close to
    the input embeddings (an unexpected state we refuse to delete blindly).
    """
    import torch

    if bool(torch.eq(head, 0).all().item()):
        return "all-zero head"

    # A head holding inf/NaN/overflow garbage cannot be a real output layer and
    # is ignored at load (tie=true), so it is safe to drop. Two flavours seen in
    # the wild: (a) the tensor itself contains inf/NaN; (b) its values are finite
    # but so large that the L2 norm overflows float32 (e.g. max|w| ~ 1e36). Both
    # mean the buffer is junk, not a usable projection.
    headf = head.float()
    n_nan = int(torch.isnan(headf).sum().item())
    n_inf = int(torch.isinf(headf).sum().item())
    if n_nan or n_inf:
        return f"non-finite garbage head ({n_nan} NaN, {n_inf} inf)"
    head_norm = headf.norm()
    if not bool(torch.isfinite(head_norm).item()):
        return f"overflowing garbage head (max|w|={headf.abs().max().item():.3g}, norm overflowed)"

    if embed is None:
        raise RuntimeError(
            "output head is not all-zero and no input-embedding tensor was found "
            "to compare against (looked for: %s)" % ", ".join(EMBED_KEY_CANDIDATES)
        )

    if head.shape != embed.shape:
        raise RuntimeError(
            "output head shape %s != input embedding shape %s; refusing to delete"
            % (tuple(head.shape), tuple(embed.shape))
        )

    cos = flat_cosine(head, embed)
    if cos >= threshold:
        return f"head ~= embeddings (cosine={cos:.4f} >= {threshold})"

    raise RuntimeError(
        "tied output head is neither all-zero nor close to the input embeddings "
        f"(cosine={cos:.4f} < threshold {threshold}). This is unexpected — "
        "inspect the checkpoint manually before deleting."
    )


# --------------------------------------------------------------------------- #
# Rewriters (assume the removal decision has already been made)
# --------------------------------------------------------------------------- #
def rewrite_single_file(path, head_key, embed_key, threshold, dry_run):
    """Drop ``head_key`` from a single-file safetensors, in place.

    Returns (bytes_saved, reason). Raises if the decision says we must not delete.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(path, framework="pt") as f:
        head = f.get_tensor(head_key)
        embed = f.get_tensor(embed_key) if embed_key else None
        reason = removal_decision(head, embed, threshold)  # may raise
        saved = head.numel() * head.element_size()
        if dry_run:
            return saved, reason
        metadata = f.metadata() or {}
        tensors = {k: f.get_tensor(k) for k in f.keys() if k != head_key}

    tmp = path + ".tmp"
    save_file(tensors, tmp, metadata=metadata or {"format": "pt"})
    os.replace(tmp, path)
    return saved, reason


def rewrite_sharded(model_dir, index_path, head_key, embed_key, threshold, dry_run):
    """Drop ``head_key`` from the shard that holds it and update the index.

    Returns (bytes_saved, reason). Raises if the decision says we must not delete.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    head_shard = os.path.join(model_dir, weight_map[head_key])

    # Load the head, and the embeddings (which may live in another shard).
    with safe_open(head_shard, framework="pt") as f:
        head = f.get_tensor(head_key)
    embed = None
    if embed_key and embed_key in weight_map:
        embed_shard = os.path.join(model_dir, weight_map[embed_key])
        with safe_open(embed_shard, framework="pt") as f:
            embed = f.get_tensor(embed_key)

    reason = removal_decision(head, embed, threshold)  # may raise
    saved = head.numel() * head.element_size()
    if dry_run:
        return saved, reason

    # Rewrite the affected shard (atomically), dropping the head tensor.
    with safe_open(head_shard, framework="pt") as f:
        metadata = f.metadata() or {}
        tensors = {k: f.get_tensor(k) for k in f.keys() if k != head_key}
    tmp = head_shard + ".tmp"
    save_file(tensors, tmp, metadata=metadata or {"format": "pt"})
    os.replace(tmp, head_shard)

    # Update the index: drop the entry and decrement the reported total size.
    del weight_map[head_key]
    if "metadata" in index and "total_size" in index["metadata"]:
        index["metadata"]["total_size"] = int(index["metadata"]["total_size"]) - saved
    tmp_idx = index_path + ".tmp"
    with open(tmp_idx, "w") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp_idx, index_path)
    return saved, reason


# --------------------------------------------------------------------------- #
# Per-model driver
# --------------------------------------------------------------------------- #
def process_model(model_dir, threshold, dry_run):
    """Process one model folder. Returns a short status string. May raise."""
    name = os.path.basename(model_dir.rstrip("/"))
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    if not config.get("tie_word_embeddings", False):
        return f"[skip] {name}: tie_word_embeddings is not true — head is needed, left untouched."

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")

    if os.path.isfile(index_path):
        keys = list(json.load(open(index_path))["weight_map"].keys())
        head_key = find_key(keys, HEAD_KEY_CANDIDATES)
        embed_key = find_key(keys, EMBED_KEY_CANDIDATES)
        if head_key is None:
            return (
                f"[ok]   {name}: tied, no output-head tensor present — already clean."
            )
        saved, reason = rewrite_sharded(
            model_dir, index_path, head_key, embed_key, threshold, dry_run
        )
        layout = "sharded"
    elif os.path.isfile(single_path):
        keys = list(read_safetensors_header(single_path).keys())
        head_key = find_key(keys, HEAD_KEY_CANDIDATES)
        embed_key = find_key(keys, EMBED_KEY_CANDIDATES)
        if head_key is None:
            return (
                f"[ok]   {name}: tied, no output-head tensor present — already clean."
            )
        saved, reason = rewrite_single_file(
            single_path, head_key, embed_key, threshold, dry_run
        )
        layout = "single-file"
    else:
        return f"[skip] {name}: no model.safetensors(.index.json) found."

    verb = "would remove" if dry_run else "removed"
    return (
        f"[done] {name}: {verb} '{head_key}' ({reason}) from {layout} model "
        f"(freed {human(saved)})."
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Remove the redundant output weight matrix from tied-embedding "
        "model(s), in place."
    )
    ap.add_argument(
        "path", help="A model folder, or a parent folder of several models."
    )
    ap.add_argument(
        "--cosine-threshold",
        type=float,
        default=DEFAULT_COSINE_THRESHOLD,
        help="A non-zero head is removed only if its flattened cosine "
        "similarity with the input embeddings is >= this value "
        f"(default {DEFAULT_COSINE_THRESHOLD}).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without modifying any file.",
    )
    args = ap.parse_args()

    if not os.path.exists(args.path):
        sys.exit(f"error: path does not exist: {args.path}")

    models = discover_model_dirs(args.path)
    if not models:
        sys.exit(f"error: no model folder found under: {args.path}")

    print(f"{'DRY RUN — ' if args.dry_run else ''}processing {len(models)} model(s):\n")
    failures = 0
    for model_dir in models:
        try:
            status = process_model(model_dir, args.cosine_threshold, args.dry_run)
        except Exception as e:  # keep going across a batch, but record the failure
            failures += 1
            status = f"[FAIL] {os.path.basename(model_dir.rstrip('/'))}: {type(e).__name__}: {e}"
        print(" ", status)

    print()
    if args.dry_run:
        print("Dry run complete — nothing was modified.")
    if failures:
        print(f"{failures} model(s) raised an exception (see [FAIL] above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
