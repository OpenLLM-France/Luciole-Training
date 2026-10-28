#!/usr/bin/env python3
"""Dump the first example of every dataset declared in a NeMo-RL SFT config.

For each entry of `data.train_data_path` (and `data.val_data_path` with --split),
or, with the NeMo-RL v0.6.0 layout, of the `data_path` of every dataset of
`data.train` (and `data.validation`), the dataset class used by training
(`load_response_dataset`, e.g. OpenAIFormatDataset / OpenAIFormatDatasetV2) is
instantiated on that single entry alone, and its first sample goes through `get_formatted_message_log` with the same
options as `sft_preprocessor` -- so the "DEBUG: Individual message turns from
apply_chat_template" block printed by NeMo-RL at training start is reproduced for
every dataset, not just the first one.

One file per dataset is written in the output directory, containing the raw
sample (as returned by the dataset class) followed by the debug print.

Usage (inside the nemo-rl venv, with the env vars the config needs, e.g. OpenLLM_OUTPUT):
    python finetune/nemo-rl/show_first_examples.py configs/tool_calling/v17_rag.yaml out_dir
    python finetune/nemo-rl/show_first_examples.py <config> <out_dir> --split both --index 3
"""

import argparse
import contextlib
import copy
import io
import json
import re
import sys
from inspect import signature
from pathlib import Path


def slugify(text, maxlen=100):
    text = re.sub(r"^[a-z]+://", "", str(text))
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text[-maxlen:] if len(text) > maxlen else text


def entry_path(entry):
    if isinstance(entry, dict):
        return entry["path"]
    if isinstance(entry, (list, tuple)):
        return entry[0]
    return entry


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def truncate_entry(entry, n_rows, tmp_dir, name):
    """Copy the first `n_rows` records of the entry's first file into `tmp_dir`.

    Lets the dataset class run on a tiny file instead of loading the whole glob
    (cheap enough for a login node). Returns the entry pointing to the copy, with
    weight 1.0 (a weight < 1 would subsample the single row away).
    """
    import glob
    import gzip

    path = entry_path(entry)
    if path.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        files = sorted(HfFileSystem().glob(path[len("hf://"):]))
        files = ["hf://" + f for f in files]
    else:
        files = sorted(glob.glob(path, recursive=True))
    if not files:
        raise FileNotFoundError(f"glob matched no file: {path}")
    src = files[0]

    if ".parquet" in src:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if src.startswith("hf://"):
            from huggingface_hub import HfFileSystem

            f = HfFileSystem().open(src[len("hf://"):], "rb")
        else:
            f = src
        pf = pq.ParquetFile(f)
        batches, got = [], 0
        for batch in pf.iter_batches(batch_size=max(n_rows, 1)):
            batches.append(batch)
            got += batch.num_rows
            if got >= n_rows:
                break
        table = pa.Table.from_batches(batches).slice(0, n_rows)
        dst = Path(tmp_dir) / f"{name}.parquet"
        pq.write_table(table, dst)
    else:
        opener = gzip.open if src.endswith(".gz") else open
        dst = Path(tmp_dir) / f"{name}.jsonl"
        with opener(src, "rt", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
            written = 0
            for line in fin:
                if line.strip():
                    fout.write(line if line.endswith("\n") else line + "\n")
                    written += 1
                    if written >= n_rows:
                        break

    if isinstance(entry, dict):
        return {**entry, "path": str(dst), "weight": 1.0}
    return {"path": str(dst), "weight": 1.0}


def to_jsonable(obj):
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


def pretty_sample(sample):
    """Raw sample, with JSON-string fields (tools, chat_template_kwargs) expanded."""
    sample = dict(sample)
    for key in ("tools", "chat_template_kwargs"):
        if isinstance(sample.get(key), str):
            try:
                sample[key] = json.loads(sample[key])
            except ValueError:
                pass
    return json.dumps(to_jsonable(sample), indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="NeMo-RL SFT yaml config")
    ap.add_argument("out_dir", help="directory receiving one file per dataset")
    ap.add_argument("--split", choices=["train", "val", "both"], default="train")
    ap.add_argument("--index", type=int, default=0, help="which example to show (default: first)")
    ap.add_argument("--full-load", action="store_true",
                    help="load each dataset entirely instead of only its first rows "
                         "(slow, needs a compute node for big datasets)")
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.datasets import load_response_dataset
    from nemo_rl.data.interfaces import TaskDataSpec
    from nemo_rl.data.llm_message_utils import get_formatted_message_log

    # same resolver as examples/run_sft.py (configs use ${mul:a,b})
    OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)
    cfg = OmegaConf.to_container(_load_config(args.config), resolve=True)
    data_cfg = cfg["data"]
    seed = cfg.get("sft", {}).get("seed", 42) or 42
    tokenizer = get_tokenizer(cfg["policy"]["tokenizer"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="first_examples_")
    # v0.6.0: data.train / data.validation are dataset configs (or lists of
    # them) holding a `data_path`, completed by data.default
    is_v060 = "train" in data_cfg
    is_v2 = not is_v060 and data_cfg["dataset_name"] == "openai_format_v2"

    jobs = []  # (split, entry, dataset config for v0.6.0)
    splits = [("train", "train", "train_data_path"), ("val", "validation", "val_data_path")]
    for split, v060_key, key in splits:
        if args.split not in (split, "both"):
            continue
        if not is_v060:
            jobs += [(split, e, None) for e in as_list(data_cfg.get(key))]
            continue
        for ds_cfg in as_list(data_cfg.get(v060_key)):
            ds_cfg = {**(data_cfg.get("default") or {}), **ds_cfg}
            jobs += [(split, e, ds_cfg) for e in as_list(ds_cfg["data_path"])]

    n_ok = 0
    for i, (split, entry, ds_cfg) in enumerate(jobs):
        name = f"{i:02d}_{split}_{slugify(entry_path(entry))}"
        out_file = out_dir / f"{name}.txt"
        print(f"[{i + 1}/{len(jobs)}] {entry_path(entry)}", flush=True)
        buf = io.StringIO()
        try:
            # instantiate the training dataset class on this single entry
            small = entry
            if not args.full_load:  # only the first rows of the first file
                small = truncate_entry(entry, args.index + 1, tmp_dir, name)
            if is_v060:
                sub_cfg = {**copy.deepcopy(ds_cfg), "data_path": [small]}
                with contextlib.redirect_stdout(io.StringIO()):  # loader chatter
                    base = load_response_dataset(sub_cfg)
                ds = base.dataset
            else:
                sub_cfg = copy.deepcopy(data_cfg)
                sub_cfg["train_data_path"] = [small]
                # the v1 loaders require a validation path: reuse the entry
                sub_cfg["val_data_path"] = None if is_v2 else [small]
                with contextlib.redirect_stdout(io.StringIO()):  # loader chatter
                    base = load_response_dataset(sub_cfg, seed)
                ds = base.formatted_ds["train"]
            sample = ds[min(args.index, len(ds) - 1)]

            buf.write(f"# config : {args.config}\n# split  : {split}\n")
            buf.write(f"# entry  : {json.dumps(to_jsonable(entry), ensure_ascii=False)}\n")
            buf.write(f"# class  : {type(base).__name__}  ({len(ds)} samples, showing #{args.index})\n\n")
            buf.write("=" * 80 + "\nRAW SAMPLE\n" + "=" * 80 + "\n")
            buf.write(pretty_sample(sample) + "\n\n")

            # same call as sft_preprocessor; reset the once-only debug flag
            # (v0.6.0 prints it with debug=True instead)
            if hasattr(get_formatted_message_log, "_debug_printed"):
                del get_formatted_message_log._debug_printed
            debug_kw = {"debug": True} if "debug" in signature(get_formatted_message_log).parameters else {}
            dbg = io.StringIO()
            with contextlib.redirect_stdout(dbg):
                message_log = get_formatted_message_log(
                    sample["messages"],
                    tokenizer,
                    base.task_spec if hasattr(base, "task_spec") else TaskDataSpec("json_dataset"),
                    add_bos_token=data_cfg.get("add_bos", True),
                    add_eos_token=data_cfg.get("add_eos", True),
                    add_generation_prompt=data_cfg.get("add_generation_prompt", False),
                    tools=sample.get("tools", None),
                    chat_template_kwargs=sample.get("chat_template_kwargs", None),
                    **debug_kw,
                )
            buf.write(dbg.getvalue())
            n_tok = sum(len(m["token_ids"]) for m in message_log)
            buf.write(f"\nTotal tokens: {n_tok}\n")
            n_ok += 1
        except Exception as e:  # keep going: one broken dataset must not hide the others
            import traceback

            buf.write(f"# ERROR: {type(e).__name__}: {e}\n\n{traceback.format_exc()}")
            print(f"    ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        out_file.write_text(buf.getvalue())
        print(f"    -> {out_file}", flush=True)

    print(f"\n{n_ok}/{len(jobs)} dataset(s) written to {out_dir}")
    return 0 if n_ok == len(jobs) else 1


def _load_config(path):
    from nemo_rl.utils.config import load_config

    return load_config(path)


if __name__ == "__main__":
    sys.exit(main())
