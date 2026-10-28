#!/usr/bin/env python3
"""Dump the first example of every dataset declared in a NeMo-RL DPO config.

DPO counterpart of `show_first_examples.py`. For each entry of the `data_path`
of `data.train` (and `data.validation` with --split), the preference dataset
class used by training (`load_preference_dataset`, e.g.
OpenAIPreferenceDatasetV2) is instantiated on that single entry alone, and the
chosen and rejected conversations of its first sample go through
`get_formatted_message_log` with the same options as `preference_preprocessor`
-- so the rendered chat template (system prompt, tools, thinking) is shown for
every dataset.

One file per dataset is written in the output directory, containing the raw
sample (as returned by the dataset class) followed by the debug print of the
chosen and the rejected conversation.

Usage (inside the nemo-rl venv, with the env vars the config needs, e.g. HF_HOME):
    python finetune/nemo-rl/show_first_examples_dpo.py configs/hybrid_thinking/dpo_math_8b_luciole_hybrid_thinking.yaml out_dir
    python finetune/nemo-rl/show_first_examples_dpo.py <config> <out_dir> --split both --index 3
"""

import argparse
import contextlib
import copy
import io
import json
import sys
from inspect import signature
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from show_first_examples import (  # noqa: E402
    _load_config,
    as_list,
    entry_path,
    pretty_sample,
    slugify,
    to_jsonable,
    truncate_entry,
)


def render(messages, tokenizer, task_spec, chat_template_kwargs, debug_kw):
    """Same call as `preference_preprocessor`, with the debug print captured."""
    from nemo_rl.data.llm_message_utils import get_formatted_message_log

    # reset the once-only debug flag (v0.6.0 prints it with debug=True instead)
    if hasattr(get_formatted_message_log, "_debug_printed"):
        del get_formatted_message_log._debug_printed
    dbg = io.StringIO()
    with contextlib.redirect_stdout(dbg):
        message_log = get_formatted_message_log(
            messages,
            tokenizer,
            task_spec,
            chat_template_kwargs=chat_template_kwargs,
            **debug_kw,
        )
    return message_log, dbg.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="NeMo-RL DPO yaml config")
    ap.add_argument("out_dir", help="directory receiving one file per dataset")
    ap.add_argument("--split", choices=["train", "val", "both"], default="train")
    ap.add_argument("--index", type=int, default=0, help="which example to show (default: first)")
    ap.add_argument("--full-load", action="store_true",
                    help="load each dataset entirely instead of only its first rows "
                         "(slow, needs a compute node for big datasets)")
    args = ap.parse_args()

    from omegaconf import OmegaConf

    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.datasets import load_preference_dataset
    from nemo_rl.data.llm_message_utils import get_formatted_message_log

    # same resolver as examples/run_dpo.py (configs use ${mul:a,b})
    OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)
    cfg = OmegaConf.to_container(_load_config(args.config), resolve=True)
    data_cfg = cfg["data"]
    max_len = data_cfg.get("max_input_seq_length")
    tokenizer = get_tokenizer(cfg["policy"]["tokenizer"])
    debug_kw = {"debug": True} if "debug" in signature(get_formatted_message_log).parameters else {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="first_examples_dpo_")

    jobs = []  # (split, entry, dataset config)
    for split, key in (("train", "train"), ("val", "validation")):
        if args.split not in (split, "both"):
            continue
        for ds_cfg in as_list(data_cfg.get(key)):
            ds_cfg = {**(data_cfg.get("default") or {}), **ds_cfg}
            if "data_path" not in ds_cfg:
                raise KeyError(
                    f"data.{key} has no `data_path` (keys: {sorted(ds_cfg)}), "
                    f"the {ds_cfg.get('dataset_name')} class would not load it."
                )
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
            sub_cfg = {**copy.deepcopy(ds_cfg), "data_path": [small]}
            with contextlib.redirect_stdout(io.StringIO()):  # loader chatter
                base = load_preference_dataset(sub_cfg)
            ds = base.dataset
            sample = ds[min(args.index, len(ds) - 1)]

            buf.write(f"# config : {args.config}\n# split  : {split}\n")
            buf.write(f"# entry  : {json.dumps(to_jsonable(entry), ensure_ascii=False)}\n")
            buf.write(f"# class  : {type(base).__name__}  ({len(ds)} samples, showing #{args.index})\n\n")
            buf.write("=" * 80 + "\nRAW SAMPLE\n" + "=" * 80 + "\n")
            buf.write(pretty_sample(sample) + "\n\n")

            # same ranking and calls as preference_preprocessor
            completions = sorted(sample["completions"], key=lambda c: c["rank"])
            n_tok = {}
            for label, completion in (("CHOSEN", completions[0]), ("REJECTED", completions[1])):
                message_log, dbg = render(
                    sample["context"] + completion["completion"],
                    tokenizer,
                    base.task_spec,
                    sample.get("chat_template_kwargs", None),
                    debug_kw,
                )
                n_tok[label] = sum(len(m["token_ids"]) for m in message_log)
                buf.write("=" * 80 + f"\n{label}\n" + "=" * 80 + "\n")
                buf.write(dbg)
                buf.write(f"\nTotal tokens: {n_tok[label]}\n\n")
            too_long = max_len is not None and max(n_tok.values()) > max_len
            buf.write(
                f"# context: {len(sample['context'])} message(s), tokens: {n_tok}"
                + (f"  -- exceeds max_input_seq_length={max_len}, masked out" if too_long else "")
                + "\n"
            )
            n_ok += 1
        except Exception as e:  # keep going: one broken dataset must not hide the others
            import traceback

            buf.write(f"# ERROR: {type(e).__name__}: {e}\n\n{traceback.format_exc()}")
            print(f"    ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        out_file.write_text(buf.getvalue())
        print(f"    -> {out_file}", flush=True)

    print(f"\n{n_ok}/{len(jobs)} dataset(s) written to {out_dir}")
    return 0 if n_ok == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
