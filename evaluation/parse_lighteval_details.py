#!/usr/bin/env python3
"""Pretty-print — and compare — the details of LLM evaluations stored in parquet.

Each row of a details parquet describes one evaluation sample with:
  - the question       (embedded in ``doc.query``),
  - the reference(s)    (``doc.choices``),
  - the model answer    (``model_response.text_post_processed[0]``),
  - the performance     (``metric``, a dict).

Several parquet files corresponding to the *same* evaluation data can be given
at once to compare models side by side. The files are checked for alignment:
they must describe the same samples, in the same order. A file may contain
fewer samples than the others, but its rows must match the *beginning* of the
data (same model input / same underlying question).

Usage:
    python parse_details.py FILE1.parquet [FILE2.parquet ...] [options]
"""

import argparse
import os
import re
import shutil
import sys


# --------------------------------------------------------------------------- #
# Terminal helpers
# --------------------------------------------------------------------------- #


def supports_color():
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


class C:
    """ANSI color codes (disabled when output is not a TTY)."""

    def __init__(self, enabled):
        e = enabled
        self.reset = "\033[0m" if e else ""
        self.bold = "\033[1m" if e else ""
        self.dim = "\033[2m" if e else ""
        self.italic = "\033[3m" if e else ""
        self.white = "\033[38;5;231m" if e else ""  # pure white
        self.label = "\033[38;5;110m" if e else ""  # soft blue
        self.question = "\033[38;5;223m" if e else ""  # warm sand
        self.reference = "\033[38;5;150m" if e else ""  # green
        self.answer = "\033[38;5;253m" if e else ""  # near white
        self.sep = "\033[38;5;240m" if e else ""  # grey rules
        self.good = "\033[38;5;114m" if e else ""  # green
        self.mid = "\033[38;5;179m" if e else ""  # amber
        self.bad = "\033[38;5;174m" if e else ""  # red
        # Distinct hues cycled per model in comparison mode.
        self.model_palette = (
            [
                "\033[38;5;75m",
                "\033[38;5;215m",
                "\033[38;5;114m",
                "\033[38;5;176m",
                "\033[38;5;180m",
                "\033[38;5;73m",
            ]
            if e
            else [""]
        )


def term_width(default=100):
    return min(shutil.get_terminal_size((default, 24)).columns, 120)


def short_tag(i):
    """Compact per-model marker: ①..⑳ then [21], [22], ..."""
    return chr(0x2460 + i) if i < 20 else f"[{i + 1}]"


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #

_CHAT_TOKENS = re.compile(r"<\|im_(start|end)\|>")
_ROLE_ONLY = {"assistant", "user", "system", "answer:", ""}
# "Question: <...> Answer:" — the passages precede it, so it is not the last line.
_QA_BLOCK = re.compile(r"Question:\s*(.*?)\s*\n\s*Answer:", re.DOTALL)
_Q_TAIL = re.compile(r"Question:\s*(.*)\Z", re.DOTALL)


def extract_question(doc):
    """Extract the question from ``doc.query``.

    LongBench-style prompts embed the question as ``Question: ... \\nAnswer:``
    after a wall of passages, so the question is not simply the last line.
    Falls back to the last meaningful line for other task formats.
    """
    query = doc.get("query", "") or ""
    if not query.strip():
        return "(no question found)"

    m = _QA_BLOCK.findall(query)
    if m:
        return m[-1].strip()

    m = _Q_TAIL.search(query)
    if m and m.group(1).strip():
        return m.group(1).strip()

    meaningful = []
    for line in query.splitlines():
        stripped = _CHAT_TOKENS.sub("", line).strip()
        if stripped.lower() in _ROLE_ONLY:
            continue
        meaningful.append(stripped)
    if not meaningful:
        return "(no question found)"
    return re.sub(r"^question\s*:\s*", "", meaningful[-1], flags=re.IGNORECASE)


_TURN_RE = re.compile(
    r"<\|im_start\|>\s*(\w+)\s*\n?(.*?)(?:<\|im_end\|>|\Z)", re.DOTALL
)


def extract_full_input(doc):
    """Return the full user turn of ``doc.query`` (system prompt excluded).

    When the prompt uses ``<|im_start|>role ... <|im_end|>`` chat markers, keep
    the content of the last ``user`` turn. Otherwise (few-shot / plain prompts
    with no role structure to strip) return the whole query unchanged.
    """
    query = doc.get("query", "") or ""
    if not query.strip():
        return "(no input found)"

    turns = _TURN_RE.findall(query)
    if turns:
        user_turns = [content for role, content in turns if role.lower() == "user"]
        if user_turns:
            return user_turns[-1].strip()
        non_system = [
            content.strip() for role, content in turns if role.lower() != "system"
        ]
        if any(non_system):
            return "\n".join(c for c in non_system if c).strip()
    return query.strip()


def extract_references(doc, sep="  |  "):
    """Return the reference answer(s) joined by ``sep``."""
    choices = doc.get("choices")
    if choices is None:
        return "(no reference)"
    refs = [str(c).strip() for c in list(choices) if str(c).strip()]
    return sep.join(refs) if refs else "(no reference)"


def extract_answer(model_response):
    """Return the post-processed model answer as a single stripped string."""
    tpp = model_response.get("text_post_processed")
    if tpp is None:
        return "(no answer)"
    try:
        text = tpp[0]
    except (IndexError, KeyError, TypeError):
        text = tpp
    return re.sub(r"\s+", " ", str(text)).strip() or "(empty answer)"


def ellipsize_middle(text, max_len):
    """Shorten ``text`` keeping head and tail, with an ellipsis in the middle."""
    if len(text) <= max_len:
        return text
    if max_len <= 5:
        return text[:max_len]
    ell = " […] "
    keep = max_len - len(ell)
    head = keep // 2 + keep % 2
    tail = keep // 2
    return text[:head] + ell + text[-tail:]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def format_metric_value(value):
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".") if value != 0 else "0"
    return str(value)


def metric_color(colors, value):
    if not isinstance(value, (int, float)):
        return colors.dim
    if value >= 0.999:
        return colors.good
    if value > 0:
        return colors.mid
    return colors.bad


def format_metrics(metric, colors):
    if not metric:
        return f"{colors.dim}(no metrics){colors.reset}"
    items = list(metric.items())
    numeric = [(k, v) for k, v in items if isinstance(v, (int, float))]
    other = [(k, v) for k, v in items if not isinstance(v, (int, float))]
    parts = []
    for key, value in numeric + other:
        col = metric_color(colors, value)
        parts.append(f"{col}{key}={format_metric_value(value)}{colors.reset}")
    return "  ".join(parts)


# --------------------------------------------------------------------------- #
# Loading & alignment
# --------------------------------------------------------------------------- #


def derive_labels(paths):
    """Build short, distinct labels by keeping the path parts that differ."""
    if len(paths) == 1:
        return [os.path.basename(paths[0])]
    parts = [os.path.normpath(p).split(os.sep) for p in paths]
    common = set(parts[0])
    for pl in parts[1:]:
        common &= set(pl)
    labels = []
    for pl in parts:
        uniq = [x for x in pl if x not in common]
        labels.append("/".join(uniq) if uniq else os.path.basename(os.sep.join(pl)))
    # Guarantee uniqueness even if derivation collapses.
    seen = {}
    out = []
    for lab in labels:
        seen[lab] = seen.get(lab, 0) + 1
        out.append(f"{lab}#{seen[lab]}" if labels.count(lab) > 1 else lab)
    return out


def load_parquet(path, match_on, cut_long_input=False):
    """Load a parquet into a list of per-sample dicts + the raw match keys."""
    import pandas as pd

    df = pd.read_parquet(path)
    samples = []
    keys = []
    for _, row in df.iterrows():
        model_response = row.get("model_response", {}) or {}
        doc = row.get("doc", {}) or {}
        metric = dict(row.get("metric", {}) or {})
        if match_on == "input":
            key = model_response.get("input", "") or ""
        else:
            key = doc.get("query", "") or ""
        keys.append(key)
        samples.append(
            {
                "question": (
                    extract_question(doc) if cut_long_input else extract_full_input(doc)
                ),
                "references": extract_references(doc),
                "answer": extract_answer(model_response),
                "metric": metric,
            }
        )
    return samples, keys


def _snippet(text, limit=200):
    """Collapse whitespace and truncate, for error messages."""
    return re.sub(r"\s+", " ", text[:limit])


def check_alignment(all_keys, labels):
    """Verify every file matches the reference on their common prefix.

    The reference is the file with the most samples. Every other file must be a
    prefix of it (same keys, same order). Returns nothing; exits on mismatch.
    """
    ref_idx = max(range(len(all_keys)), key=lambda i: len(all_keys[i]))
    ref = all_keys[ref_idx]
    for i, keys in enumerate(all_keys):
        if i == ref_idx:
            continue
        for j, key in enumerate(keys):
            if key != ref[j]:
                sys.exit(
                    "\n".join(
                        [
                            "ERROR: parquet files are NOT the same evaluation data.",
                            f"  Mismatch at sample #{j + 1}:",
                            f"    [{labels[ref_idx]}] {_snippet(ref[j])!r}",
                            f"    [{labels[i]}] {_snippet(key)!r}",
                            "  The beginning of every file must describe the same samples.",
                        ]
                    )
                )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def wrap(text, width, indent):
    """Word-wrap ``text`` to ``width`` columns, indenting continuation lines."""
    import textwrap

    lines = []
    for i, para in enumerate(text.split("\n")):
        wrapped = textwrap.wrap(para, width=width) or [""]
        for j, line in enumerate(wrapped):
            prefix = "" if (i == 0 and j == 0) else indent
            lines.append(prefix + line)
    return "\n".join(lines)


def render_shared_field(label, value, value_color, colors, width, label_w):
    text_w = max(20, width - label_w - 1)
    indent = " " * (label_w + 1)
    wrapped = wrap(value, text_w, indent)
    return (
        f"{colors.label}{colors.bold}{label:<{label_w}}{colors.reset} "
        f"{value_color}{wrapped}{colors.reset}"
    )


def render_sample(
    idx, total, question, references, per_model, colors, width, model_colors, max_answer
):
    label_w = 11
    out = []
    header = f"Sample {idx + 1}/{total}"
    out.append(
        f"{colors.white}┄┄ {header} "
        f"{'┄' * max(0, width - len(header) - 4)}{colors.reset}"
    )
    out.append(
        render_shared_field(
            "Question :", question, colors.question, colors, width, label_w
        )
    )
    out.append(
        render_shared_field(
            "Reference :", references, colors.reference, colors, width, label_w
        )
    )

    # Per-model output + metrics.
    metric_label = f"{colors.label}{colors.bold}{'Metric :':<{label_w}}{colors.reset} "
    for m in per_model:
        mcol = model_colors[m["mi"]]
        tag = f"{m['short']} " if m["short"] else ""
        if m["present"]:
            if m["short"]:
                out.append(f"{mcol}{colors.bold}▌ {tag}{colors.reset}")
            answer = ellipsize_middle(m["answer"], max_answer)
            out.append(
                render_shared_field(
                    "Output :", answer, colors.answer, colors, width, label_w
                )
            )
            metrics = format_metrics(m["metric"], colors)
            out.append(f"{metric_label}{metrics}")
        else:
            out.append(
                f"{mcol}{colors.bold}▌ {tag}{colors.reset}"
                f"{colors.dim}(no data for this sample){colors.reset}"
            )
    return "\n".join(out)


def render_summary(models, colors, width, model_colors):
    print()
    print(f"{colors.sep}{'═' * width}{colors.reset}")
    print(f"{colors.bold}{colors.label}  Summary{colors.reset}")

    # Collect metric keys in first-seen order.
    metric_keys = []
    for m in models:
        for k in m["totals"]:
            if k not in metric_keys:
                metric_keys.append(k)

    for k in metric_keys:
        print()
        print(f"  {colors.label}{colors.bold}{k}{colors.reset}")
        # Determine best average for highlighting.
        avgs = {}
        for mi, m in enumerate(models):
            vals = m["totals"].get(k)
            if vals:
                avgs[mi] = sum(vals) / len(vals)
        best = max(avgs.values()) if avgs else None
        for mi, m in enumerate(models):
            mcol = model_colors[mi]
            tag = f"{m['short']} " if m["short"] else ""
            if mi not in avgs:
                print(f"    {mcol}{tag}{colors.reset} {colors.dim}n/a{colors.reset}")
                continue
            avg = avgs[mi]
            bar_w = 28
            filled = int(round(avg * bar_w)) if 0 <= avg <= 1 else 0
            vcol = metric_color(colors, avg)
            bar = (
                f"{vcol}{'█' * filled}{colors.dim}{'░' * (bar_w - filled)}{colors.reset}"
                if 0 <= avg <= 1
                else ""
            )
            star = (
                f" {colors.good}{colors.bold}★{colors.reset}"
                if best is not None and abs(avg - best) < 1e-9 and len(avgs) > 1
                else ""
            )
            n = len(m["totals"][k])
            print(
                f"    {mcol}{tag}{colors.reset} "
                f"{vcol}{avg:.4f}{colors.reset}  {bar} {colors.dim}(n={n}){colors.reset}{star}"
            )
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Pretty-print and compare LLM evaluation details from parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "parquets",
        nargs="+",
        help="One or more details parquet files (same eval data).",
    )
    parser.add_argument(
        "--labels", default=None, help="Comma-separated model labels (one per parquet)."
    )
    parser.add_argument(
        "--match-on",
        choices=["query", "input"],
        default="query",
        help="Field used to check the files describe the same data.",
    )
    parser.add_argument(
        "-n", "--limit", type=int, default=None, help="Only show the first N samples."
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Index of the first sample to show."
    )
    parser.add_argument(
        "--max-answer",
        type=int,
        default=600,
        help="Max answer length before middle ellipsis.",
    )
    parser.add_argument(
        "--width", type=int, default=None, help="Wrap width (default: terminal width)."
    )
    parser.add_argument(
        "--no-color", action="store_true", help="Disable colored output."
    )
    parser.add_argument(
        "--sort",
        choices=["none", "asc", "desc"],
        default="none",
        help="Sort samples by the first model's first metric.",
    )
    parser.add_argument(
        "--diff-only",
        action="store_true",
        help="Only show samples where models disagree on a metric.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip per-sample output; print only the summary.",
    )
    parser.add_argument(
        "--cut_long_input",
        action="store_true",
        help="Show only the extracted question instead of the full user turn.",
    )
    args = parser.parse_args()

    for p in args.parquets:
        if not os.path.isfile(p):
            parser.error(f"File not found: {p}")

    try:
        import pandas  # noqa: F401
    except ImportError:
        sys.exit("pandas is required: pip install pandas pyarrow")

    colors = C(supports_color() and not args.no_color)
    width = args.width or term_width()

    # Labels.
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(args.parquets):
            parser.error(
                f"--labels has {len(labels)} entries but "
                f"{len(args.parquets)} parquet(s) were given."
            )
    else:
        labels = derive_labels(args.parquets)

    # Load & align.
    loaded = [
        load_parquet(p, args.match_on, args.cut_long_input) for p in args.parquets
    ]
    all_samples = [s for s, _ in loaded]
    all_keys = [k for _, k in loaded]
    check_alignment(all_keys, labels)

    lengths = [len(s) for s in all_samples]
    n_max = max(lengths)

    single = len(all_samples) == 1
    models = [
        {
            "label": labels[i],
            "short": "" if single else short_tag(i),
            "samples": all_samples[i],
            "n": lengths[i],
            "totals": {},
        }
        for i in range(len(all_samples))
    ]
    for m in models:
        for s in m["samples"]:
            for k, v in s["metric"].items():
                if isinstance(v, (int, float)):
                    m["totals"].setdefault(k, []).append(v)

    model_colors = [
        colors.model_palette[i % len(colors.model_palette)] for i in range(len(models))
    ]

    # Build the aligned row list (question/reference from the longest file).
    ref_idx = lengths.index(n_max)
    ref_samples = all_samples[ref_idx]

    def first_metric(sample):
        for v in sample["metric"].values():
            if isinstance(v, (int, float)):
                return v
        return None

    rows = []
    for j in range(n_max):
        per_model = []
        metric_vals = []
        for mi, m in enumerate(models):
            present = j < m["n"]
            s = m["samples"][j] if present else None
            if present:
                metric_vals.append(first_metric(s))
            per_model.append(
                {
                    "mi": mi,
                    "short": m["short"],
                    "present": present,
                    "answer": s["answer"] if present else "",
                    "metric": s["metric"] if present else {},
                }
            )
        vals = [v for v in metric_vals if v is not None]
        disagree = len(vals) > 1 and (max(vals) - min(vals) > 1e-9)
        rows.append(
            {
                "idx": j,
                "question": ref_samples[j]["question"],
                "references": ref_samples[j]["references"],
                "per_model": per_model,
                "sort_key": (
                    metric_vals[0]
                    if metric_vals and metric_vals[0] is not None
                    else float("-inf")
                ),
                "disagree": disagree,
            }
        )

    if args.diff_only:
        rows = [r for r in rows if r["disagree"]]
    if args.sort != "none":
        rows.sort(key=lambda r: r["sort_key"], reverse=(args.sort == "desc"))

    start = max(0, args.start)
    end = len(rows) if args.limit is None else min(len(rows), start + args.limit)
    shown = rows[start:end]

    # Banner.
    print()
    print(f"{colors.sep}╭{'─' * (width - 2)}╮{colors.reset}")
    if len(models) == 1:
        print(
            f"{colors.bold}{colors.label}  {ellipsize_middle(os.path.basename(args.parquets[0]), width - 4)}{colors.reset}"
        )
    else:
        print(
            f"{colors.bold}{colors.label}  Comparing {len(models)} models{colors.reset}"
        )
        for mi, m in enumerate(models):
            print(
                f"  {model_colors[mi]}{colors.bold}▌ {m['short']}{colors.reset} "
                f"{model_colors[mi]}{m['label']}{colors.reset} "
                f"{colors.dim}({m['n']} samples){colors.reset}"
            )
    if len(set(lengths)) > 1:
        print(
            f"{colors.dim}  sample counts differ: {lengths} — comparing on the common prefix{colors.reset}"
        )
    span = f"showing {start + 1}..{end} of {len(rows)}"
    if args.diff_only:
        span += " (disagreements only)"
    print(f"{colors.dim}  {span}{colors.reset}")
    print(f"{colors.sep}╰{'─' * (width - 2)}╯{colors.reset}")

    if not args.summary_only:
        for r in shown:
            print()
            print(
                render_sample(
                    r["idx"],
                    n_max,
                    r["question"],
                    r["references"],
                    r["per_model"],
                    colors,
                    width,
                    model_colors,
                    args.max_answer,
                )
            )

    render_summary(models, colors, width, model_colors)


if __name__ == "__main__":
    main()
