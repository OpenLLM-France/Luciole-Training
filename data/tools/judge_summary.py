# Read every quality_summary.json produced by score_preference_pairs.py under a
# root directory and print a color-coded table: one row per dataset (the parent
# directory name), one column per top-level metric. Cells are shaded as a
# heatmap so noisy / low-quality datasets stand out at a glance.
#
#   python judge_summary.py [ROOT]
#   python judge_summary.py /data-local/ogouvert/OpenLLM-BPI-output --per-aspect
#   python judge_summary.py --sort agreement_rate

import argparse
import json
import pathlib

DEFAULT_ROOT = "/data-local/ogouvert/OpenLLM-BPI-output"

# Preferred column order + short display headers for the known top-level metrics.
# Anything not listed here is appended in discovery order, headered by its key.
METRICS = [
    ("n_pairs", "pairs"),
    ("n_parse_failed", "parse_fail"),
    ("n_comparable", "comp"),
    ("agreement_rate", "agree"),
    ("chosen_wins", "c_win"),
    ("ties", "ties"),
    ("rejected_wins", "r_win"),
    ("tie_rate", "tie_r"),
    ("mean_chosen", "m_chos"),
    ("mean_rejected", "m_rej"),
    ("mean_score_gap", "gap"),
]

# How to colorize each metric: (lo, hi, higher_is_better, diverging).
# A value is normalized into [lo, hi]; for a diverging metric the midpoint
# (here always 0) is the neutral point. Metrics absent from this map (e.g. raw
# counts) are printed without a heatmap background.
COLOR_RULES = {
    "agreement_rate": (0.0, 1.0, True, False),
    "disagreement_rate": (0.0, 1.0, False, False),
    "tie_rate": (0.0, 1.0, False, False),
    "mean_chosen": (1.0, 5.0, True, False),
    # same scale/direction as mean_chosen so equal scores get the same color
    "mean_rejected": (1.0, 5.0, True, False),
    "mean_score_gap": (-1.0, 1.0, True, True),
}


def _bg(r, g, b, text):
    """Wrap text in a 24-bit background color, choosing black/white fg by luminance."""
    fg = "0;0;0" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "255;255;255"
    return f"\033[48;2;{r};{g};{b}m\033[38;2;{fg}m{text}\033[0m"


def _heat(t):
    """Map t in [0,1] to a red(0) -> yellow(0.5) -> green(1) RGB tuple."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        r, g, b = 220, int(40 + 2 * t * 175), 40
    else:
        r, g, b = int(220 - 2 * (t - 0.5) * 180), 215, 40
    return r, g, b


def colorize(key, value, raw):
    """Return the cell string, heatmap-shaded if a color rule applies."""
    if key not in COLOR_RULES or not isinstance(raw, (int, float)):
        return value
    lo, hi, higher_better, diverging = COLOR_RULES[key]
    if diverging:
        # symmetric around 0: clamp to [lo, hi], 0 -> neutral (0.5)
        span = max(abs(lo), abs(hi)) or 1.0
        t = 0.5 + 0.5 * max(-1.0, min(1.0, raw / span))
    else:
        t = (raw - lo) / (hi - lo) if hi != lo else 0.5
    if not higher_better:
        t = 1.0 - t
    return _bg(*_heat(t), value)


RATE_KEYS = {"agreement_rate", "tie_rate", "disagreement_rate"}


def fmt(raw):
    if isinstance(raw, bool) or raw is None:
        return "-" if raw is None else str(raw)
    if isinstance(raw, float):
        return f"{raw:.3f}"
    return str(raw)


def fmt_metric(key, raw):
    """Format a value, rendering rate metrics as a percentage."""
    if key in RATE_KEYS and isinstance(raw, (int, float)):
        return f"{raw * 100:.0f}%"
    return fmt(raw)


def load_rows(root):
    rows = []
    for path in sorted(pathlib.Path(root).rglob("quality_summary.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  skipping {path}: {e}")
            continue
        rows.append((path.parent.name, data))
    return rows


def build_columns(rows, per_aspect):
    """Return ordered list of (key, header, getter) columns present in the data."""
    known = {k: h for k, h in METRICS}
    seen = []
    for _, data in rows:
        for k, v in data.items():
            if k == "per_aspect":
                continue
            if k not in seen and not isinstance(v, (dict, list)):
                seen.append(k)
    ordered = [k for k, _ in METRICS if k in seen] + [k for k in seen if k not in known]
    cols = [(k, known.get(k, k), (lambda d, k=k: d.get(k))) for k in ordered]

    if per_aspect:
        aspects = []
        for _, data in rows:
            for a in (data.get("per_aspect") or {}):
                if a not in aspects:
                    aspects.append(a)
        for a in aspects:
            key = "agreement_rate"  # color these like a top-level agreement_rate
            header = f"{a[:6]}.agr"
            cols.append((key, header, (lambda d, a=a: (d.get("per_aspect") or {}).get(a, {}).get("agreement_rate"))))
    return cols


def main():
    parser = argparse.ArgumentParser(description="Color-coded table of DPO judge quality summaries")
    parser.add_argument("root", nargs="?", default=DEFAULT_ROOT, help="Directory to search for quality_summary.json files")
    parser.add_argument("--per-aspect", action="store_true", help="Append one per-aspect agreement_rate column per aspect")
    parser.add_argument("--sort", default=None, help="Metric key to sort rows by (descending)")
    parser.add_argument("--csv", default=None, help="Also write the table (no colors) to this CSV path")
    parser.add_argument("--html", default=None, help="Also write a color-coded HTML table to this path")
    parser.add_argument("--plot", default=None, help="Also save a heatmap figure (PNG) to this path")
    args = parser.parse_args()

    rows = load_rows(args.root)
    if not rows:
        print(f"No quality_summary.json found under {args.root}")
        return

    cols = build_columns(rows, args.per_aspect)
    if args.sort:
        rows.sort(key=lambda r: (r[1].get(args.sort) is None, -(r[1].get(args.sort) or 0)))

    name_w = max(len("dataset"), max(len(name) for name, _ in rows))
    cell_w = {h: max(len(h), 6) for _, h, _ in cols}

    # group the columns so the table is visually split: agreement scores | means | gaps
    def group_of(key):
        if key in ("mean_chosen", "mean_rejected"):
            return "mean"
        if key == "mean_score_gap" or key.endswith("gap"):
            return "gap"
        return "agree"

    def join_cols(render):
        out, prev = [], None
        for key, h, getter in cols:
            sep = "  │ " if (prev is not None and group_of(key) != group_of(prev)) else "  "
            out.append(sep + render(key, h, getter))
            prev = key
        return "".join(out)

    # header
    header = "dataset".ljust(name_w) + join_cols(lambda k, h, g: h.rjust(cell_w[h]))
    print(header)
    print("-" * len(header))

    for name, data in rows:
        def cell(key, h, getter, _d=data):
            raw = getter(_d)
            return colorize(key, fmt_metric(key, raw).rjust(cell_w[h]), raw)
        print(name.ljust(name_w) + join_cols(cell))

    print(f"\n{len(rows)} datasets · {args.root}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset"] + [h for _, h, _ in cols])
            for name, data in rows:
                w.writerow([name] + [fmt_metric(key, getter(data)) for key, _, getter in cols])
        print(f"CSV written to {args.csv}")

    if args.html:
        def cell_html(key, raw):
            txt = fmt_metric(key, raw)
            if key in COLOR_RULES and isinstance(raw, (int, float)):
                lo, hi, hb, dv = COLOR_RULES[key]
                if dv:
                    span = max(abs(lo), abs(hi)) or 1.0
                    t = 0.5 + 0.5 * max(-1.0, min(1.0, raw / span))
                else:
                    t = (raw - lo) / (hi - lo) if hi != lo else 0.5
                if not hb:
                    t = 1.0 - t
                r, g, b = _heat(t)
                fg = "#000" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "#fff"
                return f'<td style="background:rgb({r},{g},{b});color:{fg};text-align:right">{txt}</td>'
            return f'<td style="text-align:right">{txt}</td>'

        head = "".join(f"<th>{h}</th>" for _, h, _ in cols)
        body = ""
        for name, data in rows:
            cells = "".join(cell_html(key, getter(data)) for key, _, getter in cols)
            body += f"<tr><td>{name}</td>{cells}</tr>\n"
        html = (
            "<html><head><meta charset='utf-8'><style>"
            "table{border-collapse:collapse;font-family:monospace}"
            "td,th{padding:3px 8px;border:1px solid #ccc}th{background:#eee}"
            "</style></head><body><table>\n"
            f"<tr><th>dataset</th>{head}</tr>\n{body}</table>"
            f"<p>{len(rows)} datasets · {args.root}</p></body></html>"
        )
        with open(args.html, "w") as f:
            f.write(html)
        print(f"HTML written to {args.html}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Heatmap columns as (clear label, color-rule key, getter). Only metrics
        # with a color rule are shown, plus one score-gap column per judge aspect.
        pcols = [
            ("agreement rate", "agreement_rate", lambda d: d.get("agreement_rate")),
            ("tie rate", "tie_rate", lambda d: d.get("tie_rate")),
            ("disagreement rate", "disagreement_rate",
             lambda d: (d.get("rejected_wins") / d["n_comparable"]) if d.get("n_comparable") else None),
            ("mean chosen", "mean_chosen", lambda d: d.get("mean_chosen")),
            ("mean rejected", "mean_rejected", lambda d: d.get("mean_rejected")),
            ("overall score gap", "mean_score_gap", lambda d: d.get("mean_score_gap")),
        ]
        aspects = []
        for _, data in rows:
            for a in (data.get("per_aspect") or {}):
                if a not in aspects:
                    aspects.append(a)
        for a in aspects:
            label = "gap: " + a.replace("_", " ")
            pcols.append((label, "mean_score_gap",
                          lambda d, a=a: (d.get("per_aspect") or {}).get(a, {}).get("mean_score_gap")))

        names = [name for name, _ in rows]
        norm = []   # normalized 0..1 (1 = good) for the colors
        text = []   # original values for the annotations
        for _, data in rows:
            nrow, trow = [], []
            for _, key, getter in pcols:
                raw = getter(data)
                if isinstance(raw, (int, float)):
                    lo, hi, hb, dv = COLOR_RULES[key]
                    if dv:
                        span = max(abs(lo), abs(hi)) or 1.0
                        t = 0.5 + 0.5 * max(-1.0, min(1.0, raw / span))
                    else:
                        t = (raw - lo) / (hi - lo) if hi != lo else 0.5
                    if not hb:
                        t = 1.0 - t
                    nrow.append(t)
                    trow.append(fmt_metric(key, raw))
                else:
                    nrow.append(float("nan"))
                    trow.append("")
            norm.append(nrow)
            text.append(trow)

        # split the columns into contiguous groups so each renders as its own
        # panel, leaving real whitespace between rates | means | gaps
        def pgroup(label, color_key):
            if label.startswith("gap:"):           # per-aspect gaps
                return "gap"
            if color_key in ("mean_chosen", "mean_rejected") or label == "overall score gap":
                return "mean"
            return "rate"
        groups = []
        for idx, (label, key, _) in enumerate(pcols):
            g = pgroup(label, key)
            if groups and groups[-1][0] == g:
                groups[-1][1].append(idx)
            else:
                groups.append((g, [idx]))

        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad("#dddddd")
        fig, axes = plt.subplots(
            1, len(groups),
            figsize=(1.1 * len(pcols) + 3, 0.45 * len(names) + 1.8),
            gridspec_kw={"width_ratios": [len(idxs) for _, idxs in groups], "wspace": 0.08},
            sharey=True,
        )
        if len(groups) == 1:
            axes = [axes]
        for ax, (_, idxs) in zip(axes, groups):
            sub = [[norm[i][j] for j in idxs] for i in range(len(names))]
            ax.imshow(sub, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(idxs)))
            ax.set_xticklabels([pcols[j][0] for j in idxs], rotation=30, ha="left", fontsize=9)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            for r in range(len(names)):
                for c, j in enumerate(idxs):
                    ax.text(c, r, text[r][j], ha="center", va="center", fontsize=7, color="black")
            ax.set_xlim(-0.5, len(idxs) - 0.5)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[0].set_yticks(range(len(names)))
        axes[0].set_yticklabels(names, fontsize=8)
        fig.suptitle("DPO quality pairs - Llama 3.3 70B judge", fontsize=11, y=1.04)
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot written to {args.plot}")


if __name__ == "__main__":
    main()
