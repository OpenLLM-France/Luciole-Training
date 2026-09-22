"""Post-hoc fix for the "Table dist_<hash>" title leak in formatted TatQA configs.

format_context_chunks only renumbered "Paragraph*" titles, so BM25 table
distractors kept their raw "Table dist_<hash>" name while the gold table was
plain "Table"; the filename alone told the model which table was gold.

This renames every table chunk of a row to "Table 1..N" in context order
(context is already shuffled, so the gold position is random) and rewrites
chunk references accordingly:
  * "Table dist_<hash>" is unique -> replaced everywhere;
  * plain gold "Table" is a common word -> replaced only in identifier forms
    ("Table" in quotes, filename: Table, `Table`); free-text mentions are left
    untouched and counted.

Usage:
  venv/bin/python fix_table_titles.py -i tatqa_v4.parquet -o tatqa_v4_fixed.parquet
"""

import argparse
import collections
import re

import pyarrow as pa
import pyarrow.parquet as pq

FILENAME_TABLE_RE = re.compile(r"\* filename: (Table(?: dist_[0-9a-f]+)?)\n")
DIST_RE = re.compile(r"Table dist_[0-9a-f]+")
# Plain "Table" used as a chunk identifier (not followed by " dist_" / a number).
GOLD_ID_RES = [
    re.compile(r'"Table"'),
    re.compile(r"filename: Table(?![ \w])"),
    re.compile(r"`Table`"),
]
FREE_TABLE_RE = re.compile(r"\bTable\b(?! \d| dist_)")


def build_mapping(system: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for title in FILENAME_TABLE_RE.findall(system):
        if title in mapping:
            raise ValueError(f"duplicate table title {title!r}")
        mapping[title] = f"Table {len(mapping) + 1}"
    return mapping


def rewrite(text: str, mapping: dict[str, str], stats: collections.Counter) -> str:
    def dist_sub(m: re.Match) -> str:
        if m.group(0) not in mapping:
            stats["dist_unmapped"] += 1
            return m.group(0)
        stats["dist_replaced"] += 1
        return mapping[m.group(0)]

    text = DIST_RE.sub(dist_sub, text)
    gold = mapping.get("Table")
    for pattern in GOLD_ID_RES:
        def gold_sub(m: re.Match) -> str:
            if gold is None:
                stats["gold_ref_without_gold_table"] += 1
                return m.group(0)
            stats["gold_replaced"] += 1
            return m.group(0).replace("Table", gold)

        text = pattern.sub(gold_sub, text)
    return text


def fix_row(row: dict, stats: collections.Counter) -> dict:
    system = next(m["content"] for m in row["messages"] if m["role"] == "system")
    mapping = build_mapping(system)
    for msg in row["messages"]:
        msg["content"] = rewrite(msg["content"], mapping, stats)
        if msg["role"] == "assistant":
            stats["free_table_mentions_left"] += len(FREE_TABLE_RE.findall(msg["content"]))
    row["supporting_facts_titles"] = [
        mapping.get(t, t) for t in row["supporting_facts_titles"]
    ]
    stats["rows_with_dist"] += any(k.startswith("Table dist_") for k in mapping)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    table = pq.read_table(args.input)
    stats: collections.Counter = collections.Counter()
    rows = [fix_row(r, stats) for r in table.to_pylist()]

    leftover = sum(1 for r in rows for m in r["messages"] if "dist_" in m["content"])
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), args.output)
    print(f"rows: {len(rows)}")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")
    print(f"  messages still containing 'dist_': {leftover}")


if __name__ == "__main__":
    main()
