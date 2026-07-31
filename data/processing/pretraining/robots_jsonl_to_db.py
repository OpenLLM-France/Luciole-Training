"""Convert RobotsMerger's robotstxt_dict.jsonl into the robots.db SQLite store.

RobotsTxtFilter reads `robots.db` from its `robots_txt_path` folder when one is present,
and otherwise falls back to loading the whole jsonl into memory - which is ~20-30 GB for
a full dump, in *every* worker. This turns the merger's output into the keyed store so
lookups are per-domain from disk (~17 us) at constant memory.

Streams the jsonl, so its own memory use is flat regardless of store size.

    python robots_jsonl_to_db.py <folder produced by RobotsMerger>
    python robots_jsonl_to_db.py <folder> --output /somewhere/else/robots.db
"""

import argparse
import os
import sqlite3
import time

import orjson

BATCH = 50_000
# Column names/types RobotsTxtFilter expects: it selects `text, status_class` keyed on
# `fqdn`, and probes PRAGMA table_info(robots) to detect whether status_class exists.
SCHEMA = "CREATE TABLE robots (fqdn TEXT PRIMARY KEY, text TEXT, date TEXT, status_class TEXT)"


def rows(jsonl_path):
    with open(jsonl_path, "rb") as f:
        for line in f:
            e = orjson.loads(line)
            yield e["fqdn"], e["text"], e["date"], e.get("status_class")


def batched(it, n):
    batch = []
    for row in it:
        batch.append(row)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def convert(jsonl_path, db_path):
    if os.path.exists(db_path):
        os.remove(db_path)  # a stale db silently shadows the jsonl in the filter
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-1000000")  # 1 GB page cache while building
    conn.execute(SCHEMA)

    n, start = 0, time.perf_counter()
    for batch in batched(rows(jsonl_path), BATCH):
        # OR REPLACE keeps this idempotent if a jsonl ever repeats an fqdn
        conn.executemany("INSERT OR REPLACE INTO robots VALUES (?, ?, ?, ?)", batch)
        n += len(batch)
        if n % (BATCH * 20) == 0:
            print(f"  {n:,} rows ({n / (time.perf_counter() - start):,.0f}/s)", flush=True)
    conn.commit()
    conn.close()
    return n, time.perf_counter() - start


def verify(db_path, expected_rows):
    """Read the db back exactly the way RobotsTxtFilter will, so a broken store fails
    here rather than silently dropping documents in the pipeline."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    count = conn.execute("SELECT count(*) FROM robots").fetchone()[0]
    assert count == expected_rows, f"wrote {expected_rows} rows but db holds {count}"
    columns = {row[1] for row in conn.execute("PRAGMA table_info(robots)")}
    assert "status_class" in columns, f"status_class column missing: {columns}"
    fqdn = conn.execute("SELECT fqdn FROM robots LIMIT 1").fetchone()[0]
    t = time.perf_counter()
    got = conn.execute("SELECT text, status_class FROM robots WHERE fqdn = ?", (fqdn,)).fetchone()
    lookup_us = (time.perf_counter() - t) * 1e6
    conn.close()
    assert got is not None, f"lookup of {fqdn!r} returned nothing"
    return count, columns, fqdn, lookup_us


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="folder produced by RobotsMerger (holds robotstxt_dict.jsonl)")
    p.add_argument("--output", default=None, help="db path (default: <folder>/robots.db)")
    args = p.parse_args()

    jsonl_path = os.path.join(args.folder, "robotstxt_dict.jsonl")
    if not os.path.isfile(jsonl_path):
        raise SystemExit(f"No robotstxt_dict.jsonl in {args.folder}")
    db_path = args.output or os.path.join(args.folder, "robots.db")

    print(f"{jsonl_path} ({os.path.getsize(jsonl_path) / 1e9:.2f} GB) -> {db_path}")
    n, elapsed = convert(jsonl_path, db_path)
    count, columns, fqdn, lookup_us = verify(db_path, n)
    print(f"{count:,} rows in {elapsed:.1f}s, {os.path.getsize(db_path) / 1e9:.2f} GB")
    print(f"columns: {sorted(columns)}")
    print(f"sample lookup {fqdn!r}: {lookup_us:.1f} us")


if __name__ == "__main__":
    main()
