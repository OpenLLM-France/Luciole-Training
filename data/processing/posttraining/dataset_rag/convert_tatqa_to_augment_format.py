#!/usr/bin/env python3
"""
Convert TATQA data into augmentation format compatible with the existing pipeline.
"""

import argparse
import json
import re
from pathlib import Path


def _stringify_cell(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    return text.replace("|", "\\|").strip()


def table_to_markdown(table_rows: list[list[object]]) -> str:
    if not table_rows:
        return ""
    headers = [_stringify_cell(cell) for cell in table_rows[0]]
    if not headers:
        return ""
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for row in table_rows[1:]:
        cells = [_stringify_cell(c) for c in row]
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        elif len(cells) > len(headers):
            cells = cells[:len(headers)]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header_line, sep_line] + body)


def _extract_docs_from_broken_array(text: str) -> list[dict]:
    docs = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        m = re.search(r"\{", text[idx:])
        if not m:
            break
        start = idx + m.start()
        try:
            obj, end = decoder.raw_decode(text[start:])
            if isinstance(obj, dict) and "table" in obj and "questions" in obj:
                docs.append(obj)
            idx = start + end
        except json.JSONDecodeError:
            idx = start + 1
    return docs


def load_tatqa_docs(input_path: Path) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    text = input_path.read_text(encoding="utf-8")

    # 1) Try full JSON first.
    try:
        data = json.loads(text)
        if isinstance(data, list):
            docs = [d for d in data if isinstance(d, dict)]
            return docs, warnings
        if isinstance(data, dict):
            return [data], warnings
    except json.JSONDecodeError as e:
        warnings.append(f"Full JSON parse failed: {e}")

    # 2) Try JSONL line by line.
    docs = []
    jsonl_ok = 0
    jsonl_bad = 0
    for i, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                docs.append(obj)
                jsonl_ok += 1
        except json.JSONDecodeError:
            jsonl_bad += 1
    if jsonl_ok > 0 and jsonl_bad == 0:
        return docs, warnings
    if jsonl_ok > 0:
        warnings.append(f"JSONL parse partial: {jsonl_ok} valid lines, {jsonl_bad} invalid lines.")

    # 3) Fallback for truncated pretty-printed array.
    recovered = _extract_docs_from_broken_array(text)
    if recovered:
        warnings.append(
            f"Recovered {len(recovered)} docs from partially malformed/truncated JSON array."
        )
        return recovered, warnings

    raise ValueError(f"Could not parse input file: {input_path}")


def _format_number(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


def normalize_answer(question: dict) -> tuple[str, object, list[str] | None]:
    answer_type = question.get("answer_type", "span")
    answer_raw = question.get("answer")
    answer_spans: list[str] | None = None

    if answer_type == "multi-span":
        if isinstance(answer_raw, list):
            answer_spans = [str(x) for x in answer_raw]
            answer = ", ".join(answer_spans)
        else:
            answer_spans = [str(answer_raw)]
            answer = str(answer_raw)
        return answer, answer_raw, answer_spans

    if answer_type in {"arithmetic", "count"}:
        if isinstance(answer_raw, list) and answer_raw:
            first = answer_raw[0]
            answer = _format_number(first)
        else:
            answer = _format_number(answer_raw)
        return answer, answer_raw, answer_spans

    # Default span
    if isinstance(answer_raw, list):
        answer = str(answer_raw[0]) if answer_raw else ""
    else:
        answer = str(answer_raw)
    return answer, answer_raw, answer_spans


def convert_doc(doc: dict) -> list[dict]:
    out = []
    table_rows = doc.get("table", {}).get("table", [])
    table_uid = doc.get("table", {}).get("uid", "table")
    table_md = table_to_markdown(table_rows)

    paragraphs = sorted(doc.get("paragraphs", []), key=lambda p: p.get("order", 0))
    para_by_order = {p.get("order"): p for p in paragraphs if "order" in p}

    context_titles = ["Table"]
    context_sentences = [[table_md]]
    for p in paragraphs:
        title = f"Paragraph {p.get('order', '?')}"
        context_titles.append(title)
        context_sentences.append([str(p.get("text", ""))])

    for q in doc.get("questions", []):
        q_uid = q.get("uid", "unknown")
        row_id = f"{table_uid}_{q_uid}"
        answer_from = q.get("answer_from", "")
        rel_paragraphs = q.get("rel_paragraphs", []) or []

        sf_titles: list[str] = []
        if answer_from in {"table", "table-text"}:
            sf_titles.append("Table")
        if answer_from in {"text", "table-text"}:
            for order in rel_paragraphs:
                sf_titles.append(f"Paragraph {order}")
        # Deduplicate preserving order
        sf_titles = list(dict.fromkeys(sf_titles))

        answer_norm, answer_raw, answer_spans = normalize_answer(q)
        row = {
            "id": row_id,
            "question": q.get("question", ""),
            "answer": answer_norm,
            "answer_raw": answer_raw,
            "answer_type": q.get("answer_type", ""),
            "answer_from": answer_from,
            "derivation": q.get("derivation", ""),
            "scale": q.get("scale", ""),
            "req_comparison": bool(q.get("req_comparison", False)),
            "context": {"title": context_titles, "sentences": context_sentences},
            "supporting_facts": {"title": sf_titles, "sent_id": [0] * len(sf_titles)},
            # Compatibility with existing reporting scripts
            "type": q.get("answer_type", "unknown"),
            "level": "tatqa",
        }
        if answer_spans is not None:
            row["answer_spans"] = answer_spans
        out.append(row)
    return out


def convert(input_file: str, output_file: str) -> tuple[int, int, int]:
    docs, warnings = load_tatqa_docs(Path(input_file))
    rows_out = []
    for doc in docs:
        rows_out.extend(convert_doc(doc))

    with open(output_file, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for w in warnings:
        print(f"[warn] {w}")
    print(f"Docs parsed: {len(docs)}")
    print(f"Questions converted: {len(rows_out)}")
    return len(docs), len(rows_out), len(warnings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TATQA to augmentation format JSONL")
    parser.add_argument("--input", required=True, help="Path to TATQA input (JSON/JSONL)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
