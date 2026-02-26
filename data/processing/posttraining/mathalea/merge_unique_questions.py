#!/usr/bin/env python3
"""Merge unique questions from different seeds into one file per exercise."""

import argparse
import hashlib
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = (SCRIPT_DIR / "latex_export_max_questions").resolve()
DEFAULT_OUTPUT_DIR = (SCRIPT_DIR / "latex_export_merged").resolve()
SEEDED_TEX_FILE_PATTERN = re.compile(r"_[a-zA-Z0-9]+_[a-zA-Z0-9]+\.tex$")
MIN_ITEM_CONTENT_LENGTH = 10


@dataclass
class BalancedGroup:
    value: str
    next_index: int


@dataclass
class ExoHeader:
    enonce: str
    code: str
    header_end: int


@dataclass
class ExoSections:
    questions: str
    corrections: str | None


@dataclass
class Question:
    content: str
    hash: str
    enonce: str
    code: str


@dataclass
class MergeResult:
    success: bool
    total_questions: int = 0
    sources_count: int = 0
    reason: str = ""


@dataclass
class UniqueQuestion:
    question: str
    correction: str | None
    enonce: str
    code: str


def hash_question(question_text: str) -> str:
    return hashlib.md5(question_text.encode("utf-8")).hexdigest()


def normalize_for_hash(content: str) -> str:
    """Normalize whitespace so equivalent LaTeX questions hash identically."""
    return re.sub(r"\n+", " ", re.sub(r"\s+", " ", content)).strip()


def read_balanced_group(content: str, start_index: int) -> BalancedGroup | None:
    """Parse a nested {...} block safely (header args may contain braces)."""
    i = start_index
    while i < len(content) and content[i].isspace():
        i += 1
    if i >= len(content) or content[i] != "{":
        return None

    depth = 1
    index = i + 1
    while index < len(content) and depth > 0:
        if content[index] == "{":
            depth += 1
        elif content[index] == "}":
            depth -= 1
        index += 1

    if depth != 0:
        return None
    return BalancedGroup(value=content[i + 1 : index - 1].strip(), next_index=index)


def extract_exo_header(exo_block: str) -> ExoHeader:
    begin_token = "\\begin{EXO}"
    begin_index = exo_block.find(begin_token)
    if begin_index == -1:
        return ExoHeader(enonce="", code="", header_end=-1)

    first_arg = read_balanced_group(exo_block, begin_index + len(begin_token))
    if not first_arg:
        return ExoHeader(enonce="", code="", header_end=-1)

    second_arg = read_balanced_group(exo_block, first_arg.next_index)
    if not second_arg:
        return ExoHeader(enonce=first_arg.value, code="", header_end=first_arg.next_index)

    return ExoHeader(enonce=first_arg.value, code=second_arg.value, header_end=second_arg.next_index)


def extract_exo_sections(tex_content: str) -> ExoSections:
    m = re.search(r"\\begin\{Correction\}", tex_content, flags=re.IGNORECASE)
    if not m:
        return ExoSections(questions=tex_content, corrections=None)
    return ExoSections(questions=tex_content[: m.start()], corrections=tex_content[m.start() :])


def extract_exo_blocks(section_content: str) -> list[str]:
    return [
        m.group(0)
        for m in re.finditer(
            r"\\begin\{EXO\}[\s\S]*?\\end\{EXO\}", section_content, flags=re.IGNORECASE
        )
    ]


def extract_items_from_enumerate(enumerate_content: str) -> list[str]:
    """Extract only top-level \\item blocks from an enumerate body."""
    token_pattern = re.compile(
        r"\\begin\{(?:enumerate|itemize)\}(?:\[[^\]]*\])?|"
        r"\\end\{(?:enumerate|itemize)\}|"
        r"\\item(?:\s*\[[^\]]*\])?",
        flags=re.IGNORECASE,
    )
    top_level_items: list[tuple[int, int]] = []
    nested_depth = 0

    for match in token_pattern.finditer(enumerate_content):
        token = match.group(0).lower()
        if token.startswith(r"\begin{"):
            nested_depth += 1
        elif token.startswith(r"\end{"):
            if nested_depth > 0:
                nested_depth -= 1
        elif nested_depth == 0:
            top_level_items.append((match.start(), match.end()))

    items: list[str] = []
    for i, (_, item_start_content) in enumerate(top_level_items):
        next_item_start = (
            top_level_items[i + 1][0] if i + 1 < len(top_level_items) else len(enumerate_content)
        )
        item_content = enumerate_content[item_start_content:next_item_start].strip()
        if len(item_content) > MIN_ITEM_CONTENT_LENGTH:
            items.append(item_content)
    return items


def extract_first_enumerate_content(exo_block: str) -> str | None:
    """Return body of the first enumerate, handling nested enumerate blocks."""
    begin_match = re.search(r"\\begin\{enumerate\}(?:\[[^\]]*\])?", exo_block, flags=re.IGNORECASE)
    if not begin_match:
        return None

    token_pattern = re.compile(
        r"\\begin\{enumerate\}(?:\[[^\]]*\])?|\\end\{enumerate\}",
        flags=re.IGNORECASE,
    )
    depth = 1
    for token in token_pattern.finditer(exo_block, begin_match.end()):
        value = token.group(0).lower()
        if value.startswith(r"\begin{enumerate}"):
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return exo_block[begin_match.end() : token.start()]
    return None


def extract_enumerate_prefix(exo_block: str) -> str:
    """Return content before first enumerate inside an EXO body."""
    exo_content = extract_exo_content(exo_block)
    if not exo_content:
        return ""
    begin_match = re.search(r"\\begin\{enumerate\}(?:\[[^\]]*\])?", exo_content, flags=re.IGNORECASE)
    if not begin_match:
        return ""
    return exo_content[: begin_match.start()].strip()


def remove_unbalanced_environment_tokens(content: str) -> str:
    """Drop unmatched \\begin/\\end tokens so item fragments remain valid LaTeX."""
    token_pattern = re.compile(
        r"\\begin\{([a-zA-Z*]+)\}(?:\[[^\]]*\])?(?:\{[^{}]*\})*|\\end\{([a-zA-Z*]+)\}",
        flags=re.IGNORECASE,
    )

    stack: list[tuple[str, int, int]] = []
    remove_spans: list[tuple[int, int]] = []

    for m in token_pattern.finditer(content):
        begin_name = m.group(1)
        end_name = m.group(2)
        if begin_name:
            stack.append((begin_name.lower(), m.start(), m.end()))
            continue

        end_name = (end_name or "").lower()
        if stack and stack[-1][0] == end_name:
            stack.pop()
        else:
            remove_spans.append((m.start(), m.end()))

    for _, start, end in stack:
        remove_spans.append((start, end))

    if not remove_spans:
        return content

    out_parts: list[str] = []
    cursor = 0
    for start, end in sorted(remove_spans):
        if start > cursor:
            out_parts.append(content[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(content):
        out_parts.append(content[cursor:])
    return "".join(out_parts)


def clean_item_prefix(prefix: str) -> str:
    """Keep semantic intro text while dropping purely structural wrappers."""
    cleaned = remove_unbalanced_environment_tokens(prefix)
    cleaned = re.sub(r"\\(medskip|smallskip|bigskip|noindent|par)\b", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def is_qcm_format(exo_block: str) -> bool:
    """Enumerate-based exercises are handled item-by-item, not as a single QCM block."""
    if re.search(r"\\begin\{enumerate\}(?:\[[^\]]*\])?", exo_block, flags=re.IGNORECASE):
        return False
    return bool(re.search(r"\\textbf\{[A-Z]\}\.", exo_block))


def extract_exo_content(exo_block: str) -> str:
    header = extract_exo_header(exo_block)
    end_index = exo_block.rfind("\\end{EXO}")
    if header.header_end == -1 or end_index == -1 or end_index <= header.header_end:
        return ""
    return exo_block[header.header_end : end_index].strip()


def extract_questions_with_context(section_content: str) -> list[Question]:
    questions: list[Question] = []

    for exo_block in extract_exo_blocks(section_content):
        header = extract_exo_header(exo_block)

        if is_qcm_format(exo_block):
            content = extract_exo_content(exo_block)
            if len(content) > MIN_ITEM_CONTENT_LENGTH:
                questions.append(
                    Question(
                        content=content,
                        hash=hash_question(normalize_for_hash(content)),
                        enonce=header.enonce,
                        code=header.code,
                    )
                )
        else:
            enumerate_content = extract_first_enumerate_content(exo_block)
            raw_prefix = extract_enumerate_prefix(exo_block)
            prefix = clean_item_prefix(raw_prefix)
            if enumerate_content is not None:
                for item_content in extract_items_from_enumerate(enumerate_content):
                    contextual_item = item_content if not prefix else f"{prefix}\n\n{item_content}"
                    questions.append(
                        Question(
                            content=contextual_item,
                            hash=hash_question(normalize_for_hash(contextual_item)),
                            enonce=header.enonce,
                            code=header.code,
                        )
                    )

    return questions


def merge_exercise(exercise_dir: Path, exercise_name: str, output_dir: Path) -> MergeResult:
    files = [f.name for f in exercise_dir.iterdir() if f.is_file()]
    tex_files = sorted(f for f in files if f.endswith(".tex") and SEEDED_TEX_FILE_PATTERN.search(f))

    if not tex_files:
        return MergeResult(success=False, reason="no_seed_files")

    unique_questions: dict[str, UniqueQuestion] = {}
    seen_hashes: set[str] = set()

    for file in tex_files:
        content = (exercise_dir / file).read_text(encoding="utf-8")
        sections = extract_exo_sections(content)

        questions = extract_questions_with_context(sections.questions)
        corrections = extract_questions_with_context(sections.corrections) if sections.corrections else []
        paired_corrections = corrections
        if corrections and len(corrections) != len(questions):
            # Avoid silently assigning wrong corrections when extraction counts diverge.
            log.warning(
                "[WARN] %s/%s: question/correction count mismatch (%d vs %d), skipping corrections for this source",
                exercise_name,
                file,
                len(questions),
                len(corrections),
            )
            paired_corrections = []

        for i, question in enumerate(questions):
            # Keep positional pairing with extracted corrections from the same source file.
            correction = paired_corrections[i] if i < len(paired_corrections) else None

            if question.hash not in seen_hashes:
                # First occurrence wins for deterministic merged output.
                seen_hashes.add(question.hash)
                unique_questions[question.hash] = UniqueQuestion(
                    question=question.content,
                    correction=correction.content if correction else None,
                    enonce=question.enonce,
                    code=question.code,
                )

    if not unique_questions:
        return MergeResult(success=False, reason="no_questions")

    merged_content = ""
    for v in unique_questions.values():
        merged_content += "\\begin{EXO}{" + v.enonce + "}{" + (v.code or exercise_name) + "}\n"
        merged_content += v.question + "\n"
        merged_content += "\\end{EXO}\n\n"

    merged_content += "\\begin{Correction}\n"
    for v in unique_questions.values():
        if v.correction:
            merged_content += "\\begin{EXO}{}{}\n"
            merged_content += v.correction + "\n"
            merged_content += "\\end{EXO}\n\n"
    merged_content += "\\end{Correction}\n"

    output_path = output_dir / exercise_name
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / f"{exercise_name}_merged.tex").write_text(merged_content, encoding="utf-8")

    return MergeResult(success=True, total_questions=len(unique_questions), sources_count=len(tex_files))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge unique questions from seeded LaTeX exports into a single file per exercise."
    )
    parser.add_argument(
        "--source", dest="source_dir", default=str(DEFAULT_SOURCE_DIR),
        help="Source directory containing exercise folders (output of scrape_latex_max_questions.py)",
    )
    parser.add_argument("--output", dest="output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--exercise", default=None, help="Process only one exercise folder")
    return parser


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    ns = parse_args(argv)
    source_dir = Path(ns.source_dir).resolve()
    output_dir = Path(ns.output_dir).resolve()

    log.info("Merging unique questions from seeded exports...\n")
    log.info("Source: %s", source_dir)
    log.info("Output: %s", output_dir)
    if ns.exercise:
        log.info("Filter: %s", ns.exercise)
    log.info("")

    output_dir.mkdir(parents=True, exist_ok=True)

    exercise_dirs = sorted(
        (e for e in source_dir.iterdir() if e.is_dir()),
        key=lambda p: p.name,
    )
    if ns.exercise:
        exercise_dirs = [e for e in exercise_dirs if e.name == ns.exercise]

    total = 0
    success = 0
    failed = 0
    total_questions = 0
    reasons: dict[str, int] = {}

    for dir_entry in exercise_dirs:
        total += 1
        try:
            result = merge_exercise(dir_entry, dir_entry.name, output_dir)

            if result.success:
                success += 1
                total_questions += result.total_questions
                log.info(
                    "[OK] %s: %d unique questions (%d sources)",
                    dir_entry.name, result.total_questions, result.sources_count,
                )
            else:
                failed += 1
                reasons[result.reason] = reasons.get(result.reason, 0) + 1
        except Exception:
            failed += 1
            reasons["error"] = reasons.get("error", 0) + 1
            log.exception("[ERR] %s", dir_entry.name)

    log.info("\n========== SUMMARY ==========")
    log.info("Exercises processed          : %d", total)
    log.info("Success                      : %d", success)
    log.info("Failed                       : %d", failed)
    log.info("Total unique questions       : %d", total_questions)

    if reasons:
        log.info("\nFailure reasons:")
        labels = {
            "no_seed_files": "No seeded files",
            "no_questions": "No question extracted",
            "error": "Runtime error",
        }
        for reason, count in reasons.items():
            log.info("  %s: %d", labels.get(reason, reason), count)

    if ns.exercise and total == 0:
        log.info("\nNo exercise matched the provided --exercise value.")

    log.info("\nMerged files written to: %s", output_dir)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception:
        log.exception("[FATAL]")
        sys.exit(1)
