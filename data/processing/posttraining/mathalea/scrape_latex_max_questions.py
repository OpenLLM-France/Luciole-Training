#!/usr/bin/env python3
"""Scrape MathALÉA LaTeX exports by exercise and seed to generate as many questions as possible, mainly by varying seeds and requesting up to 1,000 questions per run.
We recommend hosting https://forge.apps.education.fr/coopmaths/mathalea locally."""

import argparse
import asyncio
import io
import json
import logging
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REF_PATH = (SCRIPT_DIR / "../src/json/refToUuidFR.json").resolve()
DEFAULT_ID_LIST_PATH = (SCRIPT_DIR / "id").resolve()
DEFAULT_OUTPUT_DIR = (SCRIPT_DIR / "latex_export_max_questions").resolve()
DEFAULT_QUESTION_COUNTS: list[int | None] = [1000, None]
DEFAULT_SEEDS = ["e906e", "a1b2c", "xYz89", "pQ4r5", "mN7s8"]
DEFAULT_HOST = "http://localhost:5173/alea/"


@dataclass
class DownloadResult:
    success: bool
    question_count: int | None = None
    seed: str | None = None


def parse_question_counts(value: str) -> list[int | None]:
    """Parse a comma-separated list of question counts ('default' maps to None)."""
    out: list[int | None] = []
    for raw in value.split(","):
        v = raw.strip()
        if not v:
            continue
        if v.lower() == "default":
            out.append(None)
            continue
        try:
            parsed = int(v)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid --counts value: {v}") from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError(f"Invalid --counts value: {v}")
        out.append(parsed)
    return out


def parse_seeds(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def remove_ext(filename: str) -> str:
    return re.sub(r"\.[^/.]+$", "", filename)


def build_latex_url(host: str, uuid: str, alea: str, question_count: int | None) -> str:
    params = [("uuid", uuid), ("d", "10"), ("alea", alea), ("v", "latex"), ("testCI", "")]
    if question_count is not None:
        params.append(("n", str(question_count)))
    return f"{host}?{urlencode(params)}"


def read_zip_buffer(buffer: bytes) -> dict[str, str]:
    """Extract text files from a zip buffer, flattening 'images/' prefix."""
    files: dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(buffer), "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("images/", "", 1)
            files[name] = zf.read(info.filename).decode("utf-8", errors="replace")
    return files


async def download_latex_for_exercise(
    page,
    uuid: str,
    id_path: str,
    output_dir: Path,
    host: str,
    question_count: int | None,
    seed: str,
    timeout_ms: int,
) -> DownloadResult:
    url = build_latex_url(host, uuid, seed, question_count)
    log.info("→ Opening: %s", url)

    try:
        await page.goto(url, timeout=timeout_ms)
        await page.wait_for_selector("button#downloadFullArchive", timeout=timeout_ms)
    except (TimeoutError, Exception):
        log.warning("   Page not loaded or download button not found")
        return DownloadResult(success=False)

    try:
        await page.click("input#Style0")
        await page.wait_for_selector(
            "button#downloadFullArchive:not([disabled])", timeout=timeout_ms, state="visible"
        )
        await asyncio.sleep(0.5)

        async with page.expect_download(timeout=timeout_ms) as download_info:
            await page.click("button#downloadFullArchive", timeout=timeout_ms)
        download = await download_info.value

        failure = await download.failure()
        if failure:
            log.warning("   Download error")
            return DownloadResult(success=False)

        dl_path = await download.path()
        if dl_path is None:
            log.warning("   Download path unavailable")
            return DownloadResult(success=False)

        data = Path(dl_path).read_bytes()
        files = read_zip_buffer(data)

        # Accept both archive layouts emitted by MathALEA.
        tex_name = next((name for name in ("main.tex", "test.tex") if name in files), None)
        if tex_name is None:
            log.warning("   No main.tex/test.tex found in zip")
            return DownloadResult(success=False)

        parts = remove_ext(id_path).split("/")
        out_dir = output_dir / (parts[0] or "misc")
        file_id = parts[-1]
        q_label = "default" if question_count is None else f"n{question_count}"

        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / f"{file_id}_{seed}_{q_label}.tex"
        output_file.write_text(files[tex_name], encoding="utf-8")
        log.info("   Saved: %s", output_file.name)

        return DownloadResult(success=True, question_count=question_count, seed=seed)
    except Exception:
        log.exception("   Error during download")
        return DownloadResult(success=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download LaTeX exports from MathALÉA for each exercise/seed combination."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Base URL of the MathALÉA server")
    parser.add_argument(
        "--ref", dest="ref_path", default=str(DEFAULT_REF_PATH),
        help="Path to refToUuidFR.json (generated by `pnpm makeJson`)",
    )
    parser.add_argument(
        "--ids", dest="ids_path", default=str(DEFAULT_ID_LIST_PATH),
        help="Path to file containing exercise IDs (one per line)",
    )
    parser.add_argument("--output", dest="output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--exercise", default=None, help="Run only one exercise ID")
    parser.add_argument(
        "--counts", default=None,
        help='Comma-separated question counts, use "default" for null (ex: 1000,default)',
    )
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds (ex: e906e,a1b2c)")
    parser.add_argument("--timeout", dest="timeout_ms", type=int, default=60000, help="Playwright timeout in ms")
    parser.add_argument("--headful", action="store_true", help="Run browser with UI (default: headless)")
    return parser


@dataclass
class Options:
    host: str
    ref_path: Path
    ids_path: Path
    output_dir: Path
    exercise: str | None
    question_counts: list[int | None]
    seeds: list[str]
    timeout_ms: int
    headless: bool


def parse_args(argv: list[str]) -> Options:
    parser = build_parser()
    ns = parser.parse_args(argv)

    if ns.timeout_ms <= 0:
        parser.error("--timeout must be positive")

    question_counts = DEFAULT_QUESTION_COUNTS if ns.counts is None else parse_question_counts(ns.counts)
    seeds = DEFAULT_SEEDS if ns.seeds is None else parse_seeds(ns.seeds)

    if not seeds:
        parser.error("At least one seed is required")
    if not question_counts:
        parser.error("At least one question count is required")

    return Options(
        host=ns.host,
        ref_path=Path(ns.ref_path).resolve(),
        ids_path=Path(ns.ids_path).resolve(),
        output_dir=Path(ns.output_dir).resolve(),
        exercise=ns.exercise,
        question_counts=question_counts,
        seeds=seeds,
        timeout_ms=ns.timeout_ms,
        headless=not ns.headful,
    )


async def main(argv: list[str]) -> None:
    options = parse_args(argv)
    options.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Host:   %s", options.host)
    log.info("Output: %s", options.output_dir)
    log.info("Seeds (%d): %s", len(options.seeds), ", ".join(options.seeds))
    log.info("Counts: %s", ", ".join("default" if c is None else str(c) for c in options.question_counts))
    if options.exercise:
        log.info("Filter: %s", options.exercise)
    log.info("")

    exercise_ids = [
        line.strip()
        for line in options.ids_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if options.exercise:
        exercise_ids = [id_ for id_ in exercise_ids if id_ == options.exercise]
    log.info("Exercises in list: %d", len(exercise_ids))

    ref_to_uuid: dict[str, str] = json.loads(options.ref_path.read_text(encoding="utf-8"))

    exercises_to_test: list[dict[str, str]] = []
    not_found_ids: list[str] = []
    entries = list(ref_to_uuid.items())
    for id_ in exercise_ids:
        found = None
        for p, uuid in entries:
            if remove_ext(p) == id_ or p == id_:
                found = (p, uuid)
                break
        if found:
            exercises_to_test.append({"id_path": found[0], "uuid": found[1], "original_id": id_})
        else:
            not_found_ids.append(id_)

    log.info("Found in refToUuidFR.json: %d", len(exercises_to_test))
    if not_found_ids:
        suffix = "..." if len(not_found_ids) > 10 else ""
        preview = ", ".join(not_found_ids[:10])
        log.info("Not found: %d — %s%s", len(not_found_ids), preview, suffix)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=options.headless)
        context = await browser.new_context()
        page = await context.new_page()

        total = len(exercises_to_test)
        total_files = 0
        success_per_exercise = 0
        failed_completely = 0
        files_per_exercise: dict[str, list[dict]] = {}

        for exo in exercises_to_test:
            id_path = exo["id_path"]
            uuid = exo["uuid"]
            original_id = exo["original_id"]
            try:
                log.info("\nExercise uuid=%s id=%s", uuid, id_path)
                exercise_has_success = False
                files_generated: list[dict] = []

                for seed in options.seeds:
                    result = DownloadResult(success=False)

                    for question_count in options.question_counts:
                        label = "n=default" if question_count is None else f"n={question_count}"
                        log.info("  [%s] %s", seed, label)
                        result = await download_latex_for_exercise(
                            page, uuid, id_path, options.output_dir,
                            options.host, question_count, seed, options.timeout_ms,
                        )
                        if result.success:
                            break

                        # Recreate a fresh page between retries.
                        try:
                            await page.close()
                        except Exception:
                            pass
                        page = await context.new_page()

                    if result.success:
                        exercise_has_success = True
                        files_generated.append({"seed": result.seed, "question_count": result.question_count})
                        total_files += 1
                    else:
                        log.info("  ✗ Failed: seed=%s", seed)

                if exercise_has_success:
                    success_per_exercise += 1
                    files_per_exercise[original_id] = files_generated
                    log.info("  ✓ %d/%d seeds succeeded", len(files_generated), len(options.seeds))
                else:
                    failed_completely += 1
                    log.info("  Total failure for this exercise")
            except Exception:
                log.exception("  Error on %s / %s", uuid, id_path)
                failed_completely += 1
                try:
                    await page.close()
                except Exception:
                    pass
                page = await context.new_page()

        try:
            await page.close()
        except Exception:
            pass
        await browser.close()

    log.info("\n========== SUMMARY ==========")
    log.info("Processed                  : %d", total)
    log.info("With at least 1 file       : %d", success_per_exercise)
    log.info("Complete failures          : %d", failed_completely)
    log.info("Total .tex files generated : %d", total_files)

    # Keep the report format stable for downstream tooling.
    report_lines = [
        "MathALÉA Scraping Report",
        "========================\n",
        f"Date    : {datetime.now(timezone.utc).isoformat()}",
        f"Host    : {options.host}",
        f"Seeds   : {', '.join(options.seeds)}",
        "Counts  : " + ", ".join("default" if c is None else str(c) for c in options.question_counts),
        f"Processed: {total}  (success: {success_per_exercise}, failures: {failed_completely})",
        f"Files generated: {total_files}",
        "\n--- Detail per exercise ---",
    ]

    for ex_id, files in files_per_exercise.items():
        detail = ", ".join(
            f"seed={f['seed']} n={f['question_count'] if f['question_count'] is not None else 'default'}"
            for f in files
        )
        report_lines.append(f"{ex_id}: {detail}")

    if failed_completely > 0:
        report_lines.append(f"\nComplete failures ({failed_completely}):")
        report_lines.append("-" * 40)
        for ex in exercises_to_test:
            if ex["original_id"] not in files_per_exercise:
                report_lines.append(ex["original_id"])

    report_path = options.output_dir / "report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("\nReport saved: %s", report_path)


if __name__ == "__main__":
    try:
        asyncio.run(main(sys.argv[1:]))
    except Exception:
        log.exception("[FATAL]")
        sys.exit(1)
