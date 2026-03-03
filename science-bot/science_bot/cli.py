"""Command-line interface for science-bot."""

import argparse
import asyncio
import csv
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from science_bot.pipeline.orchestrator import (
    OrchestratorRequest,
    OrchestratorResult,
    run_orchestrator,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CSV_PATH = REPO_ROOT / "data" / "BixBenchFiltered_50_clean.csv"
EXTRACTED_CAPSULES_ROOT = REPO_ROOT / "data" / "extracted_capsules"
REQUIRED_BENCHMARK_COLUMNS = frozenset(
    {"question", "data_folder", "capsule_uuid", "question_id", "ideal", "eval_mode"}
)
BENCHMARK_CONCURRENCY = 20
NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+")
RANGE_PATTERN = re.compile(
    r"^\s*[\(\[]\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*[\)\]]\s*$"
)


class BenchmarkRow(BaseModel):
    """Raw benchmark input row."""

    model_config = ConfigDict(extra="forbid")

    question: str
    data_folder: str
    capsule_uuid: str
    question_id: str
    ideal: str
    eval_mode: Literal["str_verifier", "range_verifier", "llm_verifier"]

    @field_validator("question", "data_folder", "capsule_uuid", "question_id", "ideal")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        """Validate that required text fields are non-empty.

        Args:
            value: Candidate field value.

        Returns:
            str: Stripped field value.

        Raises:
            ValueError: If the value is empty after stripping.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("Benchmark fields must be non-empty.")
        return stripped


class BenchmarkRowResult(BaseModel):
    """Result for one benchmark row execution."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str
    capsule_path: Path | None
    eval_mode: Literal["str_verifier", "range_verifier", "llm_verifier"]
    ideal: str
    response: str | None
    is_correct: bool
    status: Literal["completed", "failed"]
    error: str | None = None


class BenchmarkSummary(BaseModel):
    """Aggregate summary for a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    total_rows: int
    completed_rows: int
    failed_rows: int
    correct_rows: int
    incorrect_rows: int
    accuracy: float
    elapsed_seconds: float
    rows: list[BenchmarkRowResult] = Field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser.

    Returns:
        argparse.ArgumentParser: Configured parser for supported arguments.
    """
    parser = argparse.ArgumentParser(prog="science-bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--question", required=True)
    run_parser.add_argument("--capsule", required=True)

    subparsers.add_parser("benchmark")
    return parser


def load_benchmark_rows(csv_path: Path) -> list[BenchmarkRow]:
    """Load benchmark rows from disk.

    Args:
        csv_path: Path to the benchmark CSV file.

    Returns:
        list[BenchmarkRow]: Parsed benchmark rows.

    Raises:
        FileNotFoundError: If the benchmark CSV does not exist.
        ValueError: If the CSV schema or content is invalid.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = REQUIRED_BENCHMARK_COLUMNS - fieldnames
        if missing_columns:
            missing_list = ", ".join(sorted(missing_columns))
            raise ValueError(f"Benchmark CSV missing required columns: {missing_list}")

        rows: list[BenchmarkRow] = []
        for index, raw_row in enumerate(reader, start=2):
            try:
                rows.append(BenchmarkRow.model_validate(raw_row))
            except ValidationError as exc:
                raise ValueError(f"Invalid benchmark row {index}: {exc}") from exc

    return rows


def resolve_benchmark_capsule_path(
    row: BenchmarkRow,
    extracted_capsules_root: Path,
) -> Path:
    """Resolve the data folder for a benchmark row.

    Args:
        row: Benchmark row metadata.
        extracted_capsules_root: Root directory for extracted capsules.

    Returns:
        Path: Resolved capsule data path.

    Raises:
        FileNotFoundError: If the capsule directory cannot be resolved.
        ValueError: If the row's folder format is invalid or ambiguous.
    """
    benchmark_root = extracted_capsules_root / row.capsule_uuid
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"Capsule directory not found: {benchmark_root}")

    folder_name = Path(row.data_folder).name
    if not folder_name.startswith("CapsuleFolder-") or not folder_name.endswith(".zip"):
        raise ValueError(f"Unsupported data_folder value: {row.data_folder}")

    inner_uuid = folder_name.removeprefix("CapsuleFolder-").removesuffix(".zip")
    expected_path = benchmark_root / f"CapsuleData-{inner_uuid}"
    if expected_path.is_dir():
        return expected_path

    matches = sorted(
        path for path in benchmark_root.glob("CapsuleData-*") if path.is_dir()
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No CapsuleData directory found under {benchmark_root}"
        )
    raise ValueError(f"Multiple CapsuleData directories found under {benchmark_root}")


def normalize_text(value: str) -> str:
    """Normalize text for deterministic comparisons.

    Args:
        value: Raw text.

    Returns:
        str: Normalized text.
    """
    return " ".join(value.strip().split()).lower()


def score_benchmark_response(eval_mode: str, ideal: str, response: str) -> bool:
    """Score a benchmark response with a deterministic temporary rule.

    Args:
        eval_mode: Benchmark evaluation mode.
        ideal: Expected answer string.
        response: Orchestrator response text.

    Returns:
        bool: Whether the response is considered correct.

    Raises:
        ValueError: If the evaluation mode or expected range is invalid.
    """
    normalized_ideal = normalize_text(ideal)
    normalized_response = normalize_text(response)

    if eval_mode == "str_verifier":
        return normalized_response == normalized_ideal

    if eval_mode == "llm_verifier":
        return normalized_ideal in normalized_response

    if eval_mode == "range_verifier":
        match = RANGE_PATTERN.fullmatch(ideal)
        if match is None:
            raise ValueError(f"Invalid range_verifier ideal value: {ideal}")
        lower_bound = float(match.group(1))
        upper_bound = float(match.group(2))
        response_match = NUMERIC_PATTERN.search(response)
        if response_match is None:
            return False
        numeric_value = float(response_match.group(0))
        return lower_bound <= numeric_value <= upper_bound

    raise ValueError(f"Unsupported benchmark eval_mode: {eval_mode}")


async def run_benchmark(
    csv_path: Path = BENCHMARK_CSV_PATH,
    extracted_capsules_root: Path = EXTRACTED_CAPSULES_ROOT,
) -> BenchmarkSummary:
    """Run the fixed benchmark suite.

    Args:
        csv_path: Benchmark CSV location.
        extracted_capsules_root: Root directory for extracted capsules.

    Returns:
        BenchmarkSummary: Aggregate benchmark outcomes.

    Raises:
        FileNotFoundError: If the benchmark inputs are missing.
        ValueError: If the benchmark inputs are malformed.
    """
    if not extracted_capsules_root.is_dir():
        raise FileNotFoundError(
            f"Extracted capsules root not found: {extracted_capsules_root}"
        )

    rows = load_benchmark_rows(csv_path)
    semaphore = asyncio.Semaphore(BENCHMARK_CONCURRENCY)
    start_time = time.perf_counter()

    async def run_row(row: BenchmarkRow) -> BenchmarkRowResult:
        """Execute one benchmark row under the shared concurrency limit.

        Args:
            row: Benchmark row to execute.

        Returns:
            BenchmarkRowResult: Completed row result.
        """
        async with semaphore:
            try:
                capsule_path = resolve_benchmark_capsule_path(
                    row, extracted_capsules_root
                )
                orchestrator_result = await run_orchestrator(
                    OrchestratorRequest(
                        question=row.question, capsule_path=capsule_path
                    )
                )
                is_correct = score_benchmark_response(
                    row.eval_mode,
                    row.ideal,
                    orchestrator_result.answer,
                )
                return BenchmarkRowResult(
                    question_id=row.question_id,
                    question=row.question,
                    capsule_path=capsule_path,
                    eval_mode=row.eval_mode,
                    ideal=row.ideal,
                    response=orchestrator_result.answer,
                    is_correct=is_correct,
                    status="completed",
                )
            except Exception as exc:
                return BenchmarkRowResult(
                    question_id=row.question_id,
                    question=row.question,
                    capsule_path=None,
                    eval_mode=row.eval_mode,
                    ideal=row.ideal,
                    response=None,
                    is_correct=False,
                    status="failed",
                    error=str(exc),
                )

    row_results = await asyncio.gather(*(run_row(row) for row in rows))
    elapsed_seconds = time.perf_counter() - start_time
    completed_rows = sum(result.status == "completed" for result in row_results)
    failed_rows = len(row_results) - completed_rows
    correct_rows = sum(result.is_correct for result in row_results)
    incorrect_rows = len(row_results) - correct_rows
    accuracy = correct_rows / len(row_results) if row_results else 0.0
    return BenchmarkSummary(
        total_rows=len(row_results),
        completed_rows=completed_rows,
        failed_rows=failed_rows,
        correct_rows=correct_rows,
        incorrect_rows=incorrect_rows,
        accuracy=accuracy,
        elapsed_seconds=elapsed_seconds,
        rows=row_results,
    )


def format_run_output(result: OrchestratorResult) -> str:
    """Format single-run output for terminal display.

    Args:
        result: Orchestrator result to render.

    Returns:
        str: Human-readable output.
    """
    lines = [
        f"Question: {result.question}",
        f"Capsule: {result.capsule_path}",
        f"Status: {result.status}",
        f"Answer: {result.answer}",
    ]
    if result.error:
        lines.append(f"Error: {result.error}")
    return "\n".join(lines)


def format_benchmark_output(summary: BenchmarkSummary) -> str:
    """Format benchmark output for terminal display.

    Args:
        summary: Benchmark summary to render.

    Returns:
        str: Human-readable output.
    """
    lines = [
        "Benchmark Summary",
        f"Total rows: {summary.total_rows}",
        f"Completed rows: {summary.completed_rows}",
        f"Failed rows: {summary.failed_rows}",
        f"Correct rows: {summary.correct_rows}",
        f"Incorrect rows: {summary.incorrect_rows}",
        f"Accuracy: {summary.accuracy:.2%}",
        f"Elapsed seconds: {summary.elapsed_seconds:.3f}",
        "Rows:",
    ]
    for row in summary.rows:
        outcome = "correct" if row.is_correct else "incorrect"
        detail = row.error if row.error else (row.response or "")
        detail_preview = detail.replace("\n", " ")[:80]
        lines.append(f"- {row.question_id}: {row.status}, {outcome}, {detail_preview}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the top-level science-bot CLI.

    Args:
        argv: Optional command-line arguments.

    Returns:
        int: Process exit code for the CLI.
    """
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            request = OrchestratorRequest(
                question=args.question,
                capsule_path=Path(args.capsule).expanduser().resolve(),
            )
            result = asyncio.run(run_orchestrator(request))
            print(format_run_output(result))
            return 0

        if args.command == "benchmark":
            summary = asyncio.run(run_benchmark())
            print(format_benchmark_output(summary))
            return 0
    except (ValidationError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    parser.error(f"Unsupported command: {args.command}")
    return 1
