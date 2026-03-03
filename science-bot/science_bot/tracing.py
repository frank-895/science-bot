"""Structured trace writing utilities for temporary pipeline debugging."""

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field


class TraceEvent(BaseModel):
    """One structured event written to a trace stream."""

    model_config = ConfigDict(extra="forbid")

    time: str
    event: str
    stage: str
    question_id: str | None = None
    question: str | None = None
    family: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)


class RunTraceSummary(BaseModel):
    """Terminal summary for one traced run."""

    model_config = ConfigDict(extra="forbid")

    status: str
    question: str
    capsule_path: str
    classification_family: str | None = None
    resolution_iterations_used: int | None = None
    selected_files: list[str] = Field(default_factory=list)
    answer: str | None = None
    execution_family: str | None = None
    error: str | None = None


class BenchmarkTraceManifest(BaseModel):
    """Root metadata for one traced benchmark run."""

    model_config = ConfigDict(extra="forbid")

    command: str
    csv_path: str
    benchmark_directory: str
    prepared_capsule_root: str
    trace_root: str
    started_at: str
    concurrency: int


class BenchmarkRowTraceSummary(BaseModel):
    """Terminal summary for one traced benchmark row."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    status: str
    classification_family: str | None = None
    resolution_iterations_used: int | None = None
    selected_files: list[str] = Field(default_factory=list)
    answer: str | None = None
    is_correct: bool | None = None
    error: str | None = None


class BenchmarkTraceSummary(BaseModel):
    """Aggregate summary for one traced benchmark run."""

    model_config = ConfigDict(extra="forbid")

    total_rows: int
    completed_rows: int
    failed_rows: int
    correct_rows: int
    incorrect_rows: int
    accuracy: float
    elapsed_seconds: float
    rows: list[dict[str, str]] = Field(default_factory=list)


def _utc_timestamp() -> str:
    """Return a UTC timestamp safe for filenames."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _jsonable(value: object) -> object:
    """Convert a Python object into JSON-serializable data."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


class TraceWriter:
    """Best-effort structured trace writer rooted at one filesystem directory."""

    def __init__(self, root_dir: Path) -> None:
        """Initialize a trace writer.

        Args:
            root_dir: Trace directory for one run or benchmark row.
        """
        self.root_dir = root_dir
        self.disabled_due_to_error = False
        self._child_counts: dict[str, int] = {}
        self._ensure_directory()

    @classmethod
    def for_run(cls, base_dir: Path) -> "TraceWriter":
        """Create a trace writer for one CLI run command."""

        return cls(base_dir / f"run_{_utc_timestamp()}")

    @classmethod
    def for_benchmark(cls, base_dir: Path) -> "TraceWriter":
        """Create a trace writer for one CLI benchmark command."""

        return cls(base_dir / f"benchmark_{_utc_timestamp()}")

    def create_row_writer(self, question_id: str) -> "TraceWriter":
        """Create a child trace writer for one benchmark row."""

        count = self._child_counts.get(question_id, 0) + 1
        self._child_counts[question_id] = count
        suffix = "" if count == 1 else f"_{count}"
        return TraceWriter(self.root_dir / f"{question_id}{suffix}")

    def write_event(
        self,
        *,
        event: str,
        stage: str,
        payload: object | None = None,
        question_id: str | None = None,
        question: str | None = None,
        family: str | None = None,
    ) -> None:
        """Append one event to `events.jsonl`.

        Args:
            event: Stable event name.
            stage: Logical pipeline or CLI stage.
            payload: JSON-serializable event payload.
            question_id: Optional benchmark question identifier.
            question: Optional natural-language question.
            family: Optional question family.
        """
        payload_data = _jsonable(payload) if payload is not None else {}
        if not isinstance(payload_data, dict):
            payload_data = {"value": payload_data}
        trace_event = TraceEvent(
            time=datetime.now(timezone.utc).isoformat(),
            event=event,
            stage=stage,
            question_id=question_id,
            question=question,
            family=family,
            payload=cast(dict[str, object], payload_data),
        )
        self._append_jsonl("events.jsonl", trace_event.model_dump(mode="python"))

    def write_summary(self, data: object, filename: str = "summary.json") -> None:
        """Write one JSON summary artifact."""

        self._write_json(filename, _jsonable(data))

    def write_manifest(self, data: object) -> None:
        """Write the benchmark manifest artifact."""

        self.write_summary(data, filename="manifest.json")

    def write_error(self, exc: Exception, *, stage: str | None = None) -> None:
        """Write a plain-text error artifact for quick inspection."""

        if self.disabled_due_to_error:
            return
        try:
            lines = [str(exc)]
            if stage is not None:
                lines.insert(0, f"stage={stage}")
            lines.append("")
            lines.append("traceback:")
            lines.append("".join(traceback.format_exception(exc)))
            (self.root_dir / "error.txt").write_text(
                "\n".join(lines),
                encoding="utf-8",
            )
        except OSError:
            self.disabled_due_to_error = True

    def _ensure_directory(self) -> None:
        """Create the root trace directory if possible."""

        if self.disabled_due_to_error:
            return
        try:
            self.root_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.disabled_due_to_error = True

    def _write_json(self, filename: str, data: object) -> None:
        """Write one JSON file best-effort."""

        if self.disabled_due_to_error:
            return
        try:
            self._ensure_directory()
            with (self.root_dir / filename).open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
        except OSError:
            self.disabled_due_to_error = True

    def _append_jsonl(self, filename: str, data: object) -> None:
        """Append one JSONL record best-effort."""

        if self.disabled_due_to_error:
            return
        try:
            self._ensure_directory()
            with (self.root_dir / filename).open("a", encoding="utf-8") as handle:
                json.dump(data, handle, sort_keys=True)
                handle.write("\n")
        except OSError:
            self.disabled_due_to_error = True
