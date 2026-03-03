import json
from pathlib import Path

from science_bot.tracing import TraceWriter


def test_trace_writer_creates_run_and_row_directories(tmp_path: Path) -> None:
    base_dir = tmp_path / "traces"

    benchmark_writer = TraceWriter.for_benchmark(base_dir)
    first_row = benchmark_writer.create_row_writer("bix-10-q1")
    second_row = benchmark_writer.create_row_writer("bix-10-q1")

    assert benchmark_writer.root_dir.is_dir()
    assert first_row.root_dir.name == "bix-10-q1"
    assert second_row.root_dir.name == "bix-10-q1_2"


def test_trace_writer_writes_event_summary_and_error(tmp_path: Path) -> None:
    writer = TraceWriter.for_run(tmp_path / "traces")

    writer.write_event(
        event="run_started",
        stage="cli",
        question="What?",
        payload={"capsule_path": "/tmp/capsule"},
    )
    writer.write_summary({"status": "completed"})
    writer.write_error(ValueError("boom"))

    events_path = writer.root_dir / "events.jsonl"
    summary_path = writer.root_dir / "summary.json"
    error_path = writer.root_dir / "error.txt"

    assert events_path.is_file()
    assert summary_path.is_file()
    assert error_path.is_file()

    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["event"] == "run_started"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["status"] == "completed"
    assert "boom" in error_path.read_text(encoding="utf-8")
