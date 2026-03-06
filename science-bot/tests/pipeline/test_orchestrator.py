import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from science_bot.pipeline.orchestrator import (
    OrchestratorRequest,
    run_orchestrator,
)
from science_bot.tracing import TraceWriter


def test_orchestrator_request_rejects_empty_question(tmp_path: Path) -> None:
    capsule_path = tmp_path / "capsule"
    capsule_path.mkdir()

    with pytest.raises(ValidationError, match="question must be non-empty"):
        OrchestratorRequest(question="   ", capsule_path=capsule_path)


def test_orchestrator_raises_for_missing_capsule(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-capsule"

    with pytest.raises(FileNotFoundError, match="Capsule path not found"):
        asyncio.run(
            run_orchestrator(
                OrchestratorRequest(question="What is this?", capsule_path=missing_path)
            )
        )


def test_orchestrator_returns_stub_and_writes_trace(tmp_path: Path) -> None:
    capsule_path = tmp_path / "capsule"
    capsule_path.mkdir()
    trace_writer = TraceWriter.for_run(tmp_path / "traces")

    result = asyncio.run(
        run_orchestrator(
            OrchestratorRequest(
                question="How many rows?",
                capsule_path=capsule_path,
                trace_writer=trace_writer,
            )
        )
    )

    assert result.question == "How many rows?"
    assert result.capsule_path == capsule_path
    assert result.status == "completed"
    assert result.answer == "ORCHESTRATOR_STUB_RESPONSE"
    assert result.metadata["classification_family"] == "stub"
    assert result.metadata["resolution_iterations_used"] == 0
    assert result.metadata["resolution_selected_files"] == []
    assert result.metadata["execution_family"] == "stub"

    events = [
        json.loads(line)
        for line in (trace_writer.root_dir / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["event"] for event in events] == ["orchestrator_stub_used"]
