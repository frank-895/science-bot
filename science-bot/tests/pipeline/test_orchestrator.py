import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError
from science_bot.pipeline.orchestrator import (
    STUB_ANSWER,
    OrchestratorRequest,
    run_orchestrator,
)


def test_orchestrator_returns_stub_result(tmp_path: Path) -> None:
    capsule_path = tmp_path / "capsule"
    capsule_path.mkdir()

    result = asyncio.run(
        run_orchestrator(
            OrchestratorRequest(question="What is this?", capsule_path=capsule_path)
        )
    )

    assert result.question == "What is this?"
    assert result.capsule_path == capsule_path
    assert result.status == "completed"
    assert result.answer == STUB_ANSWER
    assert result.metadata["orchestrator_mode"] == "stub"


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
