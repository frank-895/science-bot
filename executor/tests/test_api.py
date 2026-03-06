import asyncio
import subprocess
from pathlib import Path

import pytest

from executor import api


def test_packages_available_contains_scikit_learn() -> None:
    assert "scikit-learn" in api.packages_available()


def test_run_python_rejects_empty_script() -> None:
    with pytest.raises(ValueError, match="script must be non-empty"):
        asyncio.run(api.run_python("  "))


def test_ensure_ready_raises_when_docker_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        command: list[str],
        raise_on_error: bool,
    ) -> subprocess.CompletedProcess[str]:
        del raise_on_error
        if command[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(
                command,
                returncode=1,
                stdout="",
                stderr="",
            )
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout="[]",
            stderr="",
        )

    monkeypatch.setattr(api, "_run_sync_command", fake_run)

    with pytest.raises(RuntimeError, match="Docker is unavailable"):
        api.ensure_ready()


def test_run_python_returns_invalid_output_when_runner_stdout_is_not_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(api, "_RUNS_ROOT", tmp_path / "runs")

    async def fake_to_thread(function, *args):
        del function, args
        return 1

    class FakeProcess:
        async def communicate(self) -> tuple[bytes, bytes]:
            return b"not-json", b""

        def kill(self) -> None:
            return None

    async def fake_create_subprocess_exec(*args, **kwargs):
        del args, kwargs
        return FakeProcess()

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    result = asyncio.run(api.run_python("print('x')"))

    assert result["status"] == "invalid_output"
    assert result["error_type"] == "invalid_output"
