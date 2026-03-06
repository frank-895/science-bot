import asyncio
import types

import pytest
from science_bot.providers import executor


def test_ensure_python_executor_ready_maps_missing_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str) -> object:
        del name
        raise ModuleNotFoundError("missing")

    monkeypatch.setattr(executor.importlib, "import_module", fake_import_module)

    with pytest.raises(
        executor.PythonExecutorUnavailableError,
        match="uv pip install -e ./executor",
    ):
        executor.ensure_python_executor_ready()


def test_ensure_python_executor_ready_delegates_to_executor_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"ready": False}

    def ensure_ready() -> None:
        called["ready"] = True

    module = types.SimpleNamespace(ensure_ready=ensure_ready)
    monkeypatch.setattr(executor.importlib, "import_module", lambda name: module)

    executor.ensure_python_executor_ready()

    assert called["ready"] is True


def test_list_available_python_packages_delegates_to_executor_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.SimpleNamespace(packages_available=lambda: ["numpy", "pandas"])
    monkeypatch.setattr(executor.importlib, "import_module", lambda name: module)

    packages = executor.list_available_python_packages()

    assert packages == ["numpy", "pandas"]


def test_run_python_returns_validated_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_python(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        return {
            "status": "completed",
            "answer": "42",
            "error_type": None,
            "error_message": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "duration_ms": 10,
            "worker": "runner[1]",
        }

    module = types.SimpleNamespace(run_python=run_python)
    monkeypatch.setattr(executor.importlib, "import_module", lambda name: module)

    result = asyncio.run(executor.run_python("print(42)"))

    assert result.status == "completed"
    assert result.answer == "42"


def test_run_python_maps_executor_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_python(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise RuntimeError("boom")

    module = types.SimpleNamespace(run_python=run_python)
    monkeypatch.setattr(executor.importlib, "import_module", lambda name: module)

    with pytest.raises(executor.PythonExecutorUnavailableError, match="boom"):
        asyncio.run(executor.run_python("print(1)"))
