import pytest


@pytest.fixture(autouse=True)
def mock_executor_preflight_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CLI tests deterministic without requiring Docker."""
    monkeypatch.setattr("science_bot.cli.ensure_executor_preflight", lambda: None)
