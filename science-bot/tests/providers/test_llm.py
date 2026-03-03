import asyncio
import json
from pathlib import Path

import pytest
from pydantic import BaseModel
from science_bot.providers import llm
from science_bot.tracing import TraceWriter


class FakeResponseModel(BaseModel):
    value: str


def test_parse_structured_rejects_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(llm.LLMConfigurationError, match="OPENAI_API_KEY is required"):
        asyncio.run(
            llm.parse_structured(
                system_prompt="system",
                user_prompt="user",
                response_model=FakeResponseModel,
            )
        )


def test_parse_structured_rejects_invalid_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(
        llm.LLMConfigurationError,
        match="timeout_seconds must be greater than zero",
    ):
        asyncio.run(
            llm.parse_structured(
                system_prompt="system",
                user_prompt="user",
                response_model=FakeResponseModel,
                timeout_seconds=0,
            )
        )


def test_parse_structured_maps_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeAsyncOpenAI:
        def __init__(self, **_: object) -> None:
            self.responses = self

        async def parse(self, **_: object) -> object:
            raise llm.OpenAIError("boom")

    monkeypatch.setattr(llm, "AsyncOpenAI", FakeAsyncOpenAI)

    with pytest.raises(
        llm.LLMProviderError, match="OpenAI structured response request failed"
    ):
        asyncio.run(
            llm.parse_structured(
                system_prompt="system",
                user_prompt="user",
                response_model=FakeResponseModel,
            )
        )


def test_parse_structured_rejects_missing_parsed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeResponse:
        output_parsed = None

    class FakeAsyncOpenAI:
        def __init__(self, **_: object) -> None:
            self.responses = self

        async def parse(self, **_: object) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(llm, "AsyncOpenAI", FakeAsyncOpenAI)

    with pytest.raises(
        llm.LLMResponseFormatError,
        match="Structured response did not include parsed output",
    ):
        asyncio.run(
            llm.parse_structured(
                system_prompt="system",
                user_prompt="user",
                response_model=FakeResponseModel,
            )
        )


def test_parse_structured_returns_parsed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeResponse:
        output_parsed = FakeResponseModel(value="ok")

    class FakeAsyncOpenAI:
        def __init__(self, **_: object) -> None:
            self.responses = self

        async def parse(self, **_: object) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(llm, "AsyncOpenAI", FakeAsyncOpenAI)

    result = asyncio.run(
        llm.parse_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=FakeResponseModel,
        )
    )

    assert result == FakeResponseModel(value="ok")


def test_parse_structured_writes_trace_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeResponse:
        output_parsed = FakeResponseModel(value="ok")

    class FakeAsyncOpenAI:
        def __init__(self, **_: object) -> None:
            self.responses = self

        async def parse(self, **_: object) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(llm, "AsyncOpenAI", FakeAsyncOpenAI)
    trace_writer = TraceWriter.for_run(tmp_path / "traces")

    asyncio.run(
        llm.parse_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=FakeResponseModel,
            trace_writer=trace_writer,
            trace_stage="classification",
        )
    )

    events_path = trace_writer.root_dir / "events.jsonl"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [event["event"] for event in events] == ["llm_request", "llm_response"]
