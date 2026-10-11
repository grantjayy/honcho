"""Unparseable deriver output is retried and reaches the fallback model.

These tests drive the real ``honcho_llm_call`` retry and fallback chain and the
real OpenAI SDK, with only the HTTP transport replaced. The deriver calls the
OpenAI transport with ``response_model=PromptRepresentation``.
"""

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError
from tenacity import wait_none

from src.config import ConfiguredModelSettings, FallbackModelSettings
from src.llm import CLIENTS, api, honcho_llm_call
from src.utils.representation import PromptRepresentation

PRIMARY = "primary-model"
FALLBACK = "fallback-model"
VALID_JSON = '{"explicit": [{"content": "the user likes tea"}]}'
TRUNCATED_PROSE = "I could not find any durable facts about the user in these mes"
PROSE = "Sure! Here are the facts:\n- the user likes coffee"


def _completion(content: str, finish_reason: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class _Provider:
    """Serves a fixed sequence of completions and records each request's model."""

    def __init__(self, replies: list[tuple[str, str]]) -> None:
        self._replies: Iterator[tuple[str, str]] = iter(replies)
        self.models: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.models.append(json.loads(request.content)["model"])
        content, finish_reason = next(self._replies)
        return httpx.Response(200, json=_completion(content, finish_reason))


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "wait_exponential", Mock(return_value=wait_none()))


async def _derive(
    replies: list[tuple[str, str]], *, structured_output_mode: str | None = None
) -> tuple[_Provider, Any]:
    provider = _Provider(replies)
    client = AsyncOpenAI(
        api_key="test-key",
        base_url="https://llm.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider.handler)),
    )
    model_config = ConfiguredModelSettings(
        model=PRIMARY,
        transport="openai",
        structured_output_mode=structured_output_mode,  # pyright: ignore[reportArgumentType]
        fallback=FallbackModelSettings(
            model=FALLBACK,
            transport="openai",
            structured_output_mode=structured_output_mode,  # pyright: ignore[reportArgumentType]
        ),
    )
    try:
        with patch.dict(CLIENTS, {"openai": client}):
            response = await honcho_llm_call(
                model_config=model_config,
                prompt="Derive observations",
                max_tokens=100,
                response_model=PromptRepresentation,
                json_mode=True,
                enable_retry=True,
                retry_attempts=3,
            )
    finally:
        await client.close()
    return provider, response


# Unparseable, non-empty output on each structured-output path the deriver can
# take: json_schema parse() cut off at max_tokens, and json_object mode.
UNPARSEABLE = [
    pytest.param(TRUNCATED_PROSE, "length", None, id="json_schema-truncated-prose"),
    pytest.param(PROSE, "stop", "json_object", id="json_object-prose"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("bad_content", "finish_reason", "mode"), UNPARSEABLE)
async def test_unparseable_output_is_retried_until_valid_json(
    bad_content: str, finish_reason: str, mode: str | None
) -> None:
    provider, response = await _derive(
        [
            (bad_content, finish_reason),
            (bad_content, finish_reason),
            (VALID_JSON, "stop"),
        ],
        structured_output_mode=mode,
    )

    assert provider.models == [PRIMARY, PRIMARY, FALLBACK]
    assert isinstance(response.content, PromptRepresentation)
    assert [obs.content for obs in response.content.explicit] == ["the user likes tea"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("bad_content", "finish_reason", "mode"), UNPARSEABLE)
async def test_unparseable_output_on_every_attempt_raises_after_fallback(
    bad_content: str, finish_reason: str, mode: str | None
) -> None:
    with pytest.raises(ValidationError):
        await _derive([(bad_content, finish_reason)] * 3, structured_output_mode=mode)


@pytest.mark.asyncio
async def test_unparseable_output_final_attempt_uses_fallback_config() -> None:
    provider = _Provider([(TRUNCATED_PROSE, "length")] * 3)
    client = AsyncOpenAI(
        api_key="test-key",
        base_url="https://llm.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider.handler)),
    )
    model_config = ConfiguredModelSettings(
        model=PRIMARY,
        transport="openai",
        fallback=FallbackModelSettings(model=FALLBACK, transport="openai"),
    )
    try:
        with (
            patch.dict(CLIENTS, {"openai": client}),
            pytest.raises(ValidationError),
        ):
            await honcho_llm_call(
                model_config=model_config,
                prompt="Derive observations",
                max_tokens=100,
                response_model=PromptRepresentation,
                json_mode=True,
                enable_retry=True,
                retry_attempts=3,
            )
    finally:
        await client.close()

    assert provider.models == [PRIMARY, PRIMARY, FALLBACK]


@pytest.mark.asyncio
async def test_truncated_json_is_repaired_without_retry() -> None:
    provider, response = await _derive(
        [('{"explicit": [{"content": "the user likes tea"}', "length")]
    )

    assert provider.models == [PRIMARY]
    assert [obs.content for obs in response.content.explicit] == ["the user likes tea"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "json_object"])
async def test_valid_empty_explicit_list_is_not_retried(mode: str | None) -> None:
    provider, response = await _derive(
        [('{"explicit": []}', "stop")], structured_output_mode=mode
    )

    assert provider.models == [PRIMARY]
    assert response.content == PromptRepresentation(explicit=[])


@pytest.mark.asyncio
async def test_json_object_empty_content_returns_empty_without_retry() -> None:
    provider, response = await _derive(
        [("", "stop")], structured_output_mode="json_object"
    )

    assert provider.models == [PRIMARY]
    assert response.content == PromptRepresentation(explicit=[])


@pytest.mark.asyncio
async def test_truncated_empty_content_is_retried_then_uses_fallback() -> None:
    # Production failure mode: a reasoning model spends all of max_tokens on
    # reasoning and returns finish_reason=length with no text.
    provider, response = await _derive(
        [("", "length"), ("", "length"), (VALID_JSON, "stop")]
    )

    assert provider.models == [PRIMARY, PRIMARY, FALLBACK]
    assert [obs.content for obs in response.content.explicit] == ["the user likes tea"]
