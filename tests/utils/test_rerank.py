from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from typing import Any, cast

import httpx
import pytest

from src.utils import rerank
from src.utils.rerank import DEFAULT_RERANK_MODEL, rerank_texts


@pytest.fixture(autouse=True)
def reset_shared_client() -> Iterator[None]:
    rerank._client = None  # pyright: ignore[reportPrivateUsage]
    rerank._client_loop = None  # pyright: ignore[reportPrivateUsage]
    yield
    rerank._client = None  # pyright: ignore[reportPrivateUsage]
    rerank._client_loop = None  # pyright: ignore[reportPrivateUsage]


class _ClientFactory:
    """Stands in for httpx.AsyncClient: builds real clients on a mock transport
    and counts how many were created."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self._handler: Callable[[httpx.Request], httpx.Response] = handler
        self._real: type[httpx.AsyncClient] = httpx.AsyncClient
        self.created: list[httpx.AsyncClient] = []

    def __call__(self, *_args: object, **kwargs: Any) -> httpx.AsyncClient:
        client = self._real(transport=httpx.MockTransport(self._handler), **kwargs)
        self.created.append(client)
        return client


def _install(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> _ClientFactory:
    factory = _ClientFactory(handler)
    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return factory


def _ranking(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": [
                {"index": 2, "relevance_score": 0.99},
                {"index": 0, "relevance_score": 0.55},
            ]
        },
    )


@pytest.mark.asyncio
async def test_rerank_texts_returns_none_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_AI_API_KEY", raising=False)
    monkeypatch.delenv("RERANK_API_KEY", raising=False)

    assert await rerank_texts(query="hello", documents=["a", "b"], top_k=2) is None


@pytest.mark.asyncio
async def test_rerank_texts_returns_none_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow", request=request)

    _install(monkeypatch, slow)

    assert await rerank_texts(query="hello", documents=["a", "b"], top_k=2) is None


@pytest.mark.asyncio
async def test_rerank_texts_returns_none_on_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    _install(monkeypatch, lambda _request: httpx.Response(200, json={"unexpected": []}))

    assert await rerank_texts(query="hello", documents=["a", "b"], top_k=2) is None


@pytest.mark.asyncio
async def test_rerank_texts_parses_voyage_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _ranking(request)

    _install(monkeypatch, handler)

    results = await rerank_texts(
        query="honcho retrieval",
        documents=["a", "b", "c"],
        top_k=2,
        timeout=3.0,
    )

    assert results is not None
    assert [result.index for result in results] == [2, 0]
    assert [result.relevance_score for result in results] == [0.99, 0.55]
    [request] = requests
    post_json = cast(dict[str, object], json.loads(request.content))
    assert str(request.url) == "https://api.voyageai.com/v1/rerank"
    assert DEFAULT_RERANK_MODEL == "rerank-2.5"
    assert post_json["model"] == "rerank-2.5"
    assert post_json["top_k"] == 2
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.extensions["timeout"]["read"] == 3.0


@pytest.mark.asyncio
async def test_rerank_texts_reuses_one_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    factory = _install(monkeypatch, _ranking)

    for _ in range(2):
        assert await rerank_texts(query="q", documents=["a", "b", "c"], top_k=2)

    assert len(factory.created) == 1


@pytest.mark.asyncio
async def test_close_rerank_client_closes_and_clears_shared_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    factory = _install(monkeypatch, _ranking)

    assert await rerank_texts(query="q", documents=["a", "b", "c"], top_k=2)
    await rerank.close_rerank_client()

    assert factory.created[0].is_closed
    assert rerank._client is None  # pyright: ignore[reportPrivateUsage]
    await rerank.close_rerank_client()  # idempotent


@pytest.mark.asyncio
async def test_rerank_texts_replaces_a_closed_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    factory = _install(monkeypatch, _ranking)

    assert await rerank_texts(query="q", documents=["a", "b", "c"], top_k=2)
    await factory.created[0].aclose()
    assert await rerank_texts(query="q", documents=["a", "b", "c"], top_k=2)

    assert len(factory.created) == 2


def test_rerank_texts_replaces_a_client_from_a_closed_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    factory = _install(monkeypatch, _ranking)

    async def call() -> None:
        assert await rerank_texts(query="q", documents=["a", "b", "c"], top_k=2)

    asyncio.run(call())
    asyncio.run(call())

    assert len(factory.created) == 2
