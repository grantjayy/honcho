from unittest.mock import Mock

import httpx
import pytest

from src.utils.rerank import rerank_texts


@pytest.mark.asyncio
async def test_rerank_texts_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_AI_API_KEY", raising=False)
    monkeypatch.delenv("RERANK_API_KEY", raising=False)

    assert await rerank_texts(query="hello", documents=["a", "b"], top_k=2) is None


@pytest.mark.asyncio
async def test_rerank_texts_returns_none_on_http_error(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")

    class FailingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            raise httpx.TimeoutException("slow")

    monkeypatch.setattr(httpx, "AsyncClient", FailingClient)

    assert await rerank_texts(query="hello", documents=["a", "b"], top_k=2) is None


@pytest.mark.asyncio
async def test_rerank_texts_returns_none_on_malformed_response(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"unexpected": []}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            return response

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    assert await rerank_texts(query="hello", documents=["a", "b"], top_k=2) is None


@pytest.mark.asyncio
async def test_rerank_texts_parses_voyage_ranking(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    calls = []
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [
            {"index": 2, "relevance_score": 0.99},
            {"index": 0, "relevance_score": 0.55},
        ]
    }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            calls.append(("init", args, kwargs))

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            calls.append(("post", args, kwargs))
            return response

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    results = await rerank_texts(
        query="honcho retrieval",
        documents=["a", "b", "c"],
        top_k=2,
        model="rerank-2.5-lite",
        timeout=3.0,
    )

    assert results is not None
    assert [result.index for result in results] == [2, 0]
    assert [result.relevance_score for result in results] == [0.99, 0.55]
    post_call = next(call for call in calls if call[0] == "post")
    assert post_call[2]["json"]["model"] == "rerank-2.5-lite"
    assert post_call[2]["json"]["top_k"] == 2
    assert post_call[2]["headers"]["Authorization"] == "Bearer test-key"
