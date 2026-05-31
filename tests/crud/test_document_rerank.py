from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from src import models
from src.crud import document
from src.utils.rerank import RerankResult


def _doc(content: str) -> models.Document:
    return models.Document(
        workspace_name="workspace",
        observer="observer",
        observed="observed",
        content=content,
    )


@pytest.mark.asyncio
async def test_maybe_rerank_documents_reorders_by_rerank_indices(monkeypatch):
    docs = [_doc("first"), _doc("second"), _doc("third")]
    monkeypatch.setattr(
        document,
        "rerank_texts",
        AsyncMock(
            return_value=[
                RerankResult(index=2, relevance_score=0.9),
                RerankResult(index=0, relevance_score=0.8),
            ]
        ),
    )

    result = await document._maybe_rerank_documents(
        query="query", documents=docs, top_k=2, rerank=True
    )

    assert [doc.content for doc in result] == ["third", "first"]


@pytest.mark.asyncio
async def test_maybe_rerank_documents_falls_back_to_vector_order(monkeypatch):
    docs = [_doc("first"), _doc("second"), _doc("third")]
    monkeypatch.setattr(document, "rerank_texts", AsyncMock(return_value=None))

    result = await document._maybe_rerank_documents(
        query="query", documents=docs, top_k=2, rerank=True
    )

    assert [doc.content for doc in result] == ["first", "second"]


@pytest.mark.asyncio
async def test_query_documents_overfetches_before_reranking(monkeypatch):
    fetched_top_k = None
    docs = [_doc(str(i)) for i in range(75)]

    class FakeDb:
        def expunge(self, doc):
            pass

    @asynccontextmanager
    async def fake_tracked_db(_name):
        yield FakeDb()

    async def fake_pgvector(*args):
        nonlocal fetched_top_k
        fetched_top_k = args[-1]
        return docs

    monkeypatch.setattr(document, "_uses_pgvector", lambda: True)
    monkeypatch.setattr(document, "tracked_db", fake_tracked_db)
    monkeypatch.setattr(document, "_query_documents_pgvector", fake_pgvector)
    mock_rerank = AsyncMock(return_value=docs[:12])
    monkeypatch.setattr(document, "_maybe_rerank_documents", mock_rerank)

    result = await document.query_documents(
        None,
        "workspace",
        "query",
        observer="observer",
        observed="observed",
        embedding=[0.1],
        top_k=12,
        overfetch_k=75,
        rerank=True,
    )

    assert fetched_top_k == 75
    assert len(result) == 12
    mock_rerank.assert_awaited_once()
