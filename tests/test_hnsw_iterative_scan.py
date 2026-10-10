"""Iterative HNSW scans keep filtered vector search from under-filling.

One HNSW index covers every observer, and pgvector stops after
``hnsw.ef_search`` (40) candidates before the observer filter runs. Without an
iterative scan, a request for 75 candidates can return far fewer.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src import models
from src.config import settings
from src.crud import document as document_crud
from src.db import ReadSessionLocal, SessionLocal, connect_args, engine

_DIMENSIONS = settings.EMBEDDING.VECTOR_DIMENSIONS


def _vector(*components: tuple[int, float]) -> list[float]:
    vector = [0.0] * _DIMENSIONS
    for index, value in components:
        vector[index] = value
    return vector


@pytest.mark.asyncio
async def test_app_sessions_use_strict_order_iterative_scan() -> None:
    try:
        for session_factory in (SessionLocal, ReadSessionLocal):
            async with session_factory() as session:
                value = (
                    await session.execute(text("SHOW hnsw.iterative_scan"))
                ).scalar()
            assert value == "strict_order"
    finally:
        await engine.dispose()


async def _seed_documents(
    db_session: AsyncSession, workspace: models.Workspace
) -> tuple[str, str]:
    """Bury one observer's documents behind closer documents of other observers."""
    target = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    observer = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    others = [
        models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
        for _ in range(4)
    ]
    db_session.add_all([target, observer, *others])
    await db_session.flush()
    db_session.add_all(
        [
            models.Collection(
                workspace_name=workspace.name,
                observer=peer.name,
                observed=target.name,
            )
            for peer in (observer, *others)
        ]
    )
    await db_session.flush()

    documents: list[models.Document] = []
    # 200 documents from other observers, all closer to the query than any
    # document of the requested observer.
    for other_index, other in enumerate(others):
        for i in range(50):
            documents.append(
                models.Document(
                    workspace_name=workspace.name,
                    observer=other.name,
                    observed=target.name,
                    content=f"other {other_index}-{i}",
                    embedding=_vector(
                        (0, 1.0), (1, 0.001 * (other_index * 50 + i + 1))
                    ),
                )
            )
    # 80 documents for the requested observer and target.
    for i in range(80):
        documents.append(
            models.Document(
                workspace_name=workspace.name,
                observer=observer.name,
                observed=target.name,
                content=f"wanted {i}",
                embedding=_vector((0, 1.0), (2, 1.0 + 0.01 * i)),
            )
        )
    db_session.add_all(documents)
    await db_session.commit()
    return observer.name, target.name


def _tracked_db_for(test_engine: AsyncEngine) -> Any:
    session_factory = async_sessionmaker(bind=test_engine, expire_on_commit=False)

    @asynccontextmanager
    async def _tracked_db(
        _name: str | None = None, *, read_only: bool = False
    ) -> AsyncGenerator[AsyncSession]:
        del read_only
        async with session_factory() as session:
            # Make the planner choose the HNSW index on this small table, the
            # way it does on the production table.
            for setting in ("enable_seqscan", "enable_bitmapscan", "enable_sort"):
                await session.execute(text(f"SET LOCAL {setting} = off"))
            try:
                yield session
            finally:
                await session.rollback()

    return _tracked_db


async def _candidates_reaching_reranker(
    test_engine: AsyncEngine, workspace_name: str, observer: str, observed: str
) -> tuple[int, str]:
    query_embedding = _vector((0, 1.0))
    tracked_db = _tracked_db_for(test_engine)

    async with tracked_db() as session:
        plan_rows = await session.execute(
            text(
                " ".join(
                    [
                        "EXPLAIN SELECT id FROM documents",
                        "WHERE observer = :observer AND observed = :observed",
                        "ORDER BY embedding <=> CAST(:embedding AS vector) LIMIT 75",
                    ]
                )
            ),
            {
                "observer": observer,
                "observed": observed,
                "embedding": str(query_embedding),
            },
        )
        plan = "\n".join(str(row[0]) for row in plan_rows)

    rerank = AsyncMock(return_value=None)
    with (
        patch.object(document_crud, "tracked_db", tracked_db),
        patch.object(document_crud, "rerank_texts", rerank),
    ):
        results = await document_crud.query_documents(
            None,
            workspace_name,
            "query",
            observer=observer,
            observed=observed,
            embedding=query_embedding,
            top_k=10,
            overfetch_k=75,
            rerank=True,
        )

    # The reranker is skipped when one or no candidate survives the filter.
    if rerank.await_args is None:
        return len(results), plan
    rerank.assert_awaited_once()
    return len(rerank.await_args.kwargs["documents"]), plan


@pytest.mark.asyncio
async def test_overfetch_reaches_reranker_through_filtered_hnsw_scan(
    db_engine: AsyncEngine,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    workspace, _ = sample_data
    observer, observed = await _seed_documents(db_session, workspace)

    without_iterative_args = {
        key: value for key, value in connect_args.items() if key != "options"
    }
    app_engine = create_async_engine(db_engine.url, connect_args=connect_args)
    plain_engine = create_async_engine(
        db_engine.url, connect_args=without_iterative_args
    )
    try:
        plain_count, plain_plan = await _candidates_reaching_reranker(
            plain_engine, workspace.name, observer, observed
        )
        app_count, app_plan = await _candidates_reaching_reranker(
            app_engine, workspace.name, observer, observed
        )
    finally:
        await app_engine.dispose()
        await plain_engine.dispose()

    assert "ix_documents_embedding_hnsw" in plain_plan
    assert "ix_documents_embedding_hnsw" in app_plan
    # The defect: the index stops at ef_search candidates before filtering.
    assert plain_count < 75
    # The fix: the app's connections keep scanning until the filter is met.
    assert app_count == 75
    assert plain_count < app_count
