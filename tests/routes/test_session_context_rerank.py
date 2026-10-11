"""Session context search reranks its candidates like the peer context route.

The reranker decides which conclusions are selected. The representation still
presents them chronologically, so these tests assert membership in the
response and check rank order only on the candidates handed to the reranker.
"""

from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.routers.peers import FOCUSED_CONTEXT_OVERFETCH_K
from src.utils.rerank import RerankResult

_DIMENSIONS = settings.EMBEDDING.VECTOR_DIMENSIONS
_QUERY = "what does the user care about?"


def _vector(*components: tuple[int, float]) -> list[float]:
    vector = [0.0] * _DIMENSIONS
    for index, value in components:
        vector[index] = value
    return vector


def _content(index: int, session_label: str = "a") -> str:
    return f"fact-{session_label}-{index:03d}"


async def _seed(
    client: TestClient,
    db_session: AsyncSession,
    workspace: models.Workspace,
    peer: models.Peer,
    *,
    per_session: dict[str, int],
) -> dict[str, str]:
    """Create one session per label and documents whose vector order is known.

    Within the whole collection, documents are ordered by distance first by
    their index, then by session label, so `fact-a-000` is nearest.
    """
    sessions: dict[str, str] = {}
    for label in per_session:
        session_id = str(generate_nanoid())
        response = client.post(
            f"/v3/workspaces/{workspace.name}/sessions",
            json={"id": session_id, "peers": {peer.name: {}}},
        )
        assert response.status_code in (200, 201)
        sessions[label] = session_id

    db_session.add(
        models.Collection(
            workspace_name=workspace.name, observer=peer.name, observed=peer.name
        )
    )
    await db_session.flush()

    labels = sorted(per_session)
    documents: list[models.Document] = []
    for label_index, label in enumerate(labels):
        for i in range(per_session[label]):
            distance_rank = i * len(labels) + label_index + 1
            documents.append(
                models.Document(
                    workspace_name=workspace.name,
                    observer=peer.name,
                    observed=peer.name,
                    session_name=sessions[label],
                    content=_content(i, label),
                    embedding=_vector((0, 1.0), (1, 0.001 * distance_rank)),
                )
            )
    db_session.add_all(documents)
    await db_session.commit()
    return sessions


def _context(
    client: TestClient,
    workspace: models.Workspace,
    session_id: str,
    peer: models.Peer,
    **params: Any,
) -> str:
    response = client.get(
        f"/v3/workspaces/{workspace.name}/sessions/{session_id}/context",
        params={"peer_target": peer.name, "search_query": _QUERY, **params},
    )
    assert response.status_code == 200, response.text
    representation = response.json()["peer_representation"]
    assert isinstance(representation, str)
    return representation


def _selected(representation: str, candidates: Sequence[str]) -> set[str]:
    return {content for content in candidates if content in representation}


def _rank(order: list[str]) -> AsyncMock:
    """A reranker that ranks documents by a fixed preference list."""

    async def _rerank(
        *, query: str, documents: Sequence[str], top_k: int, **_: Any
    ) -> list[RerankResult]:
        assert query == _QUERY
        preferred = [
            documents.index(content) for content in order if content in documents
        ]
        rest = [i for i in range(len(documents)) if i not in preferred]
        return [RerankResult(index=i) for i in (preferred + rest)[:top_k]]

    return AsyncMock(side_effect=_rerank)


@pytest.fixture
def fixed_query_embedding(mock_openai_embeddings: dict[str, Any]) -> None:
    def _embed(_text: str) -> list[float]:
        return _vector((0, 1.0))

    mock_openai_embeddings["embed"].side_effect = _embed


@pytest.mark.asyncio
@pytest.mark.usefixtures("fixed_query_embedding")
async def test_session_context_search_selects_reranker_top_k(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    workspace, peer = sample_data
    sessions = await _seed(client, db_session, workspace, peer, per_session={"a": 80})
    all_contents = [_content(i) for i in range(80)]
    reranker_pick = [_content(i) for i in (70, 3, 55, 10, 41)]
    rerank = _rank(reranker_pick)

    with patch("src.crud.document.rerank_texts", rerank):
        representation = _context(
            client,
            workspace,
            sessions["a"],
            peer,
            search_top_k=5,
            max_conclusions=5,
        )

    rerank.assert_awaited_once()
    assert rerank.await_args is not None
    candidates = list(rerank.await_args.kwargs["documents"])
    assert len(candidates) == FOCUSED_CONTEXT_OVERFETCH_K == 75
    # Rank order is checked before representation assembly: the reranker gets
    # the nearest 75 in vector order.
    assert candidates == all_contents[:75]
    assert rerank.await_args.kwargs["top_k"] == 5
    assert _selected(representation, all_contents) == set(reranker_pick)


@pytest.mark.asyncio
@pytest.mark.usefixtures("fixed_query_embedding")
@pytest.mark.parametrize("scoping", ["limit_to_session", "sessions"])
async def test_session_context_rerank_never_selects_other_sessions(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    scoping: str,
) -> None:
    workspace, peer = sample_data
    sessions = await _seed(
        client, db_session, workspace, peer, per_session={"a": 10, "b": 10, "c": 10}
    )
    in_scope_labels = ["a"] if scoping == "limit_to_session" else ["a", "c"]
    out_of_scope_labels = [label for label in "abc" if label not in in_scope_labels]
    out_of_scope = [
        _content(i, label) for label in out_of_scope_labels for i in range(10)
    ]
    all_contents = [_content(i, label) for label in "abc" for i in range(10)]
    # The reranker would put every out-of-scope conclusion first.
    rerank = _rank(out_of_scope)

    params: dict[str, Any] = {"search_top_k": 4, "max_conclusions": 4}
    if scoping == "limit_to_session":
        params["limit_to_session"] = True
    else:
        params["sessions"] = [sessions[label] for label in in_scope_labels]

    with patch("src.crud.document.rerank_texts", rerank):
        scoped = _context(client, workspace, sessions["a"], peer, **params)
        params.pop("limit_to_session", None)
        params.pop("sessions", None)
        unscoped = _context(client, workspace, sessions["a"], peer, **params)

    scoped_call, unscoped_call = rerank.await_args_list
    assert not set(out_of_scope) & set(scoped_call.kwargs["documents"])
    assert not _selected(scoped, out_of_scope)
    assert len(_selected(scoped, all_contents)) == 4
    # Control: without the restriction the same reranker selects them.
    assert set(out_of_scope) <= set(unscoped_call.kwargs["documents"])
    assert _selected(unscoped, all_contents) <= set(out_of_scope)
    assert len(_selected(unscoped, all_contents)) == 4


@pytest.mark.asyncio
@pytest.mark.usefixtures("fixed_query_embedding")
async def test_session_context_falls_back_to_vector_order_without_reranker(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    workspace, peer = sample_data
    sessions = await _seed(client, db_session, workspace, peer, per_session={"a": 80})
    all_contents = [_content(i) for i in range(80)]
    rerank = AsyncMock(return_value=None)

    with patch("src.crud.document.rerank_texts", rerank):
        representation = _context(
            client,
            workspace,
            sessions["a"],
            peer,
            search_top_k=5,
            max_conclusions=5,
        )

    rerank.assert_awaited_once()
    assert _selected(representation, all_contents) == set(all_contents[:5])


@pytest.mark.asyncio
@pytest.mark.usefixtures("fixed_query_embedding")
async def test_session_context_without_search_query_does_not_rerank(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    workspace, peer = sample_data
    sessions = await _seed(client, db_session, workspace, peer, per_session={"a": 5})
    rerank = AsyncMock(return_value=None)

    with patch("src.crud.document.rerank_texts", rerank):
        response = client.get(
            f"/v3/workspaces/{workspace.name}/sessions/{sessions['a']}/context",
            params={"peer_target": peer.name},
        )

    assert response.status_code == 200
    rerank.assert_not_awaited()
