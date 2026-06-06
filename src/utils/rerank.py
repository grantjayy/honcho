from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from logging import getLogger

import httpx

logger = getLogger(__name__)

VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"
DEFAULT_RERANK_MODEL = "rerank-2.5"
DEFAULT_RERANK_TIMEOUT_SECONDS = 3.0


@dataclass(frozen=True)
class RerankResult:
    index: int
    relevance_score: float | None = None


async def rerank_texts(
    *,
    query: str,
    documents: Sequence[str],
    top_k: int,
    model: str = DEFAULT_RERANK_MODEL,
    timeout: float = DEFAULT_RERANK_TIMEOUT_SECONDS,
) -> list[RerankResult] | None:
    """Rerank documents with Voyage and return ranked input indices.

    This helper is deliberately best-effort. Any missing credentials, network
    failure, malformed response, or empty result returns None so callers can
    preserve their original vector ranking.
    """
    api_key = (
        os.getenv("VOYAGE_API_KEY")
        or os.getenv("VOYAGE_AI_API_KEY")
        or os.getenv("RERANK_API_KEY")
    )
    if not api_key or not query or not documents or top_k <= 0:
        return None

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                VOYAGE_RERANK_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "query": query,
                    "documents": list(documents),
                    "top_k": min(top_k, len(documents)),
                },
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        logger.info("Voyage rerank unavailable; falling back to vector order: %s", exc)
        return None

    results = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return None

    parsed: list[RerankResult] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(documents):
            continue
        score = item.get("relevance_score")
        parsed.append(
            RerankResult(
                index=idx,
                relevance_score=score if isinstance(score, (int, float)) else None,
            )
        )

    return parsed or None
