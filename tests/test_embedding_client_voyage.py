from __future__ import annotations

import json

import httpx
import pytest

from src.config import ConfiguredEmbeddingModelSettings, EmbeddingModelConfig
from src.embedding_client import _EmbeddingClient


@pytest.mark.asyncio
async def test_voyage_embed_uses_query_input_type_and_default_dimension() -> None:
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = dict(json.loads(request.content))
        seen_payloads.append(payload)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": [0.1] * 1024,
                        "index": 0,
                    }
                ],
                "model": payload["model"],
                "usage": {"total_tokens": 1},
            },
        )

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="voyage",
            model="voyage-4-large",
            api_key="test-key",
        ),
        vector_dimensions=1024,
        max_input_tokens=50_000,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )
    assert isinstance(client.client, httpx.AsyncClient)
    await client.client.aclose()
    client.client = httpx.AsyncClient(
        base_url="https://api.voyageai.com/v1",
        transport=httpx.MockTransport(handler),
    )

    try:
        embedding = await client.embed("hello")
    finally:
        assert isinstance(client.client, httpx.AsyncClient)
        await client.client.aclose()

    assert len(embedding) == 1024
    assert client.max_embedding_tokens == 32_000
    assert client.max_batch_size == 1000
    assert client.max_embedding_tokens_per_request == 120_000
    assert seen_payloads == [
        {
            "model": "voyage-4-large",
            "input": ["hello"],
            "input_type": "query",
            "output_dtype": "float",
        }
    ]


@pytest.mark.asyncio
async def test_voyage_batch_embed_uses_document_input_type() -> None:
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = dict(json.loads(request.content))
        seen_payloads.append(payload)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": [float(i)] * 1024,
                        "index": i,
                    }
                    for i, _text in enumerate(payload["input"])
                ],
                "model": payload["model"],
                "usage": {"total_tokens": 2},
            },
        )

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="voyage",
            model="voyage-4-large",
            api_key="test-key",
        ),
        vector_dimensions=1024,
        max_input_tokens=50_000,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )
    assert isinstance(client.client, httpx.AsyncClient)
    await client.client.aclose()
    client.client = httpx.AsyncClient(
        base_url="https://api.voyageai.com/v1",
        transport=httpx.MockTransport(handler),
    )

    try:
        embeddings = await client.simple_batch_embed(["alpha", "beta"])
    finally:
        assert isinstance(client.client, httpx.AsyncClient)
        await client.client.aclose()

    assert [len(embedding) for embedding in embeddings] == [1024, 1024]
    assert seen_payloads == [
        {
            "model": "voyage-4-large",
            "input": ["alpha", "beta"],
            "input_type": "document",
            "output_dtype": "float",
        }
    ]


def test_voyage_embedding_config_defaults_to_voyage_4_large() -> None:
    configured = ConfiguredEmbeddingModelSettings(transport="voyage")

    assert configured.model == "voyage-4-large"
    assert configured.transport == "voyage"
    assert configured.query_input_type == "query"
    assert configured.document_input_type == "document"
    assert configured.output_dtype == "float"


def test_voyage_embedding_config_normalizes_provider_model_shorthand() -> None:
    configured = ConfiguredEmbeddingModelSettings(model="voyage/voyage-4-large")

    assert configured.model == "voyage-4-large"
    assert configured.transport == "voyage"
