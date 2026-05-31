import asyncio
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any, Literal, NamedTuple, TypeVar, cast

import httpx
import tiktoken
from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI

from .config import EmbeddingModelConfig, resolve_embedding_model_config, settings

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


async def _emit_embedding_call(
    *,
    provider: str,
    model: str,
    texts: list[str],
    input_tokens_estimate: int,
    fn: Callable[[], Awaitable[_T]],
    is_final_attempt: bool = True,
) -> _T:
    """time a single embedding-provider call, emit
    `embedding.call.completed` on both success and exception, and return the
    call's result. Errors propagate unchanged — telemetry never bleeds into
    the caller's control flow.

    Caller-supplied `texts` is used only for `input_count`; we don't keep the
    list around for the event to avoid leaking content into telemetry.

    `is_final_attempt` defaults to True so one-shot callers (`embed`,
    `simple_batch_embed`) get correct semantics without changes. Retry-loop
    callers (`_process_batch`) pass the real attempt index so dashboards
    can distinguish exhausted retries from mid-retry failures.
    """
    start = time.perf_counter()
    error: BaseException | None = None
    try:
        return await fn()
    except BaseException as exc:
        error = exc
        raise
    finally:
        if error is None:
            outcome: Literal["success", "error", "cancelled"] = "success"
        elif isinstance(error, asyncio.CancelledError):
            outcome = "cancelled"
        else:
            outcome = "error"
        _publish_embedding_event(
            provider=provider,
            model=model,
            input_count=len(texts),
            input_tokens_estimate=input_tokens_estimate,
            duration_ms=(time.perf_counter() - start) * 1000,
            outcome=outcome,
            error=error,
            is_final_attempt=is_final_attempt,
        )


def _publish_embedding_event(
    *,
    provider: str,
    model: str,
    input_count: int,
    input_tokens_estimate: int,
    duration_ms: float,
    outcome: Literal["success", "error", "cancelled"],
    error: BaseException | None,
    is_final_attempt: bool,
) -> None:
    """Build and emit the EmbeddingCallCompletedEvent. Best-effort."""
    try:
        from src.telemetry.events import (
            EmbeddingCallCompletedEvent,
            EmbeddingCallPurpose,
            emit,
        )
        from src.utils.types import (
            get_embedding_call_purpose,
            get_embedding_parent_category,
            get_embedding_run_id,
            get_embedding_workspace_name,
        )

        # call_purpose travels via ContextVar so embedding callers don't have
        # to thread it through every call site. Unknown values drop to None
        # rather than raising — keeps telemetry resilient to drift.
        purpose_slug = get_embedding_call_purpose()
        call_purpose: EmbeddingCallPurpose | None = None
        if purpose_slug:
            try:
                call_purpose = EmbeddingCallPurpose(purpose_slug)
            except ValueError:
                logger.debug(
                    "Unknown embedding_call_purpose=%r; emitting without",
                    purpose_slug,
                )

        emit(
            EmbeddingCallCompletedEvent(
                workspace_name=get_embedding_workspace_name(),
                call_purpose=call_purpose,
                parent_category=get_embedding_parent_category(),
                provider=provider,
                model=model,
                input_count=input_count,
                input_tokens_estimate=input_tokens_estimate,
                duration_ms=duration_ms,
                outcome=outcome,
                is_final_attempt=is_final_attempt,
                error_class=type(error).__name__ if error is not None else None,
                run_id=get_embedding_run_id(),
            )
        )
    except Exception:  # pragma: no cover - telemetry must not raise
        logger.debug("Failed to emit EmbeddingCallCompletedEvent", exc_info=True)


class BatchItem(NamedTuple):
    """A single item in a batch with its metadata."""

    text: str
    text_id: str
    chunk_index: int
    token_count: int


class _EmbeddingClient:
    """
    Embedding client supporting OpenAI, Gemini, and Voyage with chunking and batching support.
    """

    def __init__(
        self,
        config: EmbeddingModelConfig,
        *,
        vector_dimensions: int,
        max_input_tokens: int,
        max_tokens_per_request: int,
        send_dimensions: bool,
    ):
        self.transport: str = config.transport
        self.model: str = config.model
        self.vector_dimensions: int = vector_dimensions
        self.send_dimensions: bool = send_dimensions

        self.query_input_type = config.query_input_type
        self.document_input_type = config.document_input_type
        self.output_dtype = config.output_dtype

        if self.transport == "gemini":
            if not config.api_key:
                raise ValueError("Gemini API key is required")
            http_options = (
                genai_types.HttpOptions(base_url=config.base_url)
                if config.base_url
                else None
            )
            self.client: genai.Client | AsyncOpenAI | httpx.AsyncClient = genai.Client(
                api_key=config.api_key,
                http_options=http_options,
            )
            # Gemini has a 2048 token limit
            self.max_embedding_tokens: int = min(max_input_tokens, 2048)
            # Gemini batch size is not documented, using conservative estimate
            self.max_batch_size: int = 100
        elif self.transport == "voyage":
            if not config.api_key:
                raise ValueError("Voyage API key is required")
            self.client = httpx.AsyncClient(
                base_url=config.base_url or "https://api.voyageai.com/v1",
                headers={"Authorization": f"Bearer {config.api_key}"},
                timeout=60.0,
            )
            # Voyage 4 models support 32K-token inputs. Voyage caps the number
            # of texts per embeddings request at 1,000 and has per-request
            # token ceilings by model; keep the default conservative for
            # voyage-4-large, which is the production target here.
            self.max_embedding_tokens = min(max_input_tokens, 32_000)
            self.max_batch_size = 1000
            self.max_embedding_tokens_per_request = min(
                max_tokens_per_request, 120_000
            )
        else:  # openai
            if not config.api_key:
                raise ValueError("OpenAI API key is required")
            self.client = AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
            )
            self.max_embedding_tokens = max_input_tokens
            self.max_batch_size = 2048  # OpenAI batch limit

        try:
            self.encoding: tiktoken.Encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        if self.transport != "voyage":
            self.max_embedding_tokens_per_request: int = max_tokens_per_request

    @property
    def provider(self) -> str:
        return self.transport

    def _validate_embedding_dimensions(self, embedding: list[float]) -> list[float]:
        if len(embedding) != self.vector_dimensions:
            raise ValueError(
                f"Embedding dimension mismatch for {self.transport}:{self.model}. "
                + f"Expected {self.vector_dimensions}, got {len(embedding)}."
            )
        return embedding

    async def _voyage_embeddings(
        self, texts: list[str], *, input_type: str | None
    ) -> list[list[float]]:
        if not isinstance(self.client, httpx.AsyncClient):
            raise TypeError("Voyage transport requires an httpx.AsyncClient")

        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "output_dtype": self.output_dtype,
        }
        if self.send_dimensions:
            payload["output_dimension"] = self.vector_dimensions
        if input_type is not None:
            payload["input_type"] = input_type

        response = await self.client.post("/embeddings", json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ValueError(
                f"Voyage embeddings request failed: {exc.response.text}"
            ) from exc

        data = response.json().get("data") or []
        indexed = sorted(data, key=lambda item: int(item.get("index", 0)))
        embeddings = [
            self._validate_embedding_dimensions(item["embedding"]) for item in indexed
        ]
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Voyage returned {len(embeddings)} embeddings for {len(texts)} inputs"
            )
        return embeddings

    async def embed(self, query: str) -> list[float]:
        token_count = len(self.encoding.encode(query))

        if token_count > self.max_embedding_tokens:
            raise ValueError(
                f"Query exceeds maximum token limit of {self.max_embedding_tokens} tokens (got {token_count} tokens)"
            )

        # Bind the typed client at the dispatch site so pyright can narrow it
        # for the closures without needing `assert isinstance(...)` (bandit
        # B101). The closures close over the narrowed local, not `self.client`.
        if self.transport == "gemini":
            gemini_client = cast(genai.Client, self.client)

            async def _call_gemini() -> list[float]:
                response = await gemini_client.aio.models.embed_content(
                    model=self.model,
                    contents=query,
                    config={"output_dimensionality": self.vector_dimensions},
                )
                if not response.embeddings or not response.embeddings[0].values:
                    raise ValueError("No embedding returned from Gemini API")
                return self._validate_embedding_dimensions(
                    response.embeddings[0].values
                )

            return await _emit_embedding_call(
                provider=self.transport,
                model=self.model,
                texts=[query],
                input_tokens_estimate=token_count,
                fn=_call_gemini,
            )

        if self.transport == "voyage":

            async def _call_voyage() -> list[float]:
                return (
                    await self._voyage_embeddings(
                        [query], input_type=self.query_input_type
                    )
                )[0]

            return await _emit_embedding_call(
                provider=self.transport,
                model=self.model,
                texts=[query],
                input_tokens_estimate=token_count,
                fn=_call_voyage,
            )

        openai_client = cast(AsyncOpenAI, self.client)

        async def _call_openai() -> list[float]:
            openai_kwargs: dict[str, Any] = {"model": self.model, "input": [query]}
            if self.send_dimensions:
                openai_kwargs["dimensions"] = self.vector_dimensions
            response = await openai_client.embeddings.create(**openai_kwargs)
            return self._validate_embedding_dimensions(response.data[0].embedding)

        return await _emit_embedding_call(
            provider=self.transport,
            model=self.model,
            texts=[query],
            input_tokens_estimate=token_count,
            fn=_call_openai,
        )

    async def simple_batch_embed(self, texts: list[str]) -> list[list[float]]:
        """
        Simple batch embedding for a list of text strings.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors corresponding to input texts

        Raises:
            ValueError: If any text exceeds token limits
        """
        embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]

            async def _embed_batch(batch: list[str] = batch) -> list[list[float]]:
                """One provider call for one batch. Lifted into a closure so
                _emit_embedding_call can time + emit + propagate errors."""
                batch_embeddings: list[list[float]] = []
                if self.transport == "gemini":
                    gemini_client = cast(genai.Client, self.client)
                    # Type cast needed due to genai type signature complexity
                    response = await gemini_client.aio.models.embed_content(
                        model=self.model,
                        contents=batch,  # pyright: ignore[reportArgumentType]
                        config={"output_dimensionality": self.vector_dimensions},
                    )
                    if response.embeddings:
                        for emb in response.embeddings:
                            if emb.values:
                                batch_embeddings.append(
                                    self._validate_embedding_dimensions(emb.values)
                                )
                elif self.transport == "voyage":
                    batch_embeddings.extend(
                        await self._voyage_embeddings(
                            batch, input_type=self.document_input_type
                        )
                    )
                else:  # openai
                    openai_kwargs: dict[str, Any] = {
                        "input": batch,
                        "model": self.model,
                    }
                    if self.send_dimensions:
                        openai_kwargs["dimensions"] = self.vector_dimensions
                    openai_client = cast(AsyncOpenAI, self.client)
                    response = await openai_client.embeddings.create(**openai_kwargs)
                    batch_embeddings.extend(
                        [
                            self._validate_embedding_dimensions(data.embedding)
                            for data in response.data
                        ]
                    )
                return batch_embeddings

            try:
                # Pre-compute the tiktoken estimate ONCE for telemetry; the
                # batch contents don't change between attempts.
                tokens_estimate = sum(len(self.encoding.encode(t)) for t in batch)
                batch_embeddings = await _emit_embedding_call(
                    provider=self.transport,
                    model=self.model,
                    texts=batch,
                    input_tokens_estimate=tokens_estimate,
                    fn=_embed_batch,
                )
                embeddings.extend(batch_embeddings)
            except Exception as e:
                # Check if it's a token limit error and re-raise as ValueError for consistency
                if "token" in str(e).lower():
                    raise ValueError(
                        f"Text content exceeds maximum token limit of {self.max_embedding_tokens}."
                    ) from e
                raise

        return embeddings

    async def batch_embed(
        self, id_resource_dict: dict[str, str]
    ) -> dict[str, list[list[float]]]:
        """
        Embed multiple texts, chunking long ones and batching API calls.

        Args:
            id_resource_dict: Maps text IDs to text content

        Returns:
            Maps text IDs to lists of embedding vectors (one per chunk)
        """
        if not id_resource_dict:
            return {}

        # 1. Prepare chunks for all texts if needed
        text_chunks = self._prepare_chunks(id_resource_dict)

        # 2. Create batches that fit API limits (max 2048 embeddings per request, max 300,000 tokens per request)
        batches = self._create_batches(text_chunks)

        # 3. Process all batches concurrently
        batch_results = await asyncio.gather(
            *[self._process_batch(batch) for batch in batches],
        )

        # 4. Accumulate results preserving chunk order
        return self._accumulate_embeddings(batch_results)

    def _prepare_chunks(
        self, id_resource_dict: dict[str, str]
    ) -> dict[str, list[tuple[str, int]]]:
        """
        Chunk texts that exceed token limits.

        Args:
            id_resource_dict: Maps text IDs to text content. We tokenize with
                the embedding client's own encoding so token IDs match the
                decoder vocabulary used by the target embedding API.

        Returns:
            Maps text IDs to lists of (chunk_text, token_count) tuples
        """
        out: dict[str, list[tuple[str, int]]] = {}
        for text_id, text in id_resource_dict.items():
            tokens = self.encoding.encode(text)
            if len(tokens) > self.max_embedding_tokens:
                out[text_id] = _chunk_text_with_tokens(
                    text, tokens, self.max_embedding_tokens, self.encoding
                )
            else:
                out[text_id] = [(text, len(tokens))]
        return out

    def _create_batches(
        self, text_chunks: dict[str, list[tuple[str, int]]]
    ) -> list[list[BatchItem]]:
        """
        Group chunks into batches that fit API limits.

        Args:
            text_chunks: Maps text IDs to lists of (chunk_text, token_count) tuples

        Returns:
            List of batches, each containing BatchItem objects
        """
        batches: list[list[BatchItem]] = []
        current_batch: list[BatchItem] = []
        current_tokens = 0

        for text_id, chunks in text_chunks.items():
            for chunk_idx, (chunk_text, chunk_tokens) in enumerate(chunks):
                # Check if adding this chunk would exceed limits
                would_exceed_tokens = (
                    current_tokens + chunk_tokens
                    > self.max_embedding_tokens_per_request
                )
                would_exceed_count = len(current_batch) >= self.max_batch_size

                if current_batch and (would_exceed_tokens or would_exceed_count):
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0

                current_batch.append(
                    BatchItem(chunk_text, text_id, chunk_idx, chunk_tokens)
                )
                current_tokens += chunk_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    async def _process_batch(
        self, batch: list[BatchItem], max_retries: int = 3
    ) -> dict[str, dict[int, list[float]]]:
        """
        Process a single batch through the embeddings API with retry logic.

        Args:
            batch: List of BatchItem objects to embed
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            Maps text IDs to {chunk_index: embedding_vector} dictionaries
        """
        last_exception: Exception | None = None

        async def _call_provider() -> dict[str, dict[int, list[float]]]:
            """One provider call. Lifted out of the retry loop so
            _emit_embedding_call emits a separate event per attempt — each
            attempt is a distinct provider hit and shows up as its own line
            item in analytics."""
            result: dict[str, dict[int, list[float]]] = defaultdict(dict)
            if self.transport == "gemini":
                gemini_client = cast(genai.Client, self.client)
                response = await gemini_client.aio.models.embed_content(
                    model=self.model,
                    contents=[item.text for item in batch],
                    config={"output_dimensionality": self.vector_dimensions},
                )
                if response.embeddings:
                    for item, embedding in zip(batch, response.embeddings, strict=True):
                        if embedding.values:
                            result[item.text_id][item.chunk_index] = (
                                self._validate_embedding_dimensions(embedding.values)
                            )
            elif self.transport == "voyage":
                embeddings = await self._voyage_embeddings(
                    [item.text for item in batch],
                    input_type=self.document_input_type,
                )
                for item, embedding in zip(batch, embeddings, strict=True):
                    result[item.text_id][item.chunk_index] = embedding
            else:  # openai
                openai_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "input": [item.text for item in batch],
                }
                if self.send_dimensions:
                    openai_kwargs["dimensions"] = self.vector_dimensions
                openai_client = cast(AsyncOpenAI, self.client)
                response = await openai_client.embeddings.create(**openai_kwargs)
                for item, embedding_data in zip(batch, response.data, strict=True):
                    result[item.text_id][item.chunk_index] = (
                        self._validate_embedding_dimensions(embedding_data.embedding)
                    )
            return result

        # Token counts were computed during chunk prep; reuse them here so the
        # provider call doesn't re-encode every chunk just for the size proxy.
        batch_tokens_estimate = sum(item.token_count for item in batch)
        batch_texts = [item.text for item in batch]

        for attempt in range(max_retries):
            try:
                result = await _emit_embedding_call(
                    provider=self.transport,
                    model=self.model,
                    texts=batch_texts,
                    input_tokens_estimate=batch_tokens_estimate,
                    fn=_call_provider,
                    is_final_attempt=(attempt >= max_retries - 1),
                )
                return dict(result)

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2**attempt
                    logger.warning(
                        f"Embedding batch failed (attempt {attempt + 1}/{max_retries}), "
                        + f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.exception("Error processing batch after all retries")

        raise last_exception or RuntimeError("Batch processing failed")

    def _accumulate_embeddings(
        self, batch_results: list[dict[str, dict[int, list[float]]]]
    ) -> dict[str, list[list[float]]]:
        """
        Combine batch results into final output, preserving chunk order.

        Args:
            batch_results: List of batch results from _process_batch

        Returns:
            Maps text IDs to ordered lists of embedding vectors
        """
        all_embeddings: dict[str, dict[int, list[float]]] = defaultdict(dict)

        # Collect all embeddings by text_id and chunk_index
        for batch_result in batch_results:
            for text_id, chunk_dict in batch_result.items():
                all_embeddings[text_id].update(chunk_dict)

        # Convert to ordered lists
        return {
            text_id: [chunk_dict[i] for i in sorted(chunk_dict.keys())]
            for text_id, chunk_dict in all_embeddings.items()
        }


def _chunk_text_with_tokens(
    text: str,
    encoded_tokens: list[int],
    max_tokens: int,
    encoding: tiktoken.Encoding,
) -> list[tuple[str, int]]:
    """
    Split text into chunks that fit within token limits, with 20% overlap.

    Args:
        text: Original text to chunk
        encoded_tokens: Pre-encoded tokens for the text
        max_tokens: Maximum tokens per chunk
        encoding: Tiktoken encoding model

    Returns:
        List of (chunk_text, token_count) tuples
    """
    if len(encoded_tokens) <= max_tokens:
        return [(text, len(encoded_tokens))]

    # Use 20% overlap for better semantic continuity
    overlap_tokens = int(max_tokens * 0.2)
    step_size = max_tokens - overlap_tokens

    return [
        (
            encoding.decode(encoded_tokens[i : i + max_tokens]),
            min(max_tokens, len(encoded_tokens) - i),
        )
        for i in range(0, len(encoded_tokens), step_size)
        if i < len(encoded_tokens)  # Ensure we don't create empty chunks
    ]


class EmbeddingClient:
    """
    Singleton wrapper for the embedding client with deferred loading.

    The actual client is only initialized on first use, improving startup time
    and allowing the application to start even if API keys are not yet configured.
    """

    _instance: "_EmbeddingClient | None" = None
    _instance_signature: tuple[object, ...] | None = None
    _lock: threading.Lock = threading.Lock()
    _wrapper_instance: "EmbeddingClient | None" = None

    def __new__(cls):
        """Ensure only one instance of EmbeddingClient exists."""
        # We always return the same wrapper instance
        if cls._wrapper_instance is None:
            cls._wrapper_instance = super().__new__(cls)
        return cls._wrapper_instance

    def _get_client(self) -> _EmbeddingClient:
        """
        Get or create the underlying embedding client instance.

        Uses double-checked locking for thread-safe lazy initialization.
        """
        signature = self._get_settings_signature()
        if self._instance is None or self._instance_signature != signature:
            with self._lock:
                if self._instance is None or self._instance_signature != signature:
                    runtime_config = self._resolve_runtime_config()
                    self._instance = _EmbeddingClient(
                        runtime_config,
                        vector_dimensions=settings.EMBEDDING.VECTOR_DIMENSIONS,
                        max_input_tokens=settings.EMBEDDING.MAX_INPUT_TOKENS,
                        max_tokens_per_request=settings.EMBEDDING.MAX_TOKENS_PER_REQUEST,
                        send_dimensions=settings.EMBEDDING.resolve_send_dimensions(),
                    )
                    self._instance_signature = signature
                    logger.debug(
                        "Initialized embedding client with transport: %s model: %s",
                        runtime_config.transport,
                        runtime_config.model,
                    )

        return self._instance

    def _resolve_runtime_config(self) -> EmbeddingModelConfig:
        return resolve_embedding_model_config(settings.EMBEDDING.MODEL_CONFIG)

    def _get_settings_signature(self) -> tuple[object, ...]:
        runtime_config = self._resolve_runtime_config()
        return (
            runtime_config.transport,
            runtime_config.model,
            runtime_config.api_key,
            runtime_config.base_url,
            runtime_config.query_input_type,
            runtime_config.document_input_type,
            runtime_config.output_dtype,
            settings.EMBEDDING.VECTOR_DIMENSIONS,
            settings.EMBEDDING.MAX_INPUT_TOKENS,
            settings.EMBEDDING.MAX_TOKENS_PER_REQUEST,
            settings.EMBEDDING.resolve_send_dimensions(),
        )

    async def embed(self, query: str) -> list[float]:
        """Embed a single query string."""
        return await self._get_client().embed(query)

    async def simple_batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Simple batch embedding for a list of text strings."""
        return await self._get_client().simple_batch_embed(texts)

    async def batch_embed(
        self, id_resource_dict: dict[str, str]
    ) -> dict[str, list[list[float]]]:
        """Embed multiple texts, chunking long ones and batching API calls."""
        return await self._get_client().batch_embed(id_resource_dict)

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return self._get_client().provider

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._get_client().model

    @property
    def transport(self) -> str:
        """Get the transport name."""
        return self._get_client().transport

    @property
    def max_embedding_tokens(self) -> int:
        """Get the maximum embedding tokens."""
        return self._get_client().max_embedding_tokens

    @property
    def vector_dimensions(self) -> int:
        """Get the configured embedding dimensions."""
        return self._get_client().vector_dimensions

    @property
    def encoding(self) -> tiktoken.Encoding:
        """Get the tiktoken encoding."""
        return self._get_client().encoding


# Shared singleton embedding client instance
embedding_client = EmbeddingClient()
