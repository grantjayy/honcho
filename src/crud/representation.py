from __future__ import annotations

import datetime
import logging
import time
from contextlib import suppress
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, exceptions, models, schemas
from src.config import settings
from src.dependencies import tracked_db
from src.dreamer.dream_scheduler import check_and_schedule_dream
from src.embedding_client import embedding_client
from src.schemas import ResolvedConfiguration
from src.telemetry.events import EmbeddingCallPurpose
from src.telemetry.logging import accumulate_metric
from src.utils.formatting import format_datetime_utc
from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    Representation,
)
from src.utils.types import embedding_call_purpose

logger = logging.getLogger(__name__)


def _representation_counts(
    *,
    total: int,
    include_semantic_query: str | None,
    semantic_search_top_k: int | None,
    include_most_derived: bool,
) -> tuple[int, int, int]:
    """Return semantic, most-derived, and recent observation budgets."""
    semantic_observations = (
        min(
            max(
                0,
                semantic_search_top_k
                if semantic_search_top_k is not None
                else total // 3,
            ),
            total,
        )
        if include_semantic_query
        else 0
    )

    if include_semantic_query and include_most_derived:
        top_observations = min(max(0, total // 3), total - semantic_observations)
    elif include_most_derived:
        top_observations = min(max(0, total // 2), total - semantic_observations)
    else:
        top_observations = 0

    recent_observations = total - semantic_observations - top_observations
    return semantic_observations, top_observations, recent_observations


def _observation_text(obs: ExplicitObservation | DeductiveObservation) -> str:
    """Return the canonical text payload for an explicit or deductive observation."""
    return obs.conclusion if isinstance(obs, DeductiveObservation) else obs.content


def _normalized_observation(
    obs: ExplicitObservation | DeductiveObservation,
) -> ExplicitObservation | DeductiveObservation:
    """Return an observation with its persisted/embed text normalized."""
    text = _observation_text(obs).strip()
    if isinstance(obs, DeductiveObservation):
        return obs.model_copy(update={"conclusion": text})
    return obs.model_copy(update={"content": text})


class RepresentationManager:
    """Unified manager for representation and document queries."""

    def __init__(
        self,
        workspace_name: str,
        *,
        observer: str,
        observed: str,
    ) -> None:
        self.workspace_name: str = workspace_name
        self.observer: str = observer
        self.observed: str = observed

    async def save_representation(
        self,
        representation: Representation,
        message_ids: list[int],
        session_name: str,
        message_created_at: datetime.datetime,
        message_level_configuration: ResolvedConfiguration,
    ) -> int:
        """
        Save Representation objects to the collection as a set of documents.

        Args:
            representation: Representation object
            message_ids: Message ID range to link with observations
            session_name: Session name to link with existing summary context
            message_created_at: Timestamp when the message was created

        Returns:
            The number of *new documents saved*
        """

        new_documents = 0

        if not representation.deductive and not representation.explicit:
            logger.debug("No observations to save")
            return new_documents

        all_observations = [
            _normalized_observation(obs)
            for obs in representation.deductive + representation.explicit
            if _observation_text(obs).strip()
        ]
        if not all_observations:
            logger.debug("No non-empty observations to save")
            return new_documents

        # Batch embed all observations
        batch_embed_start = time.perf_counter()

        observation_texts = [_observation_text(obs) for obs in all_observations]
        try:
            with embedding_call_purpose(
                EmbeddingCallPurpose.CREATE_OBSERVATIONS.value,
                workspace_name=self.workspace_name,
                parent_category="representation",
            ):
                embeddings = await embedding_client.simple_batch_embed(
                    observation_texts
                )
        except ValueError as e:
            raise exceptions.ValidationException(
                "Observation content exceeds maximum token limit of "
                + f"{settings.EMBEDDING.MAX_INPUT_TOKENS}."
            ) from e

        batch_embed_duration = (time.perf_counter() - batch_embed_start) * 1000
        accumulate_metric(
            f"deriver_{message_ids[-1]}_{self.observer}",
            "embed_new_observations",
            batch_embed_duration,
            "ms",
        )

        # Batch create document objects
        create_document_start = time.perf_counter()
        async with tracked_db("representation_manager.save_representation") as db:
            new_documents = await self._save_representation_internal(
                db,
                all_observations,
                embeddings,
                message_ids,
                session_name,
                message_created_at,
                message_level_configuration,
            )

        create_document_duration = (time.perf_counter() - create_document_start) * 1000
        accumulate_metric(
            f"deriver_{message_ids[-1]}_{self.observer}",
            "save_new_observations",
            create_document_duration,
            "ms",
        )

        return new_documents

    async def _save_representation_internal(
        self,
        db: AsyncSession,
        all_observations: list[ExplicitObservation | DeductiveObservation],
        embeddings: list[list[float]],
        message_ids: list[int],
        session_name: str,
        message_created_at: datetime.datetime,
        message_level_configuration: ResolvedConfiguration,
    ) -> int:
        # get_or_create_collection already handles IntegrityError with rollback and a retry
        collection = await crud.get_or_create_collection(
            db,
            self.workspace_name,
            observer=self.observer,
            observed=self.observed,
        )

        # Prepare all documents for bulk creation
        documents_to_create: list[schemas.DocumentCreate] = []
        for obs, embedding in zip(all_observations, embeddings, strict=True):
            # NOTE: will add additional levels of reasoning in the future
            if isinstance(obs, DeductiveObservation):
                obs_level = "deductive"
                obs_content = obs.conclusion
                obs_premises = obs.premises
            else:
                obs_level = "explicit"
                obs_content = obs.content
                obs_premises = None

            metadata: schemas.DocumentMetadata = schemas.DocumentMetadata(
                message_ids=message_ids,
                premises=obs_premises,
                message_created_at=format_datetime_utc(message_created_at),
            )

            documents_to_create.append(
                schemas.DocumentCreate(
                    content=obs_content,
                    session_name=session_name,
                    level=obs_level,
                    metadata=metadata,
                    embedding=embedding,
                )
            )

        # Use bulk creation with optional duplicate detection
        accepted_documents = await crud.create_documents(
            db,
            documents_to_create,
            self.workspace_name,
            observer=self.observer,
            observed=self.observed,
            deduplicate=settings.DERIVER.DEDUPLICATE,
        )

        if message_level_configuration.dream.enabled:
            try:
                await check_and_schedule_dream(db, collection)
            except Exception as e:
                logger.warning(f"Failed to check dream scheduling: {e}")

        return len(accepted_documents)

    async def get_working_representation(
        self,
        *,
        db: AsyncSession | None = None,
        session_name: str | None = None,
        include_semantic_query: str | None = None,
        embedding: list[float] | None = None,
        semantic_search_top_k: int | None = None,
        semantic_search_max_distance: float | None = None,
        semantic_search_overfetch_k: int | None = None,
        semantic_rerank: bool = False,
        include_most_derived: bool = False,
        max_observations: int = settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
        parent_category: str | None = None,
        embedding_purpose: EmbeddingCallPurpose = EmbeddingCallPurpose.SEARCH_MEMORY,
    ) -> Representation:
        """
        Get working representation with flexible query options.

        Args:
            db: Optional database session. If provided, uses it directly;
                otherwise creates a new session via tracked_db.
            session_name: Optional session to filter by
            include_semantic_query: Query for semantic search
            embedding: Pre-computed embedding for the semantic query.
            semantic_search_top_k: Number of semantic results
            semantic_search_max_distance: Maximum distance for semantic search
            semantic_search_overfetch_k: Optional semantic vector candidate count before reranking
            semantic_rerank: Whether to rerank semantic vector candidates
            include_most_derived: Include most derived observations
            max_observations: Maximum total observations to return
            parent_category: Optional workflow attribution forwarded to the
                fallback embedding call when the caller didn't pre-compute
                an embedding (or pre-compute failed).
            embedding_purpose: Embedding call_purpose tag to use on the
                fallback embed when no pre-computed embedding was supplied.
                Defaults to SEARCH_MEMORY; callers whose route-level
                precompute uses a more specific purpose (e.g.
                SESSION_CONTEXT_SEARCH) should pass that here so the
                fallback path lands in the same analytics bucket.

        Returns:
            Representation combining various query strategies
        """
        if include_semantic_query and embedding is None:
            # Best-effort precompute when caller didn't supply one (or their
            # precompute was suppressed). The purpose is parameterized so
            # this fallback shows up in the same telemetry bucket as the
            # successful path — see embedding_purpose docstring above.
            with (
                suppress(Exception),
                embedding_call_purpose(
                    embedding_purpose.value,
                    workspace_name=self.workspace_name,
                    parent_category=parent_category,
                ),
            ):
                embedding = await embedding_client.embed(include_semantic_query)

        if db is not None:
            return await self._get_working_representation_internal(
                db,
                session_name=session_name,
                include_semantic_query=include_semantic_query,
                embedding=embedding,
                semantic_search_top_k=semantic_search_top_k,
                semantic_search_max_distance=semantic_search_max_distance,
                semantic_search_overfetch_k=semantic_search_overfetch_k,
                semantic_rerank=semantic_rerank,
                include_most_derived=include_most_derived,
                max_observations=max_observations,
            )

        precomputed_semantic_docs: list[models.Document] | None = None
        if include_semantic_query and semantic_rerank:
            semantic_observations, _, _ = _representation_counts(
                total=max_observations,
                include_semantic_query=include_semantic_query,
                semantic_search_top_k=semantic_search_top_k,
                include_most_derived=include_most_derived,
            )
            precomputed_semantic_docs = list(
                await crud.query_documents(
                    None,
                    workspace_name=self.workspace_name,
                    observer=self.observer,
                    observed=self.observed,
                    query=include_semantic_query,
                    max_distance=semantic_search_max_distance,
                    top_k=semantic_observations,
                    embedding=embedding,
                    overfetch_k=semantic_search_overfetch_k,
                    rerank=True,
                )
            )

        async with tracked_db(
            "representation_manager.get_working_representation"
        ) as new_db:
            return await self._get_working_representation_internal(
                new_db,
                session_name=session_name,
                include_semantic_query=include_semantic_query,
                embedding=embedding,
                semantic_search_top_k=semantic_search_top_k,
                semantic_search_max_distance=semantic_search_max_distance,
                semantic_search_overfetch_k=semantic_search_overfetch_k,
                semantic_rerank=False
                if precomputed_semantic_docs is not None
                else semantic_rerank,
                include_most_derived=include_most_derived,
                max_observations=max_observations,
                precomputed_semantic_docs=precomputed_semantic_docs,
            )

    # Private helper methods

    async def _get_working_representation_internal(
        self,
        db: AsyncSession,
        *,
        session_name: str | None = None,
        include_semantic_query: str | None = None,
        embedding: list[float] | None = None,
        semantic_search_top_k: int | None = None,
        semantic_search_max_distance: float | None = None,
        semantic_search_overfetch_k: int | None = None,
        semantic_rerank: bool = False,
        include_most_derived: bool = False,
        max_observations: int = settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
        precomputed_semantic_docs: list[models.Document] | None = None,
    ) -> Representation:
        """Internal implementation of get_working_representation."""
        total = max_observations
        semantic_observations, top_observations, recent_observations = (
            _representation_counts(
                total=total,
                include_semantic_query=include_semantic_query,
                semantic_search_top_k=semantic_search_top_k,
                include_most_derived=include_most_derived,
            )
        )

        representation = Representation()

        # Get semantic observations if requested
        semantic_docs: list[models.Document] = []
        if precomputed_semantic_docs is not None:
            semantic_docs = precomputed_semantic_docs
        elif include_semantic_query:
            semantic_docs = await self._query_documents_semantic(
                db,
                query=include_semantic_query,
                top_k=semantic_observations,
                max_distance=semantic_search_max_distance,
                embedding=embedding,
                overfetch_k=semantic_search_overfetch_k,
                rerank=semantic_rerank,
            )
        representation.merge_representation(Representation.from_documents(semantic_docs))

        # Get most derived observations if requested
        if include_most_derived:
            derived_docs = await self._query_documents_most_derived(
                db, top_k=top_observations
            )
            representation.merge_representation(
                Representation.from_documents(derived_docs)
            )

        # Get recent observations
        recent_docs = await self._query_documents_recent(
            db, top_k=recent_observations, session_name=session_name
        )

        representation.merge_representation(Representation.from_documents(recent_docs))

        return representation

    async def _query_documents_semantic(
        self,
        db: AsyncSession,
        query: str,
        top_k: int,
        max_distance: float | None = None,
        level: str | None = None,
        embedding: list[float] | None = None,
        overfetch_k: int | None = None,
        rerank: bool = False,
    ) -> list[models.Document]:
        """Query documents by semantic similarity."""
        try:
            if level:
                return await self._query_documents_for_level(
                    db,
                    query,
                    level,
                    top_k,
                    max_distance,
                    embedding=embedding,
                    overfetch_k=overfetch_k,
                    rerank=rerank,
                )
            else:
                documents = await crud.query_documents(
                    db,
                    workspace_name=self.workspace_name,
                    observer=self.observer,
                    observed=self.observed,
                    query=query,
                    max_distance=max_distance,
                    top_k=top_k,
                    embedding=embedding,
                    overfetch_k=overfetch_k,
                    rerank=rerank,
                )
                db.expunge_all()
                return list(documents)

        except Exception as e:
            logger.error(f"Error getting relevant observations: {e}")
            return []

    async def _query_documents_recent(
        self, db: AsyncSession, top_k: int, session_name: str | None = None
    ) -> list[models.Document]:
        """Query most recent documents."""
        stmt = (
            select(models.Document)
            .limit(top_k)
            .where(
                models.Document.workspace_name == self.workspace_name,
                models.Document.observer == self.observer,
                models.Document.observed == self.observed,
                models.Document.deleted_at.is_(None),
                *(
                    [models.Document.session_name == session_name]
                    if session_name is not None
                    else []
                ),
            )
            .order_by(models.Document.created_at.desc())
        )

        result = await db.execute(stmt)
        documents = result.scalars().all()
        db.expunge_all()
        return list(documents)

    async def _query_documents_most_derived(
        self, db: AsyncSession, top_k: int
    ) -> list[models.Document]:
        """Query most derived documents."""
        stmt = (
            select(models.Document)
            .limit(top_k)
            .where(
                models.Document.workspace_name == self.workspace_name,
                models.Document.observer == self.observer,
                models.Document.observed == self.observed,
                models.Document.deleted_at.is_(None),
            )
            .order_by(models.Document.times_derived.desc())
        )

        result = await db.execute(stmt)
        documents = result.scalars().all()
        db.expunge_all()
        return list(documents)

    async def _get_observations_internal(
        self,
        db: AsyncSession,
        query: str,
        top_k: int,
        max_distance: float,
        level: str | None,
    ) -> list[models.Document]:
        """Internal method that does the actual observation retrieval."""
        return await self._query_documents_semantic(
            db, query, top_k, max_distance, level
        )

    async def _query_documents_for_level(
        self,
        db: AsyncSession,
        query: str,
        level: str,
        count: int,
        max_distance: float | None = None,
        embedding: list[float] | None = None,
        overfetch_k: int | None = None,
        rerank: bool = False,
    ) -> list[models.Document]:
        """Query documents for a specific level."""
        documents = await crud.query_documents(
            db,
            workspace_name=self.workspace_name,
            observer=self.observer,
            observed=self.observed,
            query=query,
            max_distance=max_distance,
            top_k=count,
            filters=self._build_filter_conditions(level),
            embedding=embedding,
            overfetch_k=overfetch_k,
            rerank=rerank,
        )

        # Sort by creation time
        docs_sorted: list[models.Document] = sorted(
            list(documents), key=lambda x: x.created_at, reverse=True
        )
        return docs_sorted

    def _build_filter_conditions(
        self,
        level: str | None = None,
    ) -> dict[str, Any]:
        """
        Build filter conditions for document queries.

        Returns a flat dict of key-value pairs for vector store filtering.
        """
        filters: dict[str, Any] = {}

        if level:
            filters["level"] = level

        return filters


# Module-level functions for backward compatibility and convenience


async def get_working_representation(
    workspace_name: str,
    *,
    db: AsyncSession | None = None,
    observer: str,
    observed: str,
    session_name: str | None = None,
    include_semantic_query: str | None = None,
    embedding: list[float] | None = None,
    semantic_search_top_k: int | None = None,
    semantic_search_max_distance: float | None = None,
    semantic_search_overfetch_k: int | None = None,
    semantic_rerank: bool = False,
    include_most_derived: bool = False,
    max_observations: int = settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS,
    parent_category: str | None = None,
    embedding_purpose: EmbeddingCallPurpose = EmbeddingCallPurpose.SEARCH_MEMORY,
) -> Representation:
    """
    Get raw working representation data from the relevant document collection.

    This is a convenience function that creates a RepresentationManager and calls
    get_working_representation on it.

    Args:
        db: Optional database session. If provided, uses it directly;
            otherwise creates a new session via tracked_db.
        embedding: Pre-computed embedding for the semantic query.
        parent_category: Workflow attribution forwarded to the fallback
            embedding call when no pre-computed embedding was supplied.
        embedding_purpose: Embedding call_purpose for the fallback embed;
            callers should match it to whatever purpose their route-level
            precompute used so failure/retry paths stay in the same bucket.
    """
    manager = RepresentationManager(
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
    )
    return await manager.get_working_representation(
        db=db,
        session_name=session_name,
        include_semantic_query=include_semantic_query,
        embedding=embedding,
        semantic_search_top_k=semantic_search_top_k,
        semantic_search_max_distance=semantic_search_max_distance,
        semantic_search_overfetch_k=semantic_search_overfetch_k,
        semantic_rerank=semantic_rerank,
        include_most_derived=include_most_derived,
        max_observations=max_observations,
        parent_category=parent_category,
        embedding_purpose=embedding_purpose,
    )
