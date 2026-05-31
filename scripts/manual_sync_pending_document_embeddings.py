#!/usr/bin/env python3
"""One-shot repair for Honcho pgvector pending document embeddings.

Honcho's reconciler currently skips document embedding sync when VECTOR_STORE.TYPE is
pgvector because `get_external_vector_store()` is None and the cycle only cleans up
soft-deleted rows. This script embeds active pending documents directly into the
Postgres `documents.embedding` column and marks them synced.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from sqlalchemy import and_, func, select, update

from src import models
from src.dependencies import tracked_db
from src.embedding_client import embedding_client
from src.telemetry.events import EmbeddingCallPurpose
from src.utils.types import embedding_call_purpose


async def fetch_batch(limit: int) -> list[tuple[str, str]]:
    async with tracked_db("manual_pending_document_embedding_fetch") as db:
        result = await db.execute(
            select(models.Document.id, models.Document.content)
            .where(
                and_(
                    models.Document.workspace_name == "hermes",
                    models.Document.deleted_at.is_(None),
                    models.Document.sync_state == "pending",
                    models.Document.embedding.is_(None),
                )
            )
            .order_by(models.Document.created_at.asc(), models.Document.id.asc())
            .limit(limit)
        )
        return [(str(doc_id), str(content)) for doc_id, content in result.all()]


async def mark_present_embeddings_synced() -> int:
    async with tracked_db("manual_pending_document_embedding_present_sync") as db:
        result = await db.execute(
            update(models.Document)
            .where(
                and_(
                    models.Document.workspace_name == "hermes",
                    models.Document.deleted_at.is_(None),
                    models.Document.sync_state == "pending",
                    models.Document.embedding.is_not(None),
                )
            )
            .values(sync_state="synced", last_sync_at=func.now(), sync_attempts=0)
        )
        await db.commit()
        return int(result.rowcount or 0)


async def update_batch(rows: list[tuple[str, str]], embeddings: list[list[float]]) -> int:
    async with tracked_db("manual_pending_document_embedding_update") as db:
        updated = 0
        for (doc_id, _content), emb in zip(rows, embeddings, strict=True):
            result = await db.execute(
                update(models.Document)
                .where(models.Document.id == doc_id)
                .values(
                    embedding=emb,
                    sync_state="synced",
                    last_sync_at=func.now(),
                    sync_attempts=0,
                    internal_metadata=models.Document.internal_metadata.op("||")(
                        {
                            "manual_embedding_sync_batch": "memory-quality-pass2-20260530",
                        }
                    ),
                )
            )
            updated += int(result.rowcount or 0)
        await db.commit()
        return updated


async def remaining_counts() -> tuple[int, int, int]:
    async with tracked_db("manual_pending_document_embedding_counts") as db:
        pending = await db.scalar(
            select(func.count()).select_from(models.Document).where(
                and_(
                    models.Document.workspace_name == "hermes",
                    models.Document.deleted_at.is_(None),
                    models.Document.sync_state == "pending",
                )
            )
        )
        pending_null = await db.scalar(
            select(func.count()).select_from(models.Document).where(
                and_(
                    models.Document.workspace_name == "hermes",
                    models.Document.deleted_at.is_(None),
                    models.Document.sync_state == "pending",
                    models.Document.embedding.is_(None),
                )
            )
        )
        failed = await db.scalar(
            select(func.count()).select_from(models.Document).where(
                and_(
                    models.Document.workspace_name == "hermes",
                    models.Document.deleted_at.is_(None),
                    models.Document.sync_state == "failed",
                )
            )
        )
        return int(pending or 0), int(pending_null or 0), int(failed or 0)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=100000)
    args = parser.parse_args()

    start = time.monotonic()
    present_synced = await mark_present_embeddings_synced()
    total = 0
    batches = 0
    print(f"marked_present_embeddings_synced={present_synced}", flush=True)

    while batches < args.max_batches:
        rows = await fetch_batch(args.batch_size)
        if not rows:
            break
        batches += 1
        contents = [content for _doc_id, content in rows]
        try:
            with embedding_call_purpose(
                EmbeddingCallPurpose.VECTOR_SYNC.value,
                parent_category="manual_reconciliation",
            ):
                embeddings = await embedding_client.simple_batch_embed(contents)
            if len(embeddings) != len(rows):
                raise RuntimeError(f"embedding count mismatch: got {len(embeddings)} for {len(rows)} rows")
            updated = await update_batch(rows, embeddings)
            total += updated
            if batches % 10 == 0 or updated != len(rows):
                pending, pending_null, failed = await remaining_counts()
                print(
                    f"batch={batches} updated={updated} total={total} pending={pending} pending_null={pending_null} failed={failed}",
                    flush=True,
                )
        except Exception as exc:
            print(f"ERROR batch={batches} first_doc={rows[0][0]} size={len(rows)} error={type(exc).__name__}: {exc}", flush=True)
            raise

    pending, pending_null, failed = await remaining_counts()
    elapsed = time.monotonic() - start
    print(
        f"done batches={batches} embedded_updated={total} marked_present={present_synced} pending={pending} pending_null={pending_null} failed={failed} elapsed_seconds={elapsed:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
