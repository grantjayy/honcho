#!/usr/bin/env python3
"""Migrate Honcho pgvector embeddings to Voyage 4 Large shadow columns.

This script is intentionally operational and idempotent:
- Adds 1024-dim `embedding_voyage` columns.
- Backfills from existing text content using Voyage's first-class API.
- Builds HNSW indexes on shadow columns.
- Swaps columns in-place only when explicitly requested.

It never prints API keys.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx
import psycopg
from dotenv import load_dotenv

DEFAULT_DSN = "postgresql://postgres@127.0.0.1:5432/postgres"
DEFAULT_BASE_URL = "https://api.voyageai.com/v1"
DEFAULT_MODEL = "voyage-4-large"
DEFAULT_DIM = 1024
DEFAULT_MAX_TEXTS = 1000
DEFAULT_MAX_EST_TOKENS = 110_000


@dataclass(frozen=True)
class Row:
    id: str | int
    content: str


def estimate_tokens(text: str) -> int:
    # Conservative enough for batching without pulling tokenizer dependencies.
    return max(1, math.ceil(len(text) / 3.5))


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(format(float(v), ".8g") for v in values) + "]"


def load_api_key(env_path: Path, env_name: str) -> str:
    load_dotenv(env_path, override=False)
    value = os.getenv(env_name)
    if not value:
        raise SystemExit(f"Missing {env_name}; set it in {env_path} or environment")
    return value


def connect(dsn: str) -> psycopg.Connection[Any]:
    return psycopg.connect(dsn, autocommit=True)


def ensure_shadow_schema(conn: psycopg.Connection[Any], dim: int) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_voyage vector({dim})")
        cur.execute(f"ALTER TABLE message_embeddings ADD COLUMN IF NOT EXISTS embedding_voyage vector({dim})")


def counts(conn: psycopg.Connection[Any]) -> dict[str, int]:
    sql = """
    SELECT 'documents_total' AS key, count(*)::bigint AS value FROM documents WHERE embedding IS NOT NULL AND deleted_at IS NULL
    UNION ALL SELECT 'documents_voyage', count(*)::bigint FROM documents WHERE embedding_voyage IS NOT NULL AND deleted_at IS NULL
    UNION ALL SELECT 'documents_remaining', count(*)::bigint FROM documents WHERE embedding IS NOT NULL AND embedding_voyage IS NULL AND deleted_at IS NULL
    UNION ALL SELECT 'messages_total', count(*)::bigint FROM message_embeddings WHERE embedding IS NOT NULL
    UNION ALL SELECT 'messages_voyage', count(*)::bigint FROM message_embeddings WHERE embedding_voyage IS NOT NULL
    UNION ALL SELECT 'messages_remaining', count(*)::bigint FROM message_embeddings WHERE embedding IS NOT NULL AND embedding_voyage IS NULL
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return {str(k): int(v) for k, v in cur.fetchall()}


def fetch_rows(
    conn: psycopg.Connection[Any],
    table: str,
    limit: int,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[Row]:
    shard_clause = ""
    params: list[Any] = []
    if shard_count > 1:
        if table == "documents":
            shard_clause = "AND mod(abs(hashtext(id)), %s) = %s"
        else:
            shard_clause = "AND mod(id, %s) = %s"
        params.extend([shard_count, shard_index])

    if table == "documents":
        sql = f"""
        SELECT id, content
        FROM documents
        WHERE embedding IS NOT NULL AND embedding_voyage IS NULL AND deleted_at IS NULL
        {shard_clause}
        ORDER BY created_at ASC, id ASC
        LIMIT %s
        """
    elif table == "message_embeddings":
        sql = f"""
        SELECT id, content
        FROM message_embeddings
        WHERE embedding IS NOT NULL AND embedding_voyage IS NULL
        {shard_clause}
        ORDER BY id ASC
        LIMIT %s
        """
    else:
        raise ValueError(f"unsupported table {table!r}")
    params.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [Row(row[0], row[1] or "") for row in cur.fetchall()]


def chunk_rows(rows: list[Row], max_texts: int, max_est_tokens: int) -> Iterable[list[Row]]:
    batch: list[Row] = []
    token_sum = 0
    for row in rows:
        t = estimate_tokens(row.content)
        if batch and (len(batch) >= max_texts or token_sum + t > max_est_tokens):
            yield batch
            batch = []
            token_sum = 0
        batch.append(row)
        token_sum += t
    if batch:
        yield batch


async def embed_rows(
    client: httpx.AsyncClient,
    *,
    model: str,
    rows: list[Row],
    dim: int,
    attempt: int = 1,
) -> list[list[float]]:
    payload = {
        "model": model,
        "input": [r.content for r in rows],
        "input_type": "document",
        "output_dtype": "float",
        # Omit output_dimension intentionally: Grant asked for Voyage defaults.
    }
    try:
        response = await client.post("/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()["data"]
        ordered = sorted(data, key=lambda item: int(item["index"]))
        embeddings = [item["embedding"] for item in ordered]
        if len(embeddings) != len(rows):
            raise RuntimeError(f"embedding count mismatch: {len(embeddings)} != {len(rows)}")
        bad = [len(e) for e in embeddings if len(e) != dim]
        if bad:
            raise RuntimeError(f"embedding dimension mismatch; expected {dim}, got {bad[:5]}")
        return embeddings
    except Exception as exc:
        if len(rows) > 1 and (attempt >= 2 or isinstance(exc, httpx.HTTPStatusError)):
            mid = len(rows) // 2
            left = await embed_rows(client, model=model, rows=rows[:mid], dim=dim, attempt=1)
            right = await embed_rows(client, model=model, rows=rows[mid:], dim=dim, attempt=1)
            return left + right
        if attempt < 5:
            await asyncio.sleep(min(30.0, 2**attempt + random.random()))
            return await embed_rows(client, model=model, rows=rows, dim=dim, attempt=attempt + 1)
        raise


def update_rows(conn: psycopg.Connection[Any], table: str, rows: list[Row], embeddings: list[list[float]]) -> int:
    if not rows:
        return 0
    temp_sql = "CREATE TEMP TABLE tmp_voyage_embeddings (id text PRIMARY KEY, embedding vector(%s)) ON COMMIT DROP" % DEFAULT_DIM
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(temp_sql)
        cur.executemany(
            "INSERT INTO tmp_voyage_embeddings(id, embedding) VALUES (%s, %s::vector)",
            [(str(row.id), vector_literal(emb)) for row, emb in zip(rows, embeddings)],
        )
        if table == "documents":
            cur.execute(
                """
                UPDATE documents AS d
                SET embedding_voyage = t.embedding
                FROM tmp_voyage_embeddings AS t
                WHERE d.id = t.id AND d.embedding_voyage IS NULL
                """
            )
        else:
            cur.execute(
                """
                UPDATE message_embeddings AS m
                SET embedding_voyage = t.embedding
                FROM tmp_voyage_embeddings AS t
                WHERE m.id = t.id::bigint AND m.embedding_voyage IS NULL
                """
            )
        updated = int(cur.rowcount or 0)
    return updated


async def backfill(args: argparse.Namespace) -> None:
    api_key = load_api_key(Path(args.env_file), args.api_key_env)
    ensure_shadow_schema(connect(args.dsn), args.dim)
    conn = connect(args.dsn)
    start = time.monotonic()
    total_updated = 0
    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = httpx.Timeout(connect=20.0, read=120.0, write=120.0, pool=20.0)
    async with httpx.AsyncClient(base_url=args.base_url, headers=headers, timeout=timeout) as client:
        for table in args.tables:
            table_updated = 0
            while True:
                rows = fetch_rows(
                    conn,
                    table,
                    args.fetch_limit,
                    shard_index=args.shard_index,
                    shard_count=args.shard_count,
                )
                if not rows:
                    break
                for sub in chunk_rows(rows, args.max_texts, args.max_est_tokens):
                    embeddings = await embed_rows(client, model=args.model, rows=sub, dim=args.dim)
                    updated = update_rows(conn, table, sub, embeddings)
                    table_updated += updated
                    total_updated += updated
                    if total_updated % args.progress_every < updated or updated != len(sub):
                        c = counts(conn)
                        elapsed = time.monotonic() - start
                        print(
                            json.dumps(
                                {
                                    "event": "progress",
                                    "table": table,
                                    "updated_total": total_updated,
                                    "updated_table": table_updated,
                                    "last_batch": updated,
                                    "counts": c,
                                    "elapsed_seconds": round(elapsed, 1),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                if args.max_rows and total_updated >= args.max_rows:
                    print(json.dumps({"event": "max_rows_reached", "updated_total": total_updated}), flush=True)
                    return
    print(json.dumps({"event": "backfill_done", "updated_total": total_updated, "counts": counts(conn), "elapsed_seconds": round(time.monotonic() - start, 1)}, sort_keys=True), flush=True)


def build_indexes(conn: psycopg.Connection[Any]) -> None:
    # Must run outside an explicit transaction; connection is autocommit.
    statements = [
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_documents_embedding_voyage_hnsw ON documents USING hnsw (embedding_voyage vector_cosine_ops) WITH (m = 16, ef_construction = 64) WHERE deleted_at IS NULL AND embedding_voyage IS NOT NULL",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_message_embeddings_embedding_voyage_hnsw ON message_embeddings USING hnsw (embedding_voyage vector_cosine_ops) WHERE embedding_voyage IS NOT NULL WITH (m = 16, ef_construction = 64)",
    ]
    # Older pgvector/Postgres syntax requires WITH before WHERE.
    statements[1] = "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_message_embeddings_embedding_voyage_hnsw ON message_embeddings USING hnsw (embedding_voyage vector_cosine_ops) WITH (m = 16, ef_construction = 64) WHERE embedding_voyage IS NOT NULL"
    with conn.cursor() as cur:
        cur.execute("SET maintenance_work_mem = '64MB'")
        for stmt in statements:
            print(json.dumps({"event": "index_start", "statement": stmt.split(" ON ")[0]}), flush=True)
            cur.execute(stmt)
            print(json.dumps({"event": "index_done", "statement": stmt.split(" ON ")[0]}), flush=True)


def swap_columns(conn: psycopg.Connection[Any], require_complete: bool = True) -> None:
    c = counts(conn)
    if require_complete and (c.get("documents_remaining") or c.get("messages_remaining")):
        raise SystemExit(f"Refusing swap; shadow backfill incomplete: {c}")
    with conn.transaction(), conn.cursor() as cur:
        # Drop old active HNSW indexes; rename them so index names match active column.
        cur.execute("DROP INDEX IF EXISTS ix_documents_embedding_hnsw")
        cur.execute("DROP INDEX IF EXISTS ix_message_embeddings_embedding_hnsw")
        cur.execute("ALTER TABLE documents RENAME COLUMN embedding TO embedding_openai")
        cur.execute("ALTER TABLE message_embeddings RENAME COLUMN embedding TO embedding_openai")
        cur.execute("ALTER TABLE documents RENAME COLUMN embedding_voyage TO embedding")
        cur.execute("ALTER TABLE message_embeddings RENAME COLUMN embedding_voyage TO embedding")
        cur.execute("ALTER INDEX IF EXISTS ix_documents_embedding_voyage_hnsw RENAME TO ix_documents_embedding_hnsw")
        cur.execute("ALTER INDEX IF EXISTS ix_message_embeddings_embedding_voyage_hnsw RENAME TO ix_message_embeddings_embedding_hnsw")
    print(json.dumps({"event": "swap_done"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["ensure-schema", "counts", "backfill", "build-indexes", "swap"])
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key-env", default="VOYAGE_AI_API_KEY")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--fetch-limit", type=int, default=2500)
    parser.add_argument("--max-texts", type=int, default=DEFAULT_MAX_TEXTS)
    parser.add_argument("--max-est-tokens", type=int, default=DEFAULT_MAX_EST_TOKENS)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--tables", nargs="+", default=["documents", "message_embeddings"], choices=["documents", "message_embeddings"])
    args = parser.parse_args()
    if args.shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must satisfy 0 <= index < shard-count")

    conn = connect(args.dsn)
    if args.action == "ensure-schema":
        ensure_shadow_schema(conn, args.dim)
        print(json.dumps({"event": "schema_ready", "counts": counts(conn)}, sort_keys=True))
    elif args.action == "counts":
        print(json.dumps(counts(conn), sort_keys=True))
    elif args.action == "backfill":
        asyncio.run(backfill(args))
    elif args.action == "build-indexes":
        build_indexes(conn)
    elif args.action == "swap":
        swap_columns(conn)


if __name__ == "__main__":
    main()
