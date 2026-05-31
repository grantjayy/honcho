#!/usr/bin/env python3
"""Backfill Hermes state.db sessions into a local Honcho v3 server.

Design notes:
- Uses Honcho's documented historical-import path: workspace + peers + sessions +
  batches of messages with original created_at timestamps.
- Keeps a local idempotence ledger so reruns do not duplicate messages.
- Imports only user/assistant turns; tool/system/developer internals are skipped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
MAX_MESSAGE_CHARS = 24000  # under Honcho/Hermes 25k cap, leave headroom
BATCH_SIZE = 100


def now_ts() -> float:
    return time.time()


def iso_from_ts(ts: float) -> str:
    return dt.datetime.fromtimestamp(float(ts), tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def safe_id(value: str, max_len: int = 160) -> str:
    value = re.sub(r"[^A-Za-z0-9_.:-]+", "-", value.strip())
    value = re.sub(r"-+", "-", value).strip("-._:")
    return (value or "session")[:max_len]


def strip_memory_context(text: str) -> str:
    if not text:
        return ""
    # Avoid recursively ingesting Honcho context blocks from future captures.
    text = re.sub(r"<memory-context>.*?</memory-context>", "", text, flags=re.S)
    text = re.sub(r"\[CONTEXT COMPACTION — REFERENCE ONLY\].*?(?=\n\s*\[Grant Jordan\]|\Z)", "", text, flags=re.S)
    return text.strip()


def split_content(text: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    text = strip_memory_context(text)
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + limit)
        # Prefer paragraph or line boundary in the last 20% of the chunk.
        boundary = max(text.rfind("\n\n", start, end), text.rfind("\n", start, end))
        if boundary > start + int(limit * 0.8):
            end = boundary
        part = text[start:end].strip()
        if part:
            suffix = f"\n\n[continued chunk {len(chunks)+1}]" if chunks else ""
            chunks.append(part + suffix)
        start = end
    return chunks


class HonchoClient:
    def __init__(self, base_url: str, workspace: str, dry_run: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.workspace = workspace
        self.dry_run = dry_run

    def request(self, method: str, path: str, body: Any | None = None) -> Any:
        url = self.base_url + path
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.dry_run:
            return {"dry_run": True}
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {url} -> HTTP {e.code}: {detail}") from e

    def ensure_workspace(self) -> None:
        self.request("POST", "/v3/workspaces", {"id": self.workspace})

    def ensure_peer(self, peer: str) -> None:
        self.request("POST", f"/v3/workspaces/{self.workspace}/peers", {"id": peer})

    def ensure_session(self, session_id: str, metadata: dict[str, Any]) -> None:
        self.request("POST", f"/v3/workspaces/{self.workspace}/sessions", {"id": session_id, "metadata": metadata})

    def add_peers(self, session_id: str, user_peer: str, ai_peer: str) -> None:
        cfg = {
            user_peer: {"observe_me": True, "observe_others": True},
            ai_peer: {"observe_me": True, "observe_others": True},
        }
        self.request("POST", f"/v3/workspaces/{self.workspace}/sessions/{session_id}/peers", cfg)

    def add_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        if not messages:
            return
        for i in range(0, len(messages), BATCH_SIZE):
            self.request(
                "POST",
                f"/v3/workspaces/{self.workspace}/sessions/{session_id}/messages",
                {"messages": messages[i : i + BATCH_SIZE]},
            )


class Ledger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path)
        self.con.execute(
            "create table if not exists imported_messages (hermes_message_id integer primary key, imported_at real not null)"
        )
        self.con.commit()

    def has(self, message_id: int) -> bool:
        return self.con.execute("select 1 from imported_messages where hermes_message_id=?", (message_id,)).fetchone() is not None

    def mark_many(self, ids: list[int]) -> None:
        self.con.executemany(
            "insert or ignore into imported_messages(hermes_message_id, imported_at) values (?, ?)",
            [(i, now_ts()) for i in ids],
        )
        self.con.commit()


def load_sessions(state_db: Path, since_ts: float, limit_sessions: int | None = None) -> list[dict[str, Any]]:
    con = sqlite3.connect(state_db)
    con.row_factory = sqlite3.Row
    sessions = con.execute(
        """
        select s.*
        from sessions s
        where exists (
          select 1 from messages m
          where m.session_id=s.id and m.timestamp >= ? and m.role in ('user','assistant') and coalesce(m.content,'') <> ''
        )
        order by s.started_at asc
        """,
        (since_ts,),
    ).fetchall()
    if limit_sessions is not None:
        sessions = sessions[:limit_sessions]
    result: list[dict[str, Any]] = []
    for s in sessions:
        msgs = con.execute(
            """
            select id, role, content, timestamp, token_count, platform_message_id
            from messages
            where session_id=? and timestamp >= ? and role in ('user','assistant') and coalesce(content,'') <> ''
            order by timestamp asc, id asc
            """,
            (s["id"], since_ts),
        ).fetchall()
        if msgs:
            result.append({"session": dict(s), "messages": [dict(m) for m in msgs]})
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-db", default=str(Path.home()/".hermes/state.db"))
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--workspace", default="hermes")
    ap.add_argument("--user-peer", default="grant")
    ap.add_argument("--ai-peer", default="hermes")
    ap.add_argument("--hours", type=float, default=48)
    ap.add_argument("--limit-sessions", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ledger", default=str(Path.home()/".hermes/services/honcho/backfill-ledger.sqlite3"))
    args = ap.parse_args()

    since_ts = now_ts() - args.hours * 3600
    sessions = load_sessions(Path(args.state_db), since_ts, args.limit_sessions)
    ledger = Ledger(Path(args.ledger))
    client = HonchoClient(args.base_url, args.workspace, args.dry_run)

    total_source = sum(len(s["messages"]) for s in sessions)
    total_imported = 0
    total_chunks = 0
    skipped = 0
    print(f"sessions={len(sessions)} source_messages={total_source} since={iso_from_ts(since_ts)} dry_run={args.dry_run}")

    client.ensure_workspace()
    client.ensure_peer(args.user_peer)
    client.ensure_peer(args.ai_peer)

    for item in sessions:
        s = item["session"]
        sid = safe_id(f"hermes-{s['id']}", 200)
        meta = {
            "import_source": "hermes_state_db",
            "source_session_id": s["id"],
            "source": s.get("source"),
            "title": s.get("title"),
            "started_at": iso_from_ts(s["started_at"]),
        }
        client.ensure_session(sid, meta)
        client.add_peers(sid, args.user_peer, args.ai_peer)
        batch: list[dict[str, Any]] = []
        ids: list[int] = []
        for m in item["messages"]:
            mid = int(m["id"])
            if ledger.has(mid):
                skipped += 1
                continue
            peer = args.user_peer if m["role"] == "user" else args.ai_peer
            parts = split_content(m["content"] or "")
            for idx, part in enumerate(parts):
                batch.append({
                    "peer_id": peer,
                    "content": part,
                    "created_at": iso_from_ts(m["timestamp"] + idx * 0.001),
                    "metadata": {
                        "import_source": "hermes_state_db",
                        "source_session_id": s["id"],
                        "source_message_id": mid,
                        "source_role": m["role"],
                        "chunk_index": idx,
                        "chunk_count": len(parts),
                        "platform_message_id": m.get("platform_message_id"),
                    },
                })
                total_chunks += 1
            ids.append(mid)
        client.add_messages(sid, batch)
        if not args.dry_run:
            ledger.mark_many(ids)
        total_imported += len(ids)
        print(f"session {sid}: imported_messages={len(ids)} chunks={len(batch)}")

    print(f"done imported_messages={total_imported} chunks={total_chunks} skipped_already_imported={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
