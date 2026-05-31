# Honcho Focused Peer Context + Overfetch/Rerank Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Make self-hosted Honcho peer context return high-quality focused observations when a caller provides `search_query`, by combining semantic-only defaults with vector overfetch and Voyage reranking.

**Architecture:** Patch Honcho server-side retrieval, not Hermes and not Alloy. Query-scoped peer context (`peer.context(search_query=...)`) becomes a focused retrieval path: no recent/frequent filler by default, overfetch vector candidates from existing pgvector/document search, rerank them with Voyage, and fall back to current vector ordering if rerank is unavailable or fails. Broad no-query peer context remains broad/profile-oriented.

**Tech Stack:** FastAPI, Honcho Python service, pytest, existing Honcho CRUD representation pipeline, `httpx` for Voyage rerank API.

---

## Current Context

Grant confirmed this Honcho instance is self-hosted and only serves this Hermes agent. That means we can make properly scoped server-side changes without optimizing for arbitrary public Honcho API compatibility.

The known retrieval-quality failure has two layers:

1. **Context assembly noise:** `src/routers/peers.py::get_peer_context()` accepts `search_query` but defaults `include_most_frequent=True` and `max_conclusions=None`, so `src/crud/representation.py::_get_working_representation_internal()` blends semantic observations with most-derived and recent observations.
2. **Ranking ceiling:** `src/crud/document.py::_query_documents_pgvector()` currently ranks observations with pure vector similarity:

   ```python
   ORDER BY Document.embedding.cosine_distance(query_embedding)
   LIMIT top_k
   ```

   There is no overfetch, reranker, MMR, hybrid retrieval, or query decomposition on the observation/document path.

The chosen patch is now: **focused defaults + overfetch/rerank**.

---

## Desired Behavior

### Broad peer context stays broad

Calls like this should remain broad/profile-oriented:

```python
peer.context()
peer.context(target="grant")
```

Expected behavior:

- `include_most_frequent` defaults to `True`
- `max_conclusions` defaults to `settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS`
- no rerank is needed unless a semantic query is present
- existing broad representation/card behavior remains intact

### Query-scoped peer context becomes focused and reranked

Calls like this should become focused by default:

```python
peer.context(search_query="honcho retrieval quality")
peer.context(target="grant", search_query="honcho retrieval quality")
```

Expected effective defaults:

```python
include_most_frequent = False
search_top_k = 12
max_conclusions = 12
semantic_overfetch_k = 75
rerank_enabled = True
rerank_model = "rerank-2.5-lite"
```

Retrieval flow:

1. Embed the search query as today.
2. Fetch up to `semantic_overfetch_k` candidates by vector similarity from existing active documents.
3. Rerank those candidates against the original user query with Voyage.
4. Return the top `search_top_k` / `max_conclusions` observations.
5. If rerank is unavailable, times out, or errors, fall back to vector order and still return focused semantic-only results.

### Explicit opt-in to old blended behavior remains possible

Calls like this should still work:

```python
peer.context(
    target="grant",
    search_query="honcho retrieval quality",
    include_most_frequent=True,
    max_conclusions=100,
)
```

Expected behavior:

- caller explicitly opted into blending/frequent observations
- server honors it
- overfetch/rerank should apply only to the semantic slice, not to most-derived/recent filler

---

## Non-Goals

- Do not add Alloy.
- Do not migrate embeddings.
- Do not implement hybrid BM25/RRF in this patch.
- Do not implement MMR/cluster expansion in this patch.
- Do not prune the database.
- Do not change Deriver prompts.
- Do not restart Hermes gateway or Honcho services without Grant's explicit approval.
- Do not touch unrelated pending changes in the Honcho repo (`Dockerfile`, `src/llm/backends/openai.py`, `uv.lock`, scripts/backfill files, etc.).

---

## Design Decisions

### Keep the first rerank patch simple

Use a single rerank call after vector overfetch:

```text
vector top 75 -> Voyage rerank -> top 12
```

Do **not** implement the more complex two-stage pattern yet:

```text
coarse rerank -> cluster expansion -> fine rerank
```

That pattern may be valuable later, but the first patch should establish a reliable focused retrieval baseline.

### Rerank only focused semantic retrieval

Rerank is most valuable when `search_query` is present. Broad profile context should not pay rerank latency/cost, and broad context is not trying to answer a precise current-turn query.

### Safe fallback is required

Rerank must be opportunistic:

- missing API key -> fallback to vector top-k
- HTTP error -> fallback
- timeout -> fallback
- malformed response -> fallback
- empty rerank response -> fallback

The user-facing context should never become empty solely because rerank failed.

### Avoid holding DB sessions during external calls

Honcho’s repo guidance says: **Never hold a DB session during external calls**.

Therefore, implementation should:

1. perform DB vector candidate fetch
2. expunge/detach or convert candidates to lightweight data
3. call Voyage rerank outside any long-lived DB session / without holding a DB connection during HTTP
4. return already-fetched document objects in reranked order, or fetch by IDs after rerank if needed

The simplest initial path is to fetch document objects, close/expunge the session as current code does, then rerank the detached docs in memory.

---

## Files Likely to Change

Primary:

- `src/routers/peers.py`
- `src/crud/representation.py`
- `src/crud/document.py`
- `tests/routes/test_peers.py`
- `tests/integration/test_representation.py` or a new focused unit test file for representation/document retrieval

Likely new file:

- `src/rerank_client.py` or `src/utils/rerank.py`

Possible config file:

- `src/config.py`

Possible SDK tests:

- `tests/sdk/test_peer.py`

Do **not** modify Hermes plugin code for this patch unless tests prove the SDK cannot express the new server behavior.

---

## Proposed Constants / Config

Because this self-hosted Honcho serves only Hermes, hardcoded defaults are acceptable for the first patch. Still, keep constants centralized so they are easy to tune.

Suggested constants near the retrieval implementation or in config:

```python
FOCUSED_CONTEXT_TOP_K = 12
FOCUSED_CONTEXT_MAX_CONCLUSIONS = 12
FOCUSED_CONTEXT_OVERFETCH_K = 75
RERANK_MODEL = "rerank-2.5-lite"
RERANK_TIMEOUT_SECONDS = 3.0
```

Rerank API key resolution:

- Prefer a new env var: `VOYAGE_API_KEY`
- Optionally allow `RERANK_API_KEY` as a generic alias
- Do not expose key values in logs or tool output

If adding config settings is straightforward, prefer:

```toml
[retrieval]
focused_context_top_k = 12
focused_context_max_conclusions = 12
focused_context_overfetch_k = 75
rerank_enabled = true
rerank_model = "rerank-2.5-lite"
rerank_timeout_seconds = 3.0
```

But do not let config work balloon the patch. A small constant/env implementation is acceptable because this deployment is private/self-hosted.

---

## Implementation Plan

### Task 1: Add RED route tests for focused peer context defaults

**Objective:** Prove `GET /peers/{peer_id}/context` resolves semantic-only focused defaults when `search_query` is present and the caller omits retrieval knobs.

**Files:**

- Modify: `tests/routes/test_peers.py`

**Steps:**

1. Find existing tests around:
   - `test_get_peer_context...`
   - `test_get_peer_representation_with_search_query`
   - `test_get_peer_representation_with_include_most_frequent`
2. Add a test that patches/mocks `crud.get_working_representation` and calls peer context with only:

   ```python
   params={"search_query": "honcho retrieval quality"}
   ```

3. Assert the route passes these effective values into `crud.get_working_representation`:

   ```python
   include_most_derived=False
   semantic_search_top_k=12
   max_observations=12
   # plus overfetch/rerank plumbing once added, e.g.
   semantic_overfetch_k=75
   rerank=True
   ```

4. Run the specific test and verify it fails before implementation:

   ```bash
   uv run pytest tests/routes/test_peers.py::<new_test_name> -q
   ```

Expected RED failure: current route passes `include_most_derived=True`, `semantic_search_top_k=None`, and `max_observations=WORKING_REPRESENTATION_MAX_OBSERVATIONS`; overfetch/rerank parameters do not exist yet.

---

### Task 2: Add RED tests proving broad no-query context is unchanged

**Objective:** Guard against accidentally making all peer context narrow/expensive.

**Files:**

- Modify: `tests/routes/test_peers.py`

**Steps:**

1. Add a test that calls peer context without `search_query`.
2. Assert effective values remain broad:

   ```python
   include_most_derived=True
   semantic_search_top_k=None
   max_observations=settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS
   semantic_overfetch_k=None or top_k unchanged
   rerank=False
   ```

3. Run the specific test:

   ```bash
   uv run pytest tests/routes/test_peers.py::<new_broad_context_test_name> -q
   ```

Expected: This may already pass. If it passes immediately, keep it as a guard test but make sure Task 1 is the true RED test.

---

### Task 3: Add RED tests proving explicit opt-in is honored

**Objective:** Preserve caller control for broad blended query context.

**Files:**

- Modify: `tests/routes/test_peers.py`

**Steps:**

1. Add a test that calls peer context with:

   ```python
   params={
       "search_query": "honcho retrieval quality",
       "include_most_frequent": True,
       "max_conclusions": 100,
       "search_top_k": 30,
   }
   ```

2. Assert the route passes:

   ```python
   include_most_derived=True
   semantic_search_top_k=30
   max_observations=100
   ```

3. For rerank behavior, choose one of these and encode it explicitly:
   - preferred: rerank the semantic slice with `semantic_overfetch_k=max(75, search_top_k)` and then blend frequent/recent as before
   - simpler fallback: rerank only when `include_most_frequent` is omitted or false

Recommended first patch: **rerank the semantic slice even when `include_most_frequent=True`**, because explicit blending should not disable better ranking for semantic candidates.

4. Run the specific test:

   ```bash
   uv run pytest tests/routes/test_peers.py::<new_explicit_opt_in_test_name> -q
   ```

---

### Task 4: Implement focused defaults in `src/routers/peers.py`

**Objective:** Change peer context effective defaults only for query-scoped calls.

**Files:**

- Modify: `src/routers/peers.py`

**Implementation shape:**

Change `get_peer_context()` parameter from a hard default boolean to nullable:

```python
include_most_frequent: bool | None = Query(
    default=None,
    description=(
        "Whether to include the most frequent conclusions in the representation. "
        "When omitted, defaults to false for search_query requests and true for broad context."
    ),
)
```

Compute effective values before calling `crud.get_working_representation`:

```python
focused_query = bool(search_query)

effective_include_most_frequent = (
    include_most_frequent
    if include_most_frequent is not None
    else (False if focused_query else True)
)
effective_search_top_k = (
    search_top_k
    if search_top_k is not None
    else (12 if focused_query else None)
)
effective_max_conclusions = (
    max_conclusions
    if max_conclusions is not None
    else (
        12
        if focused_query
        else settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS
    )
)
effective_overfetch_k = 75 if focused_query else None
```

Use those values in the existing `crud.get_working_representation(...)` call, after Tasks 5-7 add the plumbing:

```python
semantic_search_top_k=effective_search_top_k,
semantic_search_overfetch_k=effective_overfetch_k,
semantic_rerank=focused_query,
include_most_derived=effective_include_most_frequent,
max_observations=effective_max_conclusions,
```

---

### Task 5: Add rerank client with safe fallback behavior

**Objective:** Add a small Voyage rerank wrapper that can rerank detached documents and fail closed to vector order.

**Files:**

- Create: `src/utils/rerank.py` or `src/rerank_client.py`
- Test: new unit tests, e.g. `tests/utils/test_rerank.py`

**Implementation shape:**

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import httpx


@dataclass(frozen=True)
class RerankResult:
    index: int
    relevance_score: float | None = None


def rerank_texts(
    *,
    query: str,
    documents: Sequence[str],
    top_k: int,
    model: str = "rerank-2.5-lite",
    timeout: float = 3.0,
) -> list[RerankResult] | None:
    api_key = os.getenv("VOYAGE_API_KEY") or os.getenv("RERANK_API_KEY")
    if not api_key or not query or not documents:
        return None

    try:
        response = httpx.post(
            "https://api.voyageai.com/v1/rerank",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "query": query,
                "documents": list(documents),
                "top_k": min(top_k, len(documents)),
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    results = payload.get("data") or []
    parsed = []
    for item in results:
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(documents):
            parsed.append(RerankResult(index=idx, relevance_score=item.get("relevance_score")))
    return parsed or None
```

**Tests:**

- no API key -> returns `None`
- HTTP timeout/error -> returns `None`
- malformed response -> returns `None`
- valid response -> returns ranked indices
- never logs or returns API key

Run RED first:

```bash
uv run pytest tests/utils/test_rerank.py -q
```

Then implement and rerun.

---

### Task 6: Add overfetch/rerank parameters to representation retrieval

**Objective:** Thread overfetch/rerank options through `crud.get_working_representation()` down to semantic document retrieval.

**Files:**

- Modify: `src/crud/representation.py`
- Modify: tests for representation retrieval, likely `tests/integration/test_representation.py` or a focused unit test

**Implementation shape:**

Add optional parameters:

```python
semantic_search_overfetch_k: int | None = None
semantic_rerank: bool = False
```

Thread them through:

- module-level `get_working_representation(...)`
- `RepresentationManager.get_working_representation(...)`
- `_get_working_representation_internal(...)`
- `_query_documents_semantic(...)`
- `_query_documents_for_level(...)` only if level-scoped semantic queries should also rerank

In `_get_working_representation_internal`, keep the final semantic count as `semantic_observations`, but pass `semantic_search_overfetch_k` to the semantic query function.

Desired behavior:

```python
semantic_docs = await self._query_documents_semantic(
    db,
    query=include_semantic_query,
    top_k=semantic_observations,
    overfetch_k=semantic_search_overfetch_k,
    rerank=semantic_rerank,
    ...
)
```

---

### Task 7: Add overfetch/rerank to document semantic query

**Objective:** Fetch a larger vector candidate pool, rerank in memory, return final top-k documents.

**Files:**

- Modify: `src/crud/document.py` or keep candidate overfetch in `src/crud/representation.py`
- Test: add focused tests for candidate pool sizing and fallback order

**Preferred minimal approach:**

Keep `query_documents(...)` backward-compatible and add optional parameters:

```python
rerank_query: str | None = None
rerank_top_k: int | None = None
```

Or more explicit:

```python
overfetch_k: int | None = None
rerank: bool = False
```

Inside query flow:

1. Compute `candidate_k = max(top_k, overfetch_k or top_k)`.
2. Use existing pgvector/external-vector logic to fetch `candidate_k` candidates.
3. If `rerank` and `query` and enough candidates:
   - convert each document to a rerank text, e.g. `doc.content`
   - call `rerank_texts(query=query, documents=texts, top_k=top_k)`
   - return docs in reranked order
4. If rerank returns `None`, return vector candidates `[:top_k]`.

Important: Avoid extra DB fetch after rerank unless necessary; preserve current detached-doc pattern.

**Tests:**

- overfetch uses larger candidate count than returned top-k
- successful rerank changes order according to returned indices
- failed rerank preserves vector order
- rerank receives document contents, not metadata-only strings
- max returned docs is top-k

---

### Task 8: Preserve explicit broad/blended behavior with reranked semantic slice

**Objective:** Ensure most-derived/recent filler behavior remains only where requested, while semantic slice still benefits from rerank.

**Files:**

- Modify: `src/crud/representation.py`
- Tests: representation tests

**Expected cases:**

1. `search_query`, omitted knobs:
   - semantic count = 12
   - overfetch = 75
   - rerank true
   - top/derived count = 0
   - recent count = 0

2. `search_query`, explicit `include_most_frequent=True`, `max_conclusions=100`, `search_top_k=30`:
   - semantic count = 30
   - top/derived count follows existing blend logic
   - recent fills remaining
   - semantic docs are reranked within their slice

3. no query:
   - no semantic docs
   - no rerank
   - existing top/recent behavior

---

### Task 9: Check peer representation endpoint for same focused/rerank defaults

**Objective:** Decide whether `POST /peers/{peer_id}/representation` should get the same focused default behavior in this self-hosted deployment.

**Files:**

- Inspect/possibly modify: `src/routers/peers.py`
- Possibly modify: `tests/routes/test_peers.py`

**Decision rule:**

If the route accepts `search_query` and is used by SDK/Hermes for focused memory, apply the same effective defaults and rerank plumbing there too.

Preferred helper if both `/context` and `/representation` need it:

```python
def _resolve_focused_representation_defaults(...):
    return EffectiveRepresentationDefaults(
        search_top_k=...,
        include_most_frequent=...,
        max_conclusions=...,
        overfetch_k=...,
        rerank=...,
    )
```

Do not over-abstract unless two call sites need it.

---

### Task 10: Run focused tests

**Objective:** Verify the behavior with relevant route, representation, and rerank tests.

**Commands:**

```bash
uv run pytest tests/utils/test_rerank.py -q
uv run pytest tests/routes/test_peers.py -q
uv run pytest tests/integration/test_representation.py -q
uv run pytest tests/sdk/test_peer.py -q
```

If integration tests require services and fail for environmental reasons, record exact failures and run the smallest mock/unit subsets that verify behavior.

---

### Task 11: Run lint/type sanity on touched files

**Objective:** Ensure style and import correctness.

**Command:**

```bash
uv run ruff check src/routers/peers.py src/crud/representation.py src/crud/document.py src/utils/rerank.py tests/routes/test_peers.py tests/utils/test_rerank.py
```

If config/schema files changed, include them too.

---

### Task 12: Manual/local behavior probe before restart

**Objective:** Validate effective behavior on the actual self-hosted corpus before applying to live Gateway context.

**Approach:**

Use a lightweight direct API/client probe against the dev/self-hosted Honcho environment if it is already running. Do not start or restart services without approval.

Probe comparisons:

1. Broad context:

   ```python
   peer.context(target="grant")
   ```

2. Focused context with rerank:

   ```python
   peer.context(target="grant", search_query="honcho retrieval quality")
   ```

3. Explicit broad opt-in:

   ```python
   peer.context(
       target="grant",
       search_query="honcho retrieval quality",
       include_most_frequent=True,
       max_conclusions=100,
   )
   ```

4. Rerank-disabled fallback probe by temporarily unsetting `VOYAGE_API_KEY` in the probe process, not globally:

   ```bash
   env -u VOYAGE_API_KEY uv run python scripts/probe.py
   ```

Record:

- char count
- observation count
- obvious relevant facts
- stale task-local facts
- harmful constraints
- latency
- whether rerank was used/fallback was triggered

---

### Task 13: Service restart plan after approval

**Objective:** Apply code changes safely to the running self-hosted Honcho + Hermes stack.

**Important:** Grant requires explicit approval before restarting Hermes gateway. Do not restart gateway silently.

Likely restart needs:

1. Restart Honcho API service/container/process so `src/routers/peers.py` and retrieval code changes are loaded.
2. Ensure `VOYAGE_API_KEY` or `RERANK_API_KEY` is available to the Honcho API process if rerank should be live.
3. Hermes gateway restart may not be needed if only Honcho server changed, but may be useful to clear cached base context in the current Hermes process.
4. If Hermes currently has `_base_context_cache` populated, patched Honcho responses may not appear until cache refresh cadence or gateway/session reset.

Recommended rollout message after implementation:

> Honcho focused overfetch/rerank patch is implemented and tests pass. To apply it live, we need to restart the Honcho API service and verify the rerank API key is in the Honcho process environment. Hermes gateway restart is optional unless we want to clear cached memory context immediately. Do you approve restarting Honcho, and do you also want gateway restarted?

---

## Verification Checklist

Before considering implementation complete:

- [ ] Route test proves query-scoped peer context defaults to semantic-only.
- [ ] Route test proves query-scoped peer context requests overfetch/rerank.
- [ ] Route test proves broad no-query peer context remains broad and does not rerank.
- [ ] Route test proves explicit `include_most_frequent=True` remains honored.
- [ ] Rerank unit tests cover no key, timeout/error, malformed response, valid response, and no secret leakage.
- [ ] Retrieval tests prove overfetch candidate count > returned top-k.
- [ ] Retrieval tests prove successful rerank changes order.
- [ ] Retrieval tests prove rerank failure falls back to vector order.
- [ ] Relevant route/representation/SDK tests pass or environmental failures are clearly documented.
- [ ] Ruff passes on touched files.
- [ ] No unrelated pending files in Honcho repo are modified/reverted.
- [ ] Live restart is requested explicitly before applying to services.

---

## Risk Assessment

### Risk: Rerank latency slows every context call

Mitigation: only rerank when `search_query` is present; set short timeout; fallback to vector order.

### Risk: Rerank API unavailable or missing key

Mitigation: `rerank_texts()` returns `None`; retrieval returns vector top-k.

### Risk: Breaking broad memory/profile behavior

Mitigation: focused defaults only when `search_query` is present; add broad-context regression test.

### Risk: Holding DB sessions during external rerank HTTP calls

Mitigation: fetch candidates first, detach/close session, rerank in memory, do not perform HTTP while holding DB connection.

### Risk: Cost explosion

Mitigation: one rerank call per focused peer context; cap candidate pool at 75; use `rerank-2.5-lite` first.

### Risk: Existing Hermes cache masks the live fix

Mitigation: after Honcho restart, either wait for cache refresh cadence or ask Grant whether to restart Hermes gateway/reset session to clear cached context.

---

## High-Level Recommendation

Implement focused defaults and overfetch/rerank together now. The focused-default part removes known frequent/recent filler noise; the overfetch/rerank part addresses the actual retrieval-engine ranking ceiling without moving to Alloy or re-embedding the corpus.

The first version should stay intentionally simple:

```text
query-scoped peer context -> semantic-only budget 12 -> vector overfetch 75 -> Voyage rerank -> top 12 -> fallback to vector order on any rerank issue
```

After validating live quality, the next possible increment would be hybrid BM25/RRF or two-stage reranking/cluster expansion, but those should wait for benchmark evidence.
