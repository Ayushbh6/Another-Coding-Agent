"""
Memory tools for ACA.

compact_context  — soft-evicts oldest past Q&A pairs from active context.
search_memory    — hybrid retriever: SQLite FTS5/BM25 + ChromaDB vector search.

Both tools are available in all permission modes (READ and above).

Design note on search_memory:
  The hybrid retriever is intentionally kept as one unified tool. The agent
  does not need to know whether it is searching Q&A pairs, tool traces, or
  task workspaces — it issues a natural language query and gets ranked results.
  The BM25 + vector merge with recency bias is handled here, not by the agent.

Register into a ToolRegistry via:
    from aca.tools import memory
    memory.register(registry)
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from aca.tools.registry import ToolCategory, ToolDefinition, ToolRegistry


# ── compact_context ───────────────────────────────────────────────────────────

def compact_context(
    session_id: str,
    turn_id: str,
    agent: str,
    context_tokens_before: int,
    compacted_turn_ids: list[str],
    context_tokens_after: int,
    db: Any = None,
) -> dict:
    """
    Record that a batch of past Q&A pairs has been evicted from active context.

    This tool does NOT delete any data. It logs the compaction event to
    context_compactions and returns the list of evicted turn IDs so the
    caller (BaseAgent) can drop them from its in-memory history list.

    Rules per ARCHITECTURE.md §13:
      - Only Category B content (past Q&A pairs) may be compacted.
      - Current-turn tool calls are never touched.
      - Eviction is oldest-first.

    Returns {"evicted_turn_ids": list[str], "tokens_freed": int}
    """
    if not compacted_turn_ids:
        return {"evicted_turn_ids": [], "tokens_freed": 0}

    tokens_freed = max(0, context_tokens_before - context_tokens_after)
    compacted_at = int(time.time() * 1000)

    if db is not None:
        for evicted_turn_id in compacted_turn_ids:
            try:
                db.execute(
                    """
                    INSERT INTO context_compactions (
                        compaction_id, session_id, turn_id, agent,
                        compacted_turn_id, context_tokens_before,
                        context_tokens_after, compacted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        session_id, turn_id, agent,
                        evicted_turn_id,
                        context_tokens_before,
                        context_tokens_after,
                        compacted_at,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[ACA][memory] DB log failed for compaction of turn {evicted_turn_id}: {exc}")
        try:
            db.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"[ACA][memory] DB commit failed after compaction: {exc}")

    return {
        "evicted_turn_ids": compacted_turn_ids,
        "tokens_freed": tokens_freed,
    }


# ── search_memory ─────────────────────────────────────────────────────────────

def search_memory(
    query: str,
    top_n: int = 10,
    db: Any = None,
    chroma_client: Any = None,
    embed_fn: Any = None,
    recency_weight: float = 0.1,
) -> dict:
    """
    Hybrid memory retrieval: SQLite FTS5/BM25 + ChromaDB vector search.

    Covers:
      - past Q&A pairs (turns table)
      - tool call inputs/outputs (tool_calls table)

    Algorithm:
      1. FTS5 BM25 query on turns_fts and tool_calls_fts
      2. Vector similarity query in ChromaDB (if chroma_client + embed_fn provided)
      3. Score merge: normalised_bm25 + normalised_vector + recency_weight * recency_score
      4. Deduplicate by ID, sort by merged score, return top_n

    When chroma_client or embed_fn is None, falls back to BM25-only results.

    Returns:
      {
        "query": str,
        "results": [
          {
            "id": str,         # turn_id or tool_call_id
            "source": str,     # "turn" | "tool_call"
            "score": float,
            "content": str,    # user_message+final_response or input+output
            "metadata": dict,
          }
        ]
      }
    """
    results: list[dict] = []

    # ── BM25 via FTS5 ─────────────────────────────────────────────────────────
    if db is not None:
        bm25_hits: list[dict] = []

        # Search turns
        try:
            rows = db.execute(
                """
                SELECT t.turn_id, t.user_message, t.final_response,
                       t.started_at, bm25(turns_fts) AS score
                FROM turns_fts
                JOIN turns t USING (rowid)
                WHERE turns_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (query, top_n * 2),
            ).fetchall()
            for row in rows:
                bm25_hits.append({
                    "id": row[0],
                    "source": "turn",
                    "bm25_score": abs(float(row[4])),
                    "started_at": row[3] or 0,
                    "content": f"Q: {row[1]}\nA: {row[2] or ''}",
                    "metadata": {"turn_id": row[0]},
                })
        except Exception as exc:  # noqa: BLE001
            print(f"[ACA][memory] FTS turns query failed: {exc}")

        # Search tool calls
        try:
            rows = db.execute(
                """
                SELECT tc.tool_call_id, tc.tool_name, tc.input_json,
                       tc.output_json, tc.started_at, bm25(tool_calls_fts) AS score
                FROM tool_calls_fts
                JOIN tool_calls tc USING (rowid)
                WHERE tool_calls_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (query, top_n * 2),
            ).fetchall()
            for row in rows:
                bm25_hits.append({
                    "id": row[0],
                    "source": "tool_call",
                    "bm25_score": abs(float(row[5])),
                    "started_at": row[4] or 0,
                    "content": f"tool={row[1]} input={row[2]} output={row[3] or ''}",
                    "metadata": {"tool_call_id": row[0], "tool_name": row[1]},
                })
        except Exception as exc:  # noqa: BLE001
            print(f"[ACA][memory] FTS tool_calls query failed: {exc}")

        results.extend(bm25_hits)

    # ── Vector search via ChromaDB ────────────────────────────────────────────
    vector_scores: dict[str, float] = {}
    if chroma_client is not None and embed_fn is not None:
        try:
            query_embedding = embed_fn(query)

            for collection_name, source_label in [("turns", "turn"), ("tool_calls", "tool_call")]:
                try:
                    col = chroma_client.get_collection(collection_name)
                    vresults = col.query(
                        query_embeddings=[query_embedding],
                        n_results=top_n,
                        include=["distances", "documents", "metadatas"],
                    )
                    ids = vresults["ids"][0]
                    distances = vresults["distances"][0]
                    docs = vresults["documents"][0]
                    metas = vresults["metadatas"][0]

                    for doc_id, dist, doc, meta in zip(ids, distances, docs, metas):
                        # Convert L2 distance to a similarity score (lower distance = higher score)
                        similarity = 1.0 / (1.0 + dist)
                        vector_scores[doc_id] = similarity

                        # Add to results only if not already present from BM25
                        existing_ids = {r["id"] for r in results}
                        if doc_id not in existing_ids:
                            results.append({
                                "id": doc_id,
                                "source": source_label,
                                "bm25_score": 0.0,
                                "started_at": meta.get("started_at", 0),
                                "content": doc,
                                "metadata": meta,
                            })
                except Exception as exc:  # noqa: BLE001
                    print(f"[ACA][memory] ChromaDB query failed for '{collection_name}': {exc}")

        except Exception as exc:  # noqa: BLE001
            print(f"[ACA][memory] Embedding failed: {exc}")

    # ── Score merge with recency bias ─────────────────────────────────────────
    if not results:
        return {"query": query, "results": []}

    # Normalise BM25 scores to [0, 1]
    max_bm25 = max((r["bm25_score"] for r in results), default=1.0) or 1.0
    now_ms = int(time.time() * 1000)
    max_age_ms = 7 * 24 * 60 * 60 * 1000  # 7 days in ms

    for r in results:
        norm_bm25 = r["bm25_score"] / max_bm25
        norm_vector = vector_scores.get(r["id"], 0.0)
        age_ms = max(0, now_ms - r.get("started_at", now_ms))
        recency_score = max(0.0, 1.0 - age_ms / max_age_ms)
        r["score"] = norm_bm25 + norm_vector + recency_weight * recency_score

    # Deduplicate by ID, keep highest score
    seen: dict[str, dict] = {}
    for r in results:
        rid = r["id"]
        if rid not in seen or r["score"] > seen[rid]["score"]:
            seen[rid] = r

    ranked = sorted(seen.values(), key=lambda x: -x["score"])[:top_n]

    # Clean up internal scoring fields before returning
    for r in ranked:
        r.pop("bm25_score", None)
        r.pop("started_at", None)

    return {"query": query, "results": ranked}


# ── Schema definitions ────────────────────────────────────────────────────────

_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "compact_context",
            "description": (
                "Record that past Q&A pairs have been evicted from active context to free token budget. "
                "Only call this when context is approaching its token limit. "
                "Only evict past Q&A pairs (Category B) — never current-turn tool calls. "
                "Evict oldest pairs first. The evicted pairs remain in the backend and are "
                "retrievable via search_memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Current session ID.",
                    },
                    "turn_id": {
                        "type": "string",
                        "description": "Current turn ID (the turn during which compaction is happening).",
                    },
                    "agent": {
                        "type": "string",
                        "description": "Agent performing the compaction: 'james' or 'worker'.",
                    },
                    "context_tokens_before": {
                        "type": "integer",
                        "description": "Estimated token count before eviction.",
                    },
                    "compacted_turn_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of turn_ids for the Q&A pairs being evicted (oldest first).",
                    },
                    "context_tokens_after": {
                        "type": "integer",
                        "description": "Estimated token count after eviction.",
                    },
                },
                "required": [
                    "session_id", "turn_id", "agent",
                    "context_tokens_before", "compacted_turn_ids", "context_tokens_after",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search the full backend memory for past information using a hybrid "
                "BM25 keyword + vector similarity approach. "
                "Use this whenever you need information that is no longer in active context — "
                "past Q&A pairs, prior tool outputs, previous task findings. "
                "Do NOT guess or hallucinate — use this tool to retrieve instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing what you are looking for.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Maximum number of results to return. Defaults to 10.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


# ── Registration ──────────────────────────────────────────────────────────────

_FN_MAP = {
    "compact_context": compact_context,
    "search_memory": search_memory,
}


def register(registry: ToolRegistry) -> None:
    """Register all memory tools into the given ToolRegistry."""
    for schema in _SCHEMAS:
        name = schema["function"]["name"]
        registry.register(ToolDefinition(
            fn=_FN_MAP[name],
            schema=schema,
            category=ToolCategory.MEMORY,
        ))
