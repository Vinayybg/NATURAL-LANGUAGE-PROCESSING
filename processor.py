"""
processor.py — Top-level processing orchestrator.
Reads raw articles from SQLite, runs VADER sentiment, chunks text,
generates MiniLM embeddings, and upserts into ChromaDB.

Run standalone:  python processor.py
Called by:       pipeline.py and the Streamlit dashboard sidebar
"""

# ── CRITICAL: must come before any nvidia_agent imports ──────────────────────
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
# ─────────────────────────────────────────────────────────────────────────────

import sqlite3
import logging
from nvidia_agent import config
from nvidia_agent.storage.vector_store import get_collection
from nvidia_agent.processing.embedder import (
    get_embed_model, chunk_text, compute_sentiment
)

log = logging.getLogger(__name__)


def run_processing() -> dict:
    """
    Process every unembedded article in SQLite.

    For each new article:
      1. compute_sentiment() → VADER compound score saved back to SQLite
      2. chunk_text()        → split into 400-word overlapping chunks
      3. model.encode()      → 384-dim MiniLM vector per chunk
      4. collection.upsert() → persist vector + metadata to ChromaDB

    Idempotent: articles already indexed in ChromaDB are skipped.
    """
    log.info("=" * 60)
    log.info("TASKS 2 & 3 — Processing & Embedding")
    log.info("=" * 60)

    # timeout=30 prevents "database is locked" when dashboard + pipeline run together
    db_conn    = sqlite3.connect(config.SQLITE_DB_PATH, timeout=30)
    collection = get_collection()
    model      = get_embed_model()

    rows = db_conn.execute(
        "SELECT id, title, content, source_name, source_type, published, url "
        "FROM articles"
    ).fetchall()

    log.info(f"Articles to process: {len(rows)}")

    new_chunks = already_indexed = errors = 0

    for (art_id, title, content, source_name, source_type, published, url) in rows:

        # Skip articles already indexed in ChromaDB
        existing = collection.get(where={"article_id": {"$eq": art_id}}, limit=1)
        if existing and existing["ids"]:
            already_indexed += 1
            continue

        # VADER sentiment on title + first 2000 chars of content
        preview   = f"{title or ''} {(content or '')[:2000]}"
        sentiment = compute_sentiment(preview)
        db_conn.execute(
            "UPDATE articles SET sentiment_score=? WHERE id=?",
            (sentiment, art_id)
        )

        # Chunk the full text
        full_text = f"{title}\n\n{content}" if title else (content or "")
        chunks    = chunk_text(full_text)

        ids = []; documents = []; embeddings = []; metadatas = []
        for i, chunk in enumerate(chunks):
            ids.append(f"{art_id}_{i}")
            documents.append(chunk)
            embeddings.append(model.encode(chunk).tolist())
            metadatas.append({
                "article_id":  art_id,
                "title":       (title or "")[:200],
                "source_name": source_name or "",
                "source_type": source_type or "",
                "published":   published or "",
                "url":         url or "",
                "sentiment":   sentiment,
                "chunk_index": i,
            })

        if not ids:
            continue

        try:
            collection.upsert(
                ids=ids, documents=documents,
                embeddings=embeddings, metadatas=metadatas
            )
            new_chunks += len(ids)
        except Exception as e:
            log.warning(f"ChromaDB upsert error for {art_id}: {e}")
            errors += 1

    db_conn.commit()
    db_conn.close()

    total_vectors = collection.count()
    log.info(
        f"Processing done | new_chunks={new_chunks} | "
        f"already_indexed={already_indexed} | errors={errors} | "
        f"total_vectors={total_vectors}"
    )
    return {
        "new_chunks":      new_chunks,
        "already_indexed": already_indexed,
        "errors":          errors,
        "total_vectors":   total_vectors,
    }


def get_recent_articles(limit: int = 30) -> list[dict]:
    """Return most recently collected articles for the dashboard news feed."""
    try:
        conn = sqlite3.connect(config.SQLITE_DB_PATH, timeout=10)
        rows = conn.execute(
            "SELECT title, url, source_name, source_type, published, sentiment_score "
            "FROM articles ORDER BY collected DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [
            {"title": r[0], "url": r[1], "source": r[2],
             "source_type": r[3], "published": r[4], "sentiment": r[5]}
            for r in rows
        ]
    except Exception:
        return []


if __name__ == "__main__":
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run_processing()
