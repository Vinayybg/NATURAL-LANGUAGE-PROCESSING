"""
storage/vector_store.py — ChromaDB vector store and BM25 hybrid retrieval.

Two retrieval methods:
  semantic_search() — cosine similarity via ChromaDB HNSW index
  hybrid_search()   — BM25 + cosine fused with Reciprocal Rank Fusion (RRF)

hybrid_search() is what the CEO chat and AI CEO agent both use.
"""

import logging
import sqlite3
from pathlib import Path

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from nvidia_agent import config
from nvidia_agent.processing.embedder import get_embed_model

log = logging.getLogger(__name__)


def get_collection() -> chromadb.Collection:
    """Open (or create) the ChromaDB persistent collection with cosine similarity."""
    Path(config.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=config.CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def semantic_search(query: str, k: int = config.TOP_K_RETRIEVAL) -> list[dict]:
    """
    Embed query → ChromaDB cosine search → return top-k chunks.
    ChromaDB distance is (1 - cosine), so score = 1 - distance.
    """
    model      = get_embed_model()
    collection = get_collection()
    total      = collection.count()
    if total == 0:
        return []

    results = collection.query(
        query_embeddings=[model.encode(query).tolist()],
        n_results=min(k, total),
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append({
            "text":        doc,
            "title":       meta.get("title", ""),
            "source":      meta.get("source_name", ""),
            "source_type": meta.get("source_type", ""),
            "url":         meta.get("url", ""),
            "published":   meta.get("published", ""),
            "sentiment":   meta.get("sentiment", 0.0),
            "score":       round(1 - dist, 4),
        })
    return docs


def hybrid_search(query: str, k: int = config.TOP_K_RETRIEVAL) -> list[dict]:
    """
    Hybrid BM25 + cosine search fused with Reciprocal Rank Fusion (RRF).

    Algorithm:
      1. Retrieve k*3 semantic candidates from ChromaDB
      2. Score those same candidates with BM25Okapi
      3. RRF score = Σ 1/(rank_i + 60) across both ranking lists
         The constant 60 (Cormack et al. 2009) reduces outlier dominance
      4. Sort by combined RRF score, return top-k

    WHY HYBRID?
    Tech content has named entities like "H100", "CUDA", "Blackwell".
    Dense embeddings catch semantic meaning; BM25 catches exact keyword matches.
    Together they cover both — this is what the CEO chat uses.
    """
    candidates = semantic_search(query, k=k * 3)
    if not candidates:
        return []

    n = len(candidates)
    tokenised   = [d["text"].lower().split() for d in candidates]
    bm25        = BM25Okapi(tokenised)
    bm25_scores = bm25.get_scores(query.lower().split())

    rrf_k  = 60
    scores = [0.0] * n

    # Semantic rank contribution (already sorted by cosine desc)
    for rank in range(n):
        scores[rank] += 1.0 / (rank + rrf_k)

    # BM25 rank contribution
    for bm25_rank, orig_idx in enumerate(
        sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)
    ):
        scores[orig_idx] += 1.0 / (bm25_rank + rrf_k)

    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return [candidates[i] for i in ranked[:k]]


def get_sentiment_stats(db_path: str = config.SQLITE_DB_PATH) -> dict:
    """Aggregate VADER sentiment scores from SQLite for dashboard charts."""
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        rows = conn.execute(
            "SELECT source_type, sentiment_score FROM articles "
            "WHERE sentiment_score IS NOT NULL"
        ).fetchall()
        conn.close()
    except Exception:
        return {}

    if not rows:
        return {}

    scores  = [r[1] for r in rows]
    by_type: dict[str, list[float]] = {}
    for st, sc in rows:
        by_type.setdefault(st, []).append(sc)

    return {
        "overall_mean": round(sum(scores) / len(scores), 4),
        "positive_pct": round(sum(1 for s in scores if s >= 0.05) / len(scores) * 100, 1),
        "negative_pct": round(sum(1 for s in scores if s <= -0.05) / len(scores) * 100, 1),
        "neutral_pct":  round(sum(1 for s in scores if -0.05 < s < 0.05) / len(scores) * 100, 1),
        "by_source_type": {
            st: round(sum(v) / len(v), 4) for st, v in by_type.items()
        },
    }
