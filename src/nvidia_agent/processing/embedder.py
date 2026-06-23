"""
processing/embedder.py — Text chunking, VADER sentiment scoring, and MiniLM embeddings.

Exports
-------
get_embed_model()     → cached SentenceTransformer instance
chunk_text(text)      → list of overlapping word-window chunks
compute_sentiment(text) → VADER compound score in [-1.0, 1.0]
sentiment_label(score)  → 'positive' | 'neutral' | 'negative'
"""

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from nvidia_agent import config

log = logging.getLogger(__name__)

# ── Embedding model ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embed_model() -> SentenceTransformer:
    """Load and cache the MiniLM embedding model (downloaded once, ~80 MB)."""
    log.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    log.info("Embedding model ready.")
    return model


# ── Text chunking ─────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int  = config.CHUNK_SIZE,
    overlap: int     = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping word-window chunks.

    Parameters
    ----------
    text       : Raw article text (title + body).
    chunk_size : Maximum words per chunk (default 400 from config).
    overlap    : Words shared between consecutive chunks (default 60 from config).

    Returns
    -------
    List of chunk strings. Returns [] for empty / whitespace-only input.

    Example
    -------
    chunk_text("word " * 500, chunk_size=400, overlap=60)
    → [chunk_0 (words 0-399), chunk_1 (words 340-739), ...]
    """
    # Guard: empty or whitespace-only input → no chunks
    if not text or not text.strip():
        return []

    words = text.split()
    if not words:
        return []

    # Single chunk — fits entirely within chunk_size
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    start  = 0
    step   = chunk_size - overlap  # advance by (chunk_size - overlap) each iteration

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += step

    return chunks


# ── Sentiment scoring ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_vader() -> SentimentIntensityAnalyzer:
    """Lazy-load and cache the VADER analyser."""
    return SentimentIntensityAnalyzer()


def compute_sentiment(text: str) -> float:
    """Return VADER compound sentiment score in the range [-1.0, 1.0].

    VADER compound thresholds (standard convention):
        >=  0.05  → positive
        <= -0.05  → negative
        between   → neutral

    Returns 0.0 for empty input.
    """
    if not text or not text.strip():
        return 0.0
    analyser = _get_vader()
    scores   = analyser.polarity_scores(text)
    return round(scores["compound"], 4)


def sentiment_label(score: float) -> str:
    """Convert a VADER compound score to a human-readable label.

    Thresholds match standard VADER conventions:
        score >=  0.05  → 'positive'
        score <= -0.05  → 'negative'
        otherwise       → 'neutral'
    """
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"
