"""
verifier_agent.py — Evidence Grounding Verifier
================================================

Scores each recommendation against retrieved evidence using SBERT cosine
similarity. Programmatic, non-LLM verification — objective grounding check.


Used by ceo_agent.py _validate_recommendations() as the grounding check:
  confidence >= 0.5 → recommendation is evidence-grounded (passes)
  confidence <  0.5 → recommendation is flagged as weakly grounded (fails)

Output metrics:
  mean_confidence   → average grounding score across all recommendations
  factual_precision → fraction of recommendations that passed (>= threshold)
"""

import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

GROUNDING_THRESHOLD = 0.5  # minimum cosine similarity to pass verification

# Lazy-load sentence-transformers to avoid import cost at startup
_sbert_model = None


def _get_model():
    """Load SBERT model once and cache it."""
    global _sbert_model
    if _sbert_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("[VERIFIER] Loading SBERT model (all-MiniLM-L6-v2)...")
            _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("[VERIFIER] SBERT model loaded")
        except ImportError:
            logger.warning("[VERIFIER] sentence-transformers not installed — using fallback")
            _sbert_model = None
    return _sbert_model


def _cosine_similarity(vec_a, vec_b) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _fallback_score(rec_text: str, evidence_texts: List[str]) -> float:
    """
    Word-overlap fallback when SBERT is not available.
    Jaccard similarity between recommendation words and evidence words.
    """
    rec_words = set(rec_text.lower().split())
    if not rec_words or not evidence_texts:
        return 0.5
    all_ev_words = set(" ".join(evidence_texts).lower().split())
    overlap = len(rec_words & all_ev_words)
    union   = len(rec_words | all_ev_words)
    return overlap / union if union > 0 else 0.0


def score_recommendation(rec_text: str, evidence_chunks: List[Dict]) -> float:
    """
    Score a single recommendation against its evidence chunks.

    Uses SBERT cosine similarity (semantic grounding):
    - Embeds the recommendation text
    - Embeds each evidence excerpt
    - Returns the MAX similarity (best evidence match)

    If SBERT not available: falls back to word-overlap Jaccard similarity.

    Args:
        rec_text: The recommendation text (action + expected_impact joined)
        evidence_chunks: List of evidence dicts with 'excerpt' or 'text' key

    Returns:
        Float 0.0–1.0 grounding score
    """
    if not evidence_chunks:
        return 0.3  # no evidence = low confidence

    # Extract text from evidence
    ev_texts = [
        str(c.get("excerpt") or c.get("text") or c.get("content") or "")[:300]
        for c in evidence_chunks if isinstance(c, dict)
    ]
    ev_texts = [t for t in ev_texts if t.strip()]

    if not ev_texts:
        return 0.3

    model = _get_model()

    if model is not None:
        try:
            # SBERT semantic similarity
            import numpy as np
            rec_vec = model.encode(rec_text[:512], normalize_embeddings=True)
            ev_vecs = model.encode(ev_texts,       normalize_embeddings=True)
            scores  = [float(np.dot(rec_vec, ev_vec)) for ev_vec in ev_vecs]
            return max(scores)  # best evidence match
        except Exception as e:
            logger.warning(f"[VERIFIER] SBERT scoring failed: {e} — using fallback")

    # Fallback: word overlap
    return min(_fallback_score(rec_text, ev_texts) * 1.5, 1.0)


def verify_recommendations(recommendations: List[Dict]) -> Dict:
    """
    Verify all recommendations for evidence grounding.

    For each recommendation:
    1. Extract its text (recommendation + expected_impact)
    2. Score against its attached evidence using SBERT cosine
    3. Flag as verified (>= 0.5) or unverified (< 0.5)

    Args:
        recommendations: List of recommendation dicts (must have 'evidence' key)

    Returns:
        {
          "verified":          [list of passing recs with confidence scores],
          "unverified":        [list of failing recs],
          "mean_confidence":   float,
          "factual_precision": float (fraction that passed),
          "scores":            [(rec_text, score), ...]
        }
    """
    logger.info(f"[VERIFIER] Verifying {len(recommendations)} recommendations...")

    if not recommendations:
        return {
            "verified":          [],
            "unverified":        [],
            "mean_confidence":   0.0,
            "factual_precision": 0.0,
            "scores":            []
        }

    verified, unverified, scores = [], [], []

    for idx, rec in enumerate(recommendations):
        if not isinstance(rec, dict):
            continue

        # Build recommendation text from whatever fields exist
        rec_text = " ".join(filter(None, [
            str(rec.get("recommendation") or rec.get("action") or rec.get("title") or ""),
            str(rec.get("expected_impact") or rec.get("impact") or ""),
            str(rec.get("rationale") or ""),
        ]))[:512]

        evidence = rec.get("evidence", [])

        # Score grounding
        confidence = score_recommendation(rec_text, evidence)
        scores.append((rec_text[:60], round(confidence, 3)))

        # Attach confidence to recommendation
        rec_with_score = {**rec, "grounding_confidence": round(confidence, 3)}

        if confidence >= GROUNDING_THRESHOLD:
            verified.append(rec_with_score)
            logger.info(f"  [VERIFIER] ✓ Rec #{idx}: confidence={confidence:.2f} — VERIFIED")
        else:
            unverified.append(rec_with_score)
            logger.warning(
                f"  [VERIFIER] ✗ Rec #{idx}: confidence={confidence:.2f} — "
                f"UNVERIFIED (below {GROUNDING_THRESHOLD})"
            )

    # Compute metrics
    all_confidences   = [s[1] for s in scores]
    mean_confidence   = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    factual_precision = len(verified) / len(recommendations) if recommendations else 0.0

    logger.info(
        f"[VERIFIER] Complete: {len(verified)}/{len(recommendations)} verified | "
        f"Mean confidence={mean_confidence:.2f} | "
        f"Factual precision={factual_precision:.0%}"
    )

    return {
        "verified":          verified,
        "unverified":        unverified,
        "mean_confidence":   round(mean_confidence, 3),
        "factual_precision": round(factual_precision, 3),
        "scores":            scores
    }
