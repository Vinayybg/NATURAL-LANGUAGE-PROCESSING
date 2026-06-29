"""
tests/test_processing.py — Unit tests for the processing layer.

Run: pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from nvidia_agent.processing.embedder import chunk_text, compute_sentiment, sentiment_label


class TestChunking:
    def test_short_text_returns_one_chunk(self):
        text = "NVIDIA reported record revenue this quarter."
        chunks = chunk_text(text)
        assert len(chunks) >= 1

    def test_long_text_produces_multiple_chunks(self):
        # 1000 words → expect at least 2 chunks with chunk_size=400, overlap=60
        text = ("word " * 1000).strip()
        chunks = chunk_text(text)
        assert len(chunks) >= 2

    def test_overlap_means_last_words_of_chunk_n_start_chunk_n1(self):
        text = ("word " * 900).strip()
        chunks = chunk_text(text)
        if len(chunks) >= 2:
            end_of_first   = chunks[0].split()[-5:]
            start_of_second = chunks[1].split()[:5]
            overlap = set(end_of_first) & set(start_of_second)
            assert len(overlap) > 0

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []

    def test_chunks_are_strings(self):
        chunks = chunk_text("NVIDIA GPU AI chip semiconductor.")
        for c in chunks:
            assert isinstance(c, str)

    def test_chunk_length_respects_limit(self):
        text = ("word " * 1000).strip()
        chunks = chunk_text(text)
        for c in chunks:
            word_count = len(c.split())
            # Allow some slack for the overlap window
            assert word_count <= 500, f"Chunk too long: {word_count} words"


class TestSentiment:
    def test_positive_news_scores_positive(self):
        text = "NVIDIA reports outstanding record-breaking revenue growth and dominant market share."
        score = compute_sentiment(text)
        assert score > 0.05, f"Expected positive, got {score}"

    def test_negative_news_scores_negative(self):
        text = "NVIDIA faces serious regulatory crisis and devastating supply chain collapse."
        score = compute_sentiment(text)
        assert score < -0.05, f"Expected negative, got {score}"

    def test_neutral_text_scores_near_zero(self):
        text = "NVIDIA released a quarterly report on Thursday."
        score = compute_sentiment(text)
        assert -0.5 < score < 0.5

    def test_score_in_valid_range(self):
        text = "NVIDIA announced new products at GTC conference."
        score = compute_sentiment(text)
        assert -1.0 <= score <= 1.0

    def test_empty_string_returns_zero(self):
        score = compute_sentiment("")
        assert score == 0.0

    def test_label_positive(self):
        assert sentiment_label(0.8)  == "positive"
        assert sentiment_label(0.05) == "positive"

    def test_label_negative(self):
        assert sentiment_label(-0.8)  == "negative"
        assert sentiment_label(-0.05) == "negative"

    def test_label_neutral(self):
        assert sentiment_label(0.0)   == "neutral"
        assert sentiment_label(0.04)  == "neutral"
        assert sentiment_label(-0.04) == "neutral"

    def test_label_boundary_values(self):
        # Exactly at boundary — 0.05 is positive, -0.05 is negative
        assert sentiment_label(0.05)  == "positive"
        assert sentiment_label(-0.05) == "negative"
