"""
tests/test_collector.py — Unit tests for the collector layer.

Run: pytest tests/
"""

import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from nvidia_agent.collectors.base import (
    make_article_id,
    save_article,
    is_nvidia_relevant,
    init_db,
)


class TestDeduplication:
    def test_same_url_title_gives_same_id(self):
        id1 = make_article_id("https://example.com/a", "NVIDIA Reports Record Revenue")
        id2 = make_article_id("https://example.com/a", "NVIDIA Reports Record Revenue")
        assert id1 == id2

    def test_different_url_gives_different_id(self):
        id1 = make_article_id("https://example.com/a", "NVIDIA News")
        id2 = make_article_id("https://example.com/b", "NVIDIA News")
        assert id1 != id2

    def test_duplicate_article_not_saved_twice(self):
        conn = init_db(":memory:")
        article = {
            "title": "NVIDIA Test",
            "content": "Some content about NVIDIA GPUs.",
            "url": "https://nvidia.com/test",
            "source_name": "Test",
            "source_type": "news",
            "published": "2025-01-01",
        }
        saved1 = save_article(conn, article)
        saved2 = save_article(conn, article)
        assert saved1 is True
        assert saved2 is False

    def test_different_articles_both_saved(self):
        conn = init_db(":memory:")
        a1 = {
            "title": "Article One", "content": "NVIDIA GPU content.",
            "url": "https://a.com/1", "source_name": "S",
            "source_type": "news", "published": "",
        }
        a2 = {
            "title": "Article Two", "content": "AMD GPU content.",
            "url": "https://a.com/2", "source_name": "S",
            "source_type": "news", "published": "",
        }
        assert save_article(conn, a1) is True
        assert save_article(conn, a2) is True
        count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        assert count == 2

    def test_empty_url_and_title_still_deduplicates(self):
        conn = init_db(":memory:")
        article = {"title": "", "content": "text", "url": "",
                   "source_name": "S", "source_type": "news", "published": ""}
        saved1 = save_article(conn, article)
        saved2 = save_article(conn, article)
        assert saved1 is True
        assert saved2 is False


class TestRelevanceFilter:
    def test_company_source_always_relevant(self):
        assert is_nvidia_relevant("any text at all", "company") is True

    def test_research_source_always_relevant(self):
        assert is_nvidia_relevant("some tech text", "research") is True

    def test_nvidia_keyword_in_news(self):
        assert is_nvidia_relevant("NVIDIA reports record revenue", "news") is True

    def test_nvda_ticker_in_news(self):
        assert is_nvidia_relevant("NVDA stock reaches new high", "news") is True

    def test_unrelated_news_filtered(self):
        assert is_nvidia_relevant("Manchester United wins the Premier League", "news") is False

    def test_h100_keyword_triggers_relevance(self):
        assert is_nvidia_relevant("H100 GPU demand surges in data centers", "news") is True

    def test_cuda_keyword_triggers_relevance(self):
        assert is_nvidia_relevant("CUDA 12 introduces new memory features", "news") is True

    def test_jensen_huang_triggers_relevance(self):
        assert is_nvidia_relevant("Jensen Huang spoke at Computex 2025", "news") is True

    def test_case_insensitive_matching(self):
        assert is_nvidia_relevant("Nvidia announces new GPU", "news") is True
        assert is_nvidia_relevant("NVIDIA announces new GPU", "news") is True
        assert is_nvidia_relevant("nvidia announces new GPU", "news") is True
