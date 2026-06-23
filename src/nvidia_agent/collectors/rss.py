"""
rss.py — RSS feed collector + NewsAPI collector (Task 1, Sources 1 & 2).
NewsAPI is integrated directly here — no separate file needed.
"""

import time
import logging
import sqlite3
import requests

from nvidia_agent import config
from nvidia_agent.collectors.base import save_article, is_nvidia_relevant
from nvidia_agent.processing.cleaner import clean_html, fetch_url

import feedparser

log = logging.getLogger(__name__)

# ── NewsAPI search queries ────────────────────────────────────────────────────
# These target the most strategically relevant NVIDIA topics.
NEWSAPI_QUERIES = [
    "NVIDIA GPU AI chip",
    "Jensen Huang NVIDIA",
    "NVDA stock earnings",
    "Blackwell GPU data center",
    "NVIDIA AMD Intel competitor",
    "NVIDIA export controls China",
    "NVIDIA autonomous driving robotics",
    "AI infrastructure hyperscaler NVIDIA",
]

NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"


# ── RSS collector ─────────────────────────────────────────────────────────────

def collect_rss(conn: sqlite3.Connection) -> int:
    """Collect from all configured RSS feeds."""
    new = 0
    for feed_cfg in config.RSS_FEEDS:
        log.info(f"  RSS: {feed_cfg['name']}")
        try:
            feed    = feedparser.parse(feed_cfg["url"])
            entries = feed.entries[:config.MAX_ARTICLES_PER_SOURCE]
        except Exception as e:
            log.warning(f"  RSS parse error for {feed_cfg['name']}: {e}")
            continue

        for entry in entries:
            title   = entry.get("title", "")
            url     = entry.get("link", "")
            summary = clean_html(entry.get("summary", entry.get("description", "")))
            pub     = entry.get("published", entry.get("updated", ""))
            full    = summary

            if url and len(summary) < 400:
                raw = fetch_url(url)
                if raw:
                    full = clean_html(raw)[:10_000]

            combined = f"{title} {full}"
            if not is_nvidia_relevant(combined, feed_cfg["type"]):
                continue
            if len(full) < config.MIN_TEXT_LENGTH:
                continue

            saved = save_article(conn, {
                "title":       title,
                "content":     full,
                "url":         url,
                "source_name": feed_cfg["name"],
                "source_type": feed_cfg["type"],
                "published":   pub,
            })
            if saved:
                new += 1

        time.sleep(0.5)

    log.info(f"  RSS total new: {new}")
    return new


# ── NewsAPI collector ─────────────────────────────────────────────────────────

def collect_newsapi(conn: sqlite3.Connection) -> int:
    """
    Collect breaking news from 150,000+ sources via NewsAPI.
    Requires NEWS_API_KEY in .env — get a free key at https://newsapi.org
    Free tier: 100 requests/day, articles up to 1 month old.
    Skips gracefully if key is missing so the rest of collection still runs.
    """
    api_key = getattr(config, "NEWS_API_KEY", "") or ""
    if not api_key:
        log.info("  NewsAPI: NEWS_API_KEY not set — skipping (add to .env to enable)")
        return 0

    new   = 0
    seen  = set()   # deduplicate URLs within this run

    for query in NEWSAPI_QUERIES:
        try:
            resp = requests.get(
                NEWSAPI_ENDPOINT,
                params={
                    "q":        query,
                    "language": "en",
                    "sortBy":   "publishedAt",   # newest first
                    "pageSize": 20,
                    "apiKey":   api_key,
                },
                timeout=15,
            )

            if resp.status_code == 401:
                log.error("  NewsAPI: Invalid API key — check NEWS_API_KEY in .env")
                break
            if resp.status_code == 426:
                log.warning("  NewsAPI: Free tier limit reached for today")
                break
            if resp.status_code != 200:
                log.warning(f"  NewsAPI: HTTP {resp.status_code} for query '{query}'")
                continue

            articles = resp.json().get("articles", [])
            log.info(f"  NewsAPI query '{query}': {len(articles)} articles")

            for art in articles:
                url = art.get("url", "")

                # skip [Removed] articles (NewsAPI free tier limitation)
                if not url or url in seen or "[Removed]" in url:
                    continue
                seen.add(url)

                title       = art.get("title", "") or ""
                description = art.get("description", "") or ""
                content     = art.get("content", "") or ""
                source_name = art.get("source", {}).get("name", "NewsAPI")
                published   = art.get("publishedAt", "")

                # Combine all available text
                full_text = f"{title}. {description} {content}".strip()

                # Apply same relevance filter as RSS
                if not is_nvidia_relevant(full_text, "news"):
                    continue
                if len(full_text) < config.MIN_TEXT_LENGTH:
                    continue

                saved = save_article(conn, {
                    "title":       title,
                    "content":     full_text,
                    "url":         url,
                    "source_name": source_name,
                    "source_type": "news",
                    "published":   published,
                })
                if saved:
                    new += 1

            # Stay well within rate limits (free tier: 100 req/day)
            time.sleep(1.5)

        except requests.exceptions.Timeout:
            log.warning(f"  NewsAPI timeout for query '{query}'")
        except requests.exceptions.ConnectionError:
            log.warning(f"  NewsAPI connection error for query '{query}'")
        except Exception as e:
            log.warning(f"  NewsAPI error for query '{query}': {e}")

    log.info(f"  NewsAPI total new: {new}")
    return new
