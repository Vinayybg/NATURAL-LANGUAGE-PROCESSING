"""
collectors/wikipedia.py — Wikipedia REST API collector using the summary endpoint.
Uses the /page/summary/ endpoint which is more permissive than the action API.
"""

import time
import logging
import sqlite3
from datetime import datetime, timezone
import requests
from nvidia_agent import config
from nvidia_agent.collectors.base import save_article

log = logging.getLogger(__name__)


def collect_wikipedia(conn: sqlite3.Connection) -> int:
    new = 0
    headers = {
        "User-Agent": "NvidiaIntelligenceAgent/1.0 (university research project; contact@example.com)",
        "Accept": "application/json",
    }

    for topic in config.WIKIPEDIA_TOPICS:
        # Use the REST summary endpoint — much less rate-limited than action API
        slug = topic.replace(" ", "_")
        url  = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(slug)}"

        try:
            r = requests.get(url, headers=headers, timeout=15)

            if r.status_code == 404:
                log.warning(f"  Wikipedia: not found '{topic}'")
                time.sleep(1)
                continue

            if r.status_code != 200:
                log.warning(f"  Wikipedia HTTP {r.status_code} for '{topic}'")
                time.sleep(3)
                continue

            data    = r.json()
            extract = data.get("extract", "")
            title   = data.get("title", topic)
            page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

            if not page_url:
                page_url = f"https://en.wikipedia.org/wiki/{requests.utils.quote(slug)}"

            if len(extract) < 100:
                log.warning(f"  Wikipedia: too short '{topic}' ({len(extract)} chars)")
                time.sleep(1)
                continue

            saved = save_article(conn, {
                "title":       f"Wikipedia: {title}",
                "content":     extract[:15_000],
                "url":         page_url,
                "source_name": "Wikipedia",
                "source_type": "research",
                "published":   datetime.now(timezone.utc).isoformat(),
            })

            if saved:
                new += 1
                log.info(f"  Wikipedia: saved '{topic}' ({len(extract)} chars)")
            else:
                log.info(f"  Wikipedia: already exists '{topic}'")

        except Exception as e:
            log.warning(f"  Wikipedia error for '{topic}': {e}")

        time.sleep(1)   # 1 second between requests — REST API is more permissive

    log.info(f"  Wikipedia total new: {new}")
    return new