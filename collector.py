"""
collector.py — Top-level collection orchestrator (Task 1).

DATA SOURCES:
  1. RSS feeds       — 30 feeds: NVIDIA official, news, Google News, industry, community
  2. NewsAPI         — 8 targeted queries (requires NEWS_API_KEY in .env, optional)
  3. Wikipedia       — 40 background research articles via REST API

NOTE ON REDDIT:
  Reddit's API now requires OAuth authentication for direct access.
  Reddit content IS still collected via RSS feeds in config.py:
    - Reddit r/nvidia          → RSS feed in RSS_FEEDS list
    - Reddit r/MachineLearning → RSS feed in RSS_FEEDS list
    - Reddit r/hardware        → RSS feed in RSS_FEEDS list
    - Reddit r/artificial      → RSS feed in RSS_FEEDS list
  These are stored with source_type="community" and appear in the Community tab.
  The reddit.py OAuth module is not used — Reddit RSS workaround covers community data.

Run standalone:  python collector.py
Called by:       pipeline.py and the Streamlit dashboard sidebar
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import logging
from nvidia_agent import config
from nvidia_agent.collectors.base      import init_db
from nvidia_agent.collectors.rss       import collect_rss, collect_newsapi
from nvidia_agent.collectors.wikipedia import collect_wikipedia

log = logging.getLogger(__name__)


def run_collection() -> dict:
    """
    Run all data collection stages in sequence.

    Sources collected:
      - RSS:       30 feeds including NVIDIA official, news outlets, Reddit RSS,
                   Google News, industry publications, SEC EDGAR
      - NewsAPI:   8 targeted NVIDIA queries (skipped if no key)
      - Wikipedia: 40 background topics via REST API

    Returns dict with new article count, total count, and per-source breakdown.
    """
    log.info("=" * 60)
    log.info("TASK 1 — Live Data Collection")
    log.info("=" * 60)

    conn = init_db()

    # ── Stage 1: RSS feeds (30 sources incl. Reddit RSS workaround) ──────────
    log.info("Running RSS collection (30 feeds including Reddit RSS)...")
    rss_new = collect_rss(conn)

    # ── Stage 2: NewsAPI — silently skipped if NEWS_API_KEY not set ──────────
    log.info("Running NewsAPI collection (skipped if no key)...")
    newsapi_new = collect_newsapi(conn)

    # ── Stage 3: Wikipedia background research ────────────────────────────────
    log.info("Running Wikipedia collection (40 topics)...")
    wiki_new = collect_wikipedia(conn)

    results = {
        "rss":       rss_new,
        "newsapi":   newsapi_new,
        "wikipedia": wiki_new,
    }

    total_new   = sum(results.values())
    total_in_db = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    conn.close()

    log.info(
        f"Collection done | new={total_new} | total={total_in_db} | "
        f"breakdown={results}"
    )

    if total_in_db < 100:
        log.warning(
            f"Only {total_in_db} articles in DB — spec requires 100+. "
            "Run collection again or add NEWS_API_KEY to .env."
        )

    return {"new": total_new, "total": total_in_db, "breakdown": results}


if __name__ == "__main__":
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    result = run_collection()
    print(f"\n✅ Collection complete!")
    print(f"   New articles : {result['new']}")
    print(f"   Total in DB  : {result['total']}")
    print(f"   Breakdown    :")
    for source, count in result["breakdown"].items():
        print(f"     {source:<12}: {count}")
