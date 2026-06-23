"""
collectors/base.py — Shared database helpers used by all collectors.
"""

import hashlib
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path

from nvidia_agent import config

log = logging.getLogger(__name__)

NVIDIA_KEYWORDS = [
    "nvidia", "nvda", "jensen huang", "cuda", "gpu", "blackwell",
    "hopper", "h100", "h200", "rtx", "geforce", "ai chip",
    "dgx", "grace hopper", "rubin", "gb200",
]


def init_db(db_path: str = config.SQLITE_DB_PATH) -> sqlite3.Connection:
    """Create the articles table and indexes if they don't exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id              TEXT PRIMARY KEY,
            title           TEXT,
            content         TEXT,
            url             TEXT,
            source_name     TEXT,
            source_type     TEXT,
            published       TEXT,
            collected       TEXT,
            sentiment_score REAL DEFAULT 0.0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_source    ON articles(source_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_type      ON articles(source_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_collected ON articles(collected)")
    conn.commit()
    return conn


def make_article_id(url: str, title: str) -> str:
    """MD5 fingerprint of url+title — used as PRIMARY KEY for deduplication."""
    return hashlib.md5(f"{url}{title}".encode()).hexdigest()


def save_article(conn: sqlite3.Connection, article: dict) -> bool:
    """Insert article; silently skip duplicates. Returns True if new."""
    aid = make_article_id(article.get("url", ""), article.get("title", ""))
    try:
        conn.execute(
            "INSERT INTO articles "
            "(id, title, content, url, source_name, source_type, published, collected) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                aid,
                (article.get("title") or "")[:500],
                (article.get("content") or "")[:50_000],
                article.get("url", ""),
                article.get("source_name", ""),
                article.get("source_type", ""),
                article.get("published", ""),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def is_nvidia_relevant(text: str, source_type: str) -> bool:
    """Return True if article is relevant to NVIDIA (always true for company sources)."""
    if source_type in ("company", "research"):
        return True
    return any(k in text.lower() for k in NVIDIA_KEYWORDS)


def get_article_count(db_path: str = config.SQLITE_DB_PATH) -> int:
    """Return total article count from SQLite."""
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        n    = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0


def get_source_breakdown(db_path: str = config.SQLITE_DB_PATH) -> list[dict]:
    """Return per-source article counts for dashboard chart."""
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        rows = conn.execute(
            "SELECT source_name, COUNT(*) FROM articles "
            "GROUP BY source_name ORDER BY 2 DESC"
        ).fetchall()
        conn.close()
        return [{"source": r[0], "count": r[1]} for r in rows]
    except Exception:
        return []
