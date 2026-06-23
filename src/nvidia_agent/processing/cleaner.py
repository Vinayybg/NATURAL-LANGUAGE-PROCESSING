"""
processing/cleaner.py — Text cleaning and HTTP fetch utilities.
"""

import re
import logging
import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


def clean_html(raw: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    soup = BeautifulSoup(raw or "", "lxml")
    return re.sub(r"\s+", " ", soup.get_text()).strip()


def fetch_url(url: str, timeout: int = 10) -> str | None:
    """GET a URL, return raw response text or None on failure."""
    headers = {"User-Agent": "Mozilla/5.0 (NvidiaAgentBot/1.0)"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        log.warning(f"fetch_url failed for {url}: {e}")
        return None
