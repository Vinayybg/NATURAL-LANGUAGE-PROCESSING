"""
config.py — Central configuration for the NVIDIA Strategic Intelligence Agent.
All runtime parameters live here. No other file has hardcoded values.
Edit this file (or use .env) to switch company, LLM, sources, or tuning params.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve the project root (two levels up from src/nvidia_agent/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


# ── Company ───────────────────────────────────────────────────────────────────
COMPANY_NAME   = "NVIDIA Corporation"
COMPANY_TICKER = "NVDA"
COMPANY_SECTOR = "Semiconductors & AI"
COMPANY_HQ     = "Santa Clara, California, USA"


# ── LLM backend ───────────────────────────────────────────────────────────────
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.1-8b-instant")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "qwen3:8b")

HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_MODEL   = os.getenv("HF_MODEL",   "mistralai/Mistral-7B-Instruct-v0.3")


# ── NewsAPI ───────────────────────────────────────────────────────────────────
# Provides breaking news from 150,000+ sources worldwide.
# Free tier: 100 requests/day, articles up to 1 month old.
# Get your free key at: https://newsapi.org/register
# Then add to your .env file:  NEWS_API_KEY=your_key_here
# If not set, NewsAPI collection is silently skipped — everything else still runs.
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")


# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ── Storage paths ─────────────────────────────────────────────────────────────
DATA_DIR        = PROJECT_ROOT / "data"
SQLITE_DB_PATH  = str(DATA_DIR / "articles.db")
CHROMA_DB_PATH  = str(DATA_DIR / "chroma_db")
LOGS_DIR        = PROJECT_ROOT / "logs"
COLLECTION_NAME = "nvidia_intelligence"


# ── RSS Data sources ──────────────────────────────────────────────────────────
RSS_FEEDS = [
    # ── NVIDIA Official ───────────────────────────────────────────────────────
    {"name": "NVIDIA Newsroom",  "url": "https://nvidianews.nvidia.com/releases.xml",  "type": "company"},
    {"name": "NVIDIA Blog",      "url": "https://blogs.nvidia.com/feed/",              "type": "company"},

    # ── Tech News ─────────────────────────────────────────────────────────────
    {"name": "TechCrunch AI",         "url": "https://techcrunch.com/category/artificial-intelligence/feed/", "type": "news"},
    {"name": "The Verge",             "url": "https://www.theverge.com/rss/index.xml",                        "type": "news"},
    {"name": "Ars Technica",          "url": "https://feeds.arstechnica.com/arstechnica/technology-lab",      "type": "news"},
    {"name": "Reuters Technology",    "url": "https://feeds.reuters.com/reuters/technologyNews",              "type": "news"},
    {"name": "VentureBeat AI",        "url": "https://venturebeat.com/category/ai/feed/",                    "type": "news"},
    {"name": "Wired Technology",      "url": "https://www.wired.com/feed/category/science/latest/rss",       "type": "news"},
    {"name": "ZDNet AI",              "url": "https://www.zdnet.com/topic/artificial-intelligence/rss.xml",  "type": "news"},
    {"name": "MIT Technology Review", "url": "https://www.technologyreview.com/feed/",                       "type": "news"},
    {"name": "The Register",          "url": "https://www.theregister.com/headlines.atom",                   "type": "news"},
    {"name": "BBC Technology",        "url": "https://feeds.bbci.co.uk/news/technology/rss.xml",             "type": "news"},
    {"name": "Forbes AI",             "url": "https://www.forbes.com/innovation/ai/feed/",                   "type": "news"},
    {"name": "Business Insider Tech", "url": "https://feeds.businessinsider.com/custom/all",                 "type": "news"},

    # ── Google News (real-time search-based feeds, no key required) ───────────
    {"name": "Google News — NVIDIA",
     "url":  "https://news.google.com/rss/search?q=NVIDIA+AI+chip&hl=en-US&gl=US&ceid=US:en",
     "type": "news"},
    {"name": "Google News — Jensen Huang",
     "url":  "https://news.google.com/rss/search?q=Jensen+Huang&hl=en-US&gl=US&ceid=US:en",
     "type": "news"},
    {"name": "Google News — NVDA stock",
     "url":  "https://news.google.com/rss/search?q=NVDA+stock+earnings&hl=en-US&gl=US&ceid=US:en",
     "type": "news"},
    {"name": "Google News — AMD vs NVIDIA",
     "url":  "https://news.google.com/rss/search?q=AMD+NVIDIA+GPU+AI+competitor&hl=en-US&gl=US&ceid=US:en",
     "type": "news"},

    # ── Research & Industry ───────────────────────────────────────────────────
    {"name": "IEEE Spectrum",       "url": "https://spectrum.ieee.org/feeds/feed.rss",       "type": "research"},
    {"name": "SemiAnalysis",        "url": "https://www.semianalysis.com/feed",               "type": "research"},
    {"name": "AnandTech",           "url": "https://www.anandtech.com/rss/",                  "type": "industry"},
    {"name": "Tom's Hardware",      "url": "https://www.tomshardware.com/feeds/all",           "type": "industry"},
    {"name": "ExtremeTech",         "url": "https://www.extremetech.com/feed",                "type": "industry"},
    {"name": "NotebookCheck",       "url": "https://www.notebookcheck.net/feeds/news.xml",    "type": "industry"},
    {"name": "Digital Trends Tech", "url": "https://www.digitaltrends.com/computing/feed/",  "type": "industry"},

    # ── SEC EDGAR — Official NVIDIA filings ───────────────────────────────────
    {"name": "SEC EDGAR — NVIDIA 8-K",
     "url":  "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001045810&type=8-K&dateb=&owner=include&count=10&search_text=&output=atom",
     "type": "company"},

    # ── Community / Discussion ────────────────────────────────────────────────
    {"name": "Hacker News",             "url": "https://hnrss.org/frontpage",                        "type": "community"},
    {"name": "Reddit r/nvidia",         "url": "https://www.reddit.com/r/nvidia/.rss?limit=25",      "type": "community"},
    {"name": "Reddit r/MachineLearning","url": "https://www.reddit.com/r/MachineLearning/.rss?limit=25", "type": "community"},
    {"name": "Reddit r/hardware",       "url": "https://www.reddit.com/r/hardware/.rss?limit=25",    "type": "community"},
    {"name": "Reddit r/artificial",     "url": "https://www.reddit.com/r/artificial/.rss?limit=25",  "type": "community"},
]


# ── Reddit — OAuth config (optional, not actively used) ──────────────────────
REDDIT_SUBREDDITS = []
REDDIT_QUERY      = "NVIDIA"
REDDIT_LIMIT      = 0


# ── Wikipedia topics (40 background research articles) ───────────────────────
WIKIPEDIA_TOPICS = [
    # Core NVIDIA
    "Nvidia", "CUDA", "Jensen Huang", "GeForce",
    "Blackwell (GPU architecture)", "Hopper (microarchitecture)",
    "Ampere (microarchitecture)", "NVIDIA NVLink", "NVIDIA DGX", "NVIDIA Jetson",

    # Competitors
    "AMD", "Intel", "Qualcomm", "Google TPU", "Cerebras Systems",

    # AI & ML
    "Artificial intelligence", "Machine learning", "Deep learning",
    "Large language model", "Generative artificial intelligence",
    "Transformer (deep learning architecture)",
    "Artificial intelligence accelerator", "Reinforcement learning",
    "Computer vision", "Natural language processing",

    # Hardware & Semiconductors
    "Graphics processing unit", "Semiconductor industry", "Semiconductor",
    "Taiwan Semiconductor Manufacturing Company", "High-bandwidth memory",
    "Data center", "Cloud computing", "Tensor processing unit",

    # Industry context
    "Moore's law", "Autonomous vehicle", "Robotics", "Metaverse",
    "Supercomputer", "Sovereign AI", "AI safety",
]


# ── Collection tuning ─────────────────────────────────────────────────────────
# 15 articles per source keeps collection fast (~5 minutes for a full run).
# You have 30 feeds × 15 = 450 max RSS articles — more than enough with Wikipedia.
# Increase to 25-40 only if you want a longer, slower collection run.
MAX_ARTICLES_PER_SOURCE = 15
MIN_TEXT_LENGTH         = 80


# ── Processing tuning ─────────────────────────────────────────────────────────
CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 60    # overlapping words between consecutive chunks


# ── Retrieval tuning ──────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 12   # chunks retrieved per RAG query (higher = more context)


# ── LLM generation ────────────────────────────────────────────────────────────
LLM_MAX_TOKENS  = 2048   # room for CEO briefing which can be long
LLM_TEMPERATURE = 0.3


# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
