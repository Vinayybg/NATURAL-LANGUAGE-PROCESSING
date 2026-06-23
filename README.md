# NVIDIA Strategic Intelligence Agent

> **NLP Final Examination — M.Sc. Data Science & AI**  
> SRH University Heidelberg · June 2026  
> AI-powered executive intelligence platform for strategic decision-making

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Quick Start](#2-quick-start)
3. [System Architecture Diagram](#3-system-architecture-diagram)
4. [Data Flow Diagram](#4-data-flow-diagram)
5. [Technology Stack](#5-technology-stack)
6. [Design Decisions](#6-design-decisions)
7. [AI Pipeline](#7-ai-pipeline)
8. [Project Structure](#8-project-structure)
9. [Dashboard Sections](#9-dashboard-sections)
10. [Configuration Reference](#10-configuration-reference)
11. [Examination Compliance](#11-examination-compliance)

---

## 1. Project Overview

The NVIDIA Strategic Intelligence Agent is a fully automated RAG-powered system
that transforms live public information into executive-level strategic intelligence.

The system continuously collects news, company announcements, research publications,
and community discussions about NVIDIA Corporation. It processes and embeds this
information into a vector knowledge base, then runs an AI CEO Agent that identifies
opportunities, risks, and trends — generating evidence-backed recommendations and
an executive briefing for the CEO.

**The core distinction:** This is not an information retrieval system.
It is a strategic decision-making system. Every recommendation is grounded in
live evidence, prioritised by business impact, and accompanied by a three-dimensional
risk assessment.

### Key Metrics

| Metric | Value |
|--------|-------|
| Articles collected | Grows with each pipeline run (100+ minimum required) |
| Data source types | 5 — Company, News, Research, Industry, Community |
| Active RSS feeds | 31 |
| Wikipedia background topics | 40 |
| Vector chunks in ChromaDB | Grows with articles (approx. 2-3 chunks per article) |
| Embedding dimensions | 384 (all-MiniLM-L6-v2) |
| LLM | llama-3.1-8b-instant (open-source, Llama 3 Community Licence) |
| Dashboard sections | 7 |
| Unit tests | 19 |

> **Note:** Article and chunk counts shown in the dashboard reflect the live database
> state and increase each time `python pipeline.py --collect` is run.

---

## 2. Quick Start

### Prerequisites

- Python 3.11 or 3.12
- Free Groq API key — [console.groq.com](https://console.groq.com)
- 2 GB free disk space (model weights + data)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/nvidia_strategic_intelligence.git
cd nvidia_strategic_intelligence

# 2. Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Running the Pipeline

```bash
# Step 1 — Collect live data (~3-5 minutes)
python pipeline.py --collect

# Step 2 — Process and embed (~2 minutes, downloads model on first run)
python pipeline.py --process

# Step 3 — Run AI analysis (~2 minutes)
python pipeline.py --analyse

# Or run all three steps at once and save a report
python pipeline.py --all --report

# Step 4 — Launch the dashboard
streamlit run app.py
```

### Alternative LLM Backends

```bash
# Use Ollama (local, no internet after model download)
# Install: https://ollama.com then: ollama pull qwen3:8b
# In .env set: LLM_BACKEND=ollama

# Use HuggingFace Inference API (free)
# In .env set: LLM_BACKEND=huggingface
#              HF_API_KEY=your_token
```

---

## 3. System Architecture Diagram

```
+-------------------------------------------------------------------------+
|              NVIDIA STRATEGIC INTELLIGENCE AGENT                        |
|                   5-Layer Pipeline Architecture                         |
+-------------------------------------------------------------------------+

+----------------------------------------------------------------------+
|  LAYER 1 - LIVE DATA COLLECTION                              Task 1  |
|                                                                      |
|  +-------------+  +-------------+  +-------------+  +-----------+  |
|  | NVIDIA      |  | 28 News &   |  |  Wikipedia  |  |  NewsAPI  |  |
|  | Official    |  | Industry    |  |  REST API   |  | optional  |  |
|  | RSS + SEC   |  | RSS Feeds   |  |  40 topics  |  | 8 queries |  |
|  | EDGAR 8-K   |  | + Google    |  |             |  |           |  |
|  |             |  | News RSS    |  |             |  |           |  |
|  +------+------+  +------+------+  +------+------+  +-----+-----+  |
|         |                |                |                |        |
|         +----------------+----------------+----------------+        |
|                          |                                          |
|          feedparser / requests / BeautifulSoup4 / lxml             |
|          MD5 deduplication / NVIDIA keyword relevance filter       |
+--------------------------|-------------------------------------------+
                           |
+--------------------------|-------------------------------------------+
|  LAYER 2 - SQLITE STORAGE                                    Task 2  |
|                                                                      |
|  articles.db  WAL mode  B-tree indexes on source, type, date        |
|                                                                      |
|  id (MD5 PRIMARY KEY) | title | content | url | source_name         |
|  source_type | published | collected | sentiment_score              |
+--------------------------+-------------------------------------------+
                           |
+--------------------------v-------------------------------------------+
|  LAYER 3 - INFORMATION PROCESSING                            Task 3  |
|                                                                      |
|  [1] VADER sentiment scoring  -> compound in [-1.0, +1.0]           |
|      Written back to SQLite · powers Section 5 charts               |
|                                                                      |
|  [2] Word-based chunking      -> 400 words / 60-word overlap        |
|      step = 340 words · every sentence appears in at least 1 chunk  |
|                                                                      |
|  [3] MiniLM-L6-v2 encoding   -> 384-dimensional vector per chunk    |
|      Loaded once with @lru_cache · ~50ms per chunk on CPU           |
+--------------------------+-------------------------------------------+
                           |
+--------------------------v-------------------------------------------+
|  LAYER 4 - CHROMADB VECTOR STORE                             Task 2  |
|                                                                      |
|  HNSW cosine index · 384 dimensions · grows with each pipeline run  |
|  metadata: source_name | source_type | url | sentiment | chunk_idx  |
|                                                                      |
|  Retrieval: hybrid_search(query, k=12)                              |
|  [1] semantic_search() -> ChromaDB cosine -> top 36 candidates      |
|  [2] BM25Okapi over candidates -> keyword scores                    |
|  [3] RRF fusion (k=60) -> ranked top 12                             |
+--------------------------+-------------------------------------------+
                           |
+--------------------------v-------------------------------------------+
|  LAYER 5 - AI CEO AGENT                               Tasks 4, 5, 6 |
|                                                                      |
|  5 sequential RAG -> LLM calls with 15s inter-task delays:          |
|                                                                      |
|  [1] analyse_opportunities()   -> 3 opportunities  impact+evidence  |
|  [2] analyse_risks()           -> 3 risks   severity+mitigation     |
|  [3] analyse_trends()          -> 4 trends  relevance to NVIDIA     |
|  [4] generate_recommendations()-> 4 actions 3-part risk assessment  |
|  [5] generate_ceo_briefing()   -> what happened / why / what next   |
|                                                                      |
|  LLM: Groq llama-3.1-8b-instant (open-source / free tier)          |
|  Output: typed Python dataclasses -> JSON -> Streamlit session state |
+--------------------------+-------------------------------------------+
                           |
+--------------------------v-------------------------------------------+
|  STREAMLIT DASHBOARD                                  Deliverable 2  |
|                                                                      |
|  3 tabs · 7 sections · Plotly charts · CEO chat (live RAG Q&A)      |
+----------------------------------------------------------------------+
```

---

## 4. Data Flow Diagram

```
User runs: python pipeline.py --all --report
|
+-> collector.py::run_collection()
|     |
|     +-> collect_rss()
|     |     +-- feedparser.parse(feed_url)           # fetch XML
|     |     +-- clean_html(summary)                  # strip HTML
|     |     +-- fetch_url(url) if summary short      # get full article
|     |     +-- is_nvidia_relevant(text, type)        # keyword filter
|     |     +-- save_article(conn, article)           # MD5 -> SQLite
|     |
|     +-> collect_newsapi()                          # 8 queries x 20 articles
|     |     +-- same pipeline as RSS
|     |
|     +-> collect_wikipedia()
|           +-- GET /api/rest_v1/page/summary/{topic}
|           +-- save_article(conn, article)           # source_type="research"
|
+-> processor.py::run_processing()
|     |
|     +-- get_embed_model()                          # load MiniLM (cached)
|     +-- get_collection()                           # open ChromaDB
|     |
|     +-- for each unembedded article:
|           +-- compute_sentiment(title + content)   # VADER -> float
|           +-- UPDATE articles SET sentiment_score   # write to SQLite
|           +-- chunk_text(full_text)                # 400w / 60w overlap
|           +-- model.encode(chunk)                  # -> 384-dim vector
|           +-- collection.upsert(ids, docs,         # batch -> ChromaDB
|                                 embeddings, meta)
|
+-> ceo_agent.py::run_intelligence_engine()
      |
      +-- analyse_opportunities()
      |     +-- hybrid_search("NVIDIA growth sovereign AI...") -> 12 chunks
      |     +-- _call_llm_safe(prompt + context)               -> JSON
      |     +-- [Opportunity(title, impact, confidence, evidence), ...]
      |
      +-- _task_pause(15s)
      |
      +-- analyse_risks()           # same pattern, different query
      +-- _task_pause(15s)
      +-- analyse_trends()          # same pattern, different query
      +-- _task_pause(15s)
      |
      +-- generate_recommendations(opportunities, risks)
      |     +-- hybrid_search("NVIDIA strategic priorities...")
      |     +-- LLM receives: opps + risks summary + live context
      |     +-- [Recommendation(text, priority, evidence,
      |                          expected_impact, RiskAssessment), ...]
      |
      +-- _task_pause(15s)
      |
      +-- generate_ceo_briefing(opps, risks, recs)
            +-- LLM receives all previous outputs as structured text
            +-- {what_happened, why_it_matters,
                 what_to_do_next, executive_summary}

                          |
                          v

User runs: streamlit run app.py
|
+-> app.py
      +-- Sidebar    -> pipeline buttons (collect / process / analyse)
      +-- Tab 1      -> Section 1 (Company Overview)
      |                 Section 2 (Market Intelligence - 6 feed tabs)
      +-- Tab 2      -> Section 3 (Opportunities)
      |                 Section 4 (Risks)
      |                 Section 5 (Sentiment Analysis)
      |                 Section 6 (Recommendations)
      |                 Section 7 (CEO Briefing)
      +-- Tab 3      -> CEO Advisor chat
                          +-- hybrid_search(question, k=8)
                          +-- _call_llm_safe(context + question)
                          +-- answer + clickable source chips
```

---

## 5. Technology Stack

### Core Libraries

| Category | Library | Version | Purpose |
|----------|---------|---------|---------|
| Dashboard | `streamlit` | >=1.35 | Reactive web UI — all 7 sections and CEO chat |
| Charts | `plotly` | >=5.22 | Gauge, donut, horizontal bar, scatter charts |
| Data frames | `pandas` | >=2.2 | SQLite to DataFrame for filtering and charting |
| RSS parsing | `feedparser` | >=6.0 | Handles RSS 1.0, RSS 2.0, and Atom automatically |
| HTTP | `requests` | >=2.31 | Wikipedia REST API, NewsAPI, LLM API calls |
| HTML cleaning | `beautifulsoup4` + `lxml` | >=4.12 | Strip HTML tags from RSS article bodies |
| Sentiment | `vaderSentiment` | >=3.3 | Rule-based VADER — fast, no GPU, news-calibrated |
| Embeddings | `sentence-transformers` | >=3.0 | all-MiniLM-L6-v2 — 384-dim vectors, CPU-fast |
| Vector DB | `chromadb` | >=0.5 | Persistent HNSW cosine index, local, no server |
| BM25 | `rank_bm25` | >=0.2 | BM25Okapi lexical ranking for hybrid search |
| Groq client | `groq` | >=0.9 | Free-tier cloud API for llama-3.1-8b-instant |
| Ollama | `ollama` | >=0.2 | Local runner for qwen3:8b (offline capable) |
| Config | `python-dotenv` | >=1.0 | Load API keys from .env file |
| Testing | `pytest` | >=8.2 | 19 unit tests across processing and collector |

### Language Models — All Open-Source

| Backend | Model | Access | Speed | Licence |
|---------|-------|--------|-------|---------|
| Groq (default) | llama-3.1-8b-instant | Free API key | ~0.5s | Llama 3 Community |
| Ollama (local) | qwen3:8b | Download once | ~10-30s | Apache 2.0 |
| HuggingFace | Mistral-7B-Instruct-v0.3 | Free API key | ~60s | Apache 2.0 |

No paid commercial APIs are used. OpenAI, Anthropic, and Google Gemini are
explicitly excluded per the examination specification.

### Storage

| Store | Technology | Purpose |
|-------|-----------|---------|
| Raw articles | SQLite 3 (articles.db) | Structured metadata, full text, sentiment scores |
| Vector index | ChromaDB (chroma_db/) | 384-dim embeddings, HNSW cosine index |

### Data Sources

| Type | Sources | Count |
|------|---------|-------|
| Company | NVIDIA Newsroom, NVIDIA Blog, SEC EDGAR 8-K | 3 |
| News | TechCrunch, Reuters, BBC, The Verge, Ars Technica, MIT Tech Review, Wired, ZDNet, The Register, Forbes, VentureBeat, Business Insider | 12 |
| News real-time | Google News RSS (NVIDIA, Jensen Huang, NVDA stock, AMD vs NVIDIA) | 4 |
| Industry | Tom's Hardware, AnandTech, ExtremeTech, NotebookCheck, Digital Trends | 5 |
| Research | IEEE Spectrum, SemiAnalysis | 2 |
| Community | Reddit r/nvidia, r/MachineLearning, r/hardware, r/artificial (via RSS), Hacker News | 5 |
| Research background | Wikipedia REST API | 40 topics |
| News optional | NewsAPI | 8 targeted queries |

---

## 6. Design Decisions

### 6.1 Why Two Databases — SQLite and ChromaDB?

SQLite answers structured queries — filter by source type, sort by date, group by source.
ChromaDB answers semantic queries — find the 12 chunks most similar in meaning to a question.

These are fundamentally different query types. SQLite has no vector similarity index.
ChromaDB has no SQL GROUP BY or ORDER BY. Using both gives the dashboard full power:
SQLite for the news feed tabs and sentiment charts, ChromaDB for all RAG retrieval.

### 6.2 Why VADER Over a Transformer for Sentiment?

VADER (Valence Aware Dictionary and Sentiment Reasoner, Hutto & Gilbert 2014) was
calibrated specifically for news and social media text. Processing hundreds of articles takes
under 1 second on CPU. A transformer model like FinBERT would take over 2 minutes for
the same articles and requires a GPU for acceptable speed. At this scale, VADER's
accuracy is sufficient and the speed advantage is decisive.

Thresholds: >= +0.05 positive, <= -0.05 negative, between is neutral.
These are the original empirically-determined thresholds from the VADER paper.

### 6.3 Why Hybrid Search Over Pure Semantic or Keyword?

Tech news contains precise named entities — H100, Blackwell B200, NVLink, TSMC N3P —
that are exact tokens. Pure cosine embeddings may rank a generic "GPU supply chain"
chunk above a specific "H100 yield constraint" chunk for the query "H100 supply problems".
BM25 catches exact keyword matches but misses semantic synonyms entirely.

Hybrid search using Reciprocal Rank Fusion combines both:

    RRF_score = 1 / (cosine_rank + 60) + 1 / (bm25_rank + 60)

The constant 60 (Cormack, Clarke & Buettcher, 2009) prevents any single top-ranked
result from dominating the fusion. This is the standard value validated across
multiple retrieval benchmarks in the original RRF paper.

### 6.4 Why Overlapping Chunks?

A sentence straddling a non-overlapping chunk boundary is split across two vectors —
neither chunk contains the complete sentence, making it potentially unretrievable.
With 60-word overlap (step = 400 - 60 = 340), every sentence is guaranteed to appear
completely within at least one chunk.

### 6.5 Why JSON-Forced LLM Output?

Free-text LLM responses are unpredictable and fragile to parse. By enforcing a strict
JSON schema in the system prompt, the output maps directly to typed Python dataclasses
(Opportunity, Risk, Recommendation, RiskAssessment). This enables the dashboard to
render confidence bars, severity badges, and evidence links without string parsing.

The _parse_json() function implements a three-tier fallback — direct parse, regex
extraction, empty dict — so the system never crashes on malformed output.

### 6.6 Why the src/ Package Layout?

The src/nvidia_agent/ layout prevents accidental imports from the project root during
testing and follows modern Python packaging standards (PEP 517/518). Entry point scripts
at the root add src/ to sys.path at startup using os.path manipulation, making the
package importable without installation.

### 6.7 Why Exponential Backoff?

The Groq free tier enforces tokens-per-minute and requests-per-minute limits. Five
analysis tasks in sequence can hit these limits. Exponential backoff: 30s, 60s, 120s,
240s, 480s, 960s. Plus 15-second inter-task delays between all 5 analysis tasks.

### 6.8 Why Reddit via RSS Instead of OAuth API?

Reddit changed their API policy in June 2023 and now requires OAuth authentication.
Reddit's public RSS feeds still work without authentication and provide the same
content. Four subreddits are collected via RSS: r/nvidia, r/MachineLearning,
r/hardware, r/artificial — all stored with source_type="community".

---

## 7. AI Pipeline

### Overview

The AI pipeline is a 5-task RAG (Retrieval-Augmented Generation) system. Each task
retrieves relevant chunks from ChromaDB, injects them as numbered evidence sources
into a structured JSON prompt, and calls the LLM to produce typed strategic intelligence.

```
Task 1: analyse_opportunities()      -> 3 opportunities with evidence
Task 2: analyse_risks()              -> 3 risks with evidence
Task 3: analyse_trends()             -> 4 trends with NVIDIA relevance
        [Tasks 1-3 are independent]

Task 4: generate_recommendations()   -> receives outputs of Tasks 1+2
        -> 4 prioritised CEO actions with 3-part risk assessment

Task 5: generate_ceo_briefing()      -> receives outputs of Tasks 1+2+4
        -> executive summary answering what/why/what-next
```

### Task Pattern — The RAG Loop

Every analysis task follows the same four-step pattern:

```
Step 1 RETRIEVE:
  hybrid_search(domain_specific_query, k=12)
  -> BM25 + cosine + RRF -> top 12 chunks from ChromaDB

Step 2 AUGMENT:
  Format chunks as numbered context:
  [SOURCE 1] NVIDIA Newsroom -- Title
  Relevance: 0.91 | Sentiment: 0.45 (positive)
  <700 chars of chunk text>
  URL: https://...

Step 3 GENERATE:
  _call_llm_safe(system_prompt + augmented_prompt)
  -> Groq llama-3.1-8b-instant
  -> JSON array or object

Step 4 PARSE AND TYPE:
  _parse_json(response) -> dict / list
  -> Opportunity / Risk / Trend / Recommendation dataclass
  -> Evidence objects from top 3 retrieved chunks
```

### Retrieval Query Focus per Task

| Task | Query Focus |
|------|-------------|
| Opportunities | growth, new markets, AI demand, sovereign AI, partnerships, robotics |
| Risks | AMD Intel competition, export controls, China, supply chain, TSMC, Huawei |
| Trends | inference shift, custom silicon, enterprise AI adoption, edge computing |
| Recommendations | strategic priorities, CUDA moat, AI infrastructure, market leadership |
| CEO Briefing | synthesised from previous task outputs — no new retrieval |

### Evidence Chain — Task 6 Compliance

Every Opportunity, Risk, and Recommendation contains an evidence list with up to
3 sources. Each Evidence object:

```python
@dataclass
class Evidence:
    source:    str    # e.g. "NVIDIA Newsroom"
    excerpt:   str    # first 200 chars of the retrieved chunk
    url:       str    # clickable link to original article
    sentiment: float  # VADER compound score of this chunk
```

This closes the RAG loop — every AI-generated finding is traceable to a specific,
dated, source-attributed article displayed in the dashboard with a clickable link.

### Three-Dimensional Risk Assessment — Task 6

Every recommendation includes a RiskAssessment with three distinct dimensions:

```python
@dataclass
class RiskAssessment:
    financial:   str  # capital risk and ROI uncertainty
    operational: str  # execution difficulty and team requirements
    strategic:   str  # competitive positioning risk if recommendation is wrong
```

### Rate Limit Management

```
analyse_opportunities()     [wait 15s]
analyse_risks()             [wait 15s]
analyse_trends()            [wait 15s]
generate_recommendations()  [wait 15s]
generate_ceo_briefing()
```

All LLM calls use _call_llm_safe() which implements exponential backoff on HTTP 429
responses with optional retry-after header parsing.

### CEO Chat — On-Demand RAG

Beyond the batch analysis, the CEO Advisor tab provides real-time Q&A:

```
CEO question
-> hybrid_search(question, k=8) -> 8 most relevant chunks
-> _call_llm_safe(context + question + last 4 chat messages)
-> answer with [1][2][3] source citations
-> clickable source chips in the dashboard
```

---

## 8. Project Structure

```
nvidia_strategic_intelligence/
|
+-- app.py                         # Streamlit dashboard (1,300+ lines)
+-- collector.py                   # Entry point: runs all data collection
+-- processor.py                   # Entry point: embedding and indexing
+-- pipeline.py                    # CLI: --collect / --process / --analyse / --all
|
+-- src/
|   +-- nvidia_agent/
|       +-- config.py              # Central configuration — all constants
|       |
|       +-- collectors/
|       |   +-- base.py            # SQLite setup, MD5 dedup, relevance filter
|       |   +-- rss.py             # RSS/Atom feeds + NewsAPI
|       |   +-- wikipedia.py       # Wikipedia REST API (40 topics)
|       |
|       +-- processing/
|       |   +-- cleaner.py         # HTML stripping, URL fetching
|       |   +-- embedder.py        # Chunking, VADER sentiment, MiniLM encoding
|       |
|       +-- storage/
|       |   +-- vector_store.py    # ChromaDB HNSW + BM25 + RRF hybrid search
|       |
|       +-- agent/
|           +-- llm.py             # LLM routing: Groq / Ollama / HuggingFace
|           +-- models.py          # Typed dataclasses for all AI outputs
|           +-- ceo_agent.py       # 5 RAG tasks + CEO chat + serialisation
|
+-- tests/
|   +-- test_processing.py         # 10 tests: chunking, sentiment, boundaries
|   +-- test_collector.py          # 9 tests: MD5 dedup, relevance filter
|
+-- data/                          # Generated at runtime — excluded from Git
|   +-- articles.db                # SQLite raw article store
|   +-- chroma_db/                 # ChromaDB vector index
|   +-- report.json                # Latest intelligence report (optional)
|
+-- requirements.txt
+-- .env.example
+-- .gitignore
+-- nvidia-logo-green-3840x2160-24758.png
```

---

## 9. Dashboard Sections

| Section | Tab | Data Source | Key Elements |
|---------|-----|-------------|-------------|
| 1 Company Overview | Overview | SQLite COUNT + config | 8 KPI cards, source bar chart |
| 2 Market Intelligence | Overview | SQLite SELECT | 6 tabs: All / Company / News / Competitors / Community / Research |
| 3 Opportunity Monitor | Intelligence | AI report | 3 cards: title, impact, confidence bar, evidence block |
| 4 Risk Monitor | Intelligence | AI report | 3 cards: title, category badge, severity, mitigation, evidence |
| 5 Sentiment Analysis | Intelligence | SQLite sentiment_score | Gauge + donut + news vs public + source bar chart |
| 6 Strategic Recommendations | Intelligence | AI report | 4 priority-sorted cards: action, impact, 3-part risk, evidence |
| 7 CEO Briefing | Intelligence | AI report | Full-width summary + what happened / why / what next |
| CEO Advisor | CEO Advisor | Live RAG | Chat with 10 suggested questions + source chips |

---

## 10. Configuration Reference

All runtime parameters live in `src/nvidia_agent/config.py`.
API keys and backend selection live in `.env`.

### .env Variables

```bash
# Required for default Groq backend
GROQ_API_KEY=your_groq_api_key_here

# Optional additional news source
NEWS_API_KEY=your_newsapi_key_here

# LLM backend selection (default: groq)
LLM_BACKEND=groq              # groq | ollama | huggingface

# Override model names (optional)
GROQ_MODEL=llama-3.1-8b-instant
OLLAMA_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434

# HuggingFace (if LLM_BACKEND=huggingface)
HF_API_KEY=your_hf_token_here
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3

# Logging level
LOG_LEVEL=INFO
```

### Key Pipeline Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| MAX_ARTICLES_PER_SOURCE | 15 | Articles per RSS feed per run |
| CHUNK_SIZE | 400 | Words per embedding chunk |
| CHUNK_OVERLAP | 60 | Shared words between consecutive chunks |
| TOP_K_RETRIEVAL | 12 | Chunks retrieved per RAG query |
| LLM_TEMPERATURE | 0.3 | LLM creativity (0 = deterministic) |
| LLM_MAX_TOKENS | 2048 | Maximum response length in tokens |

### Running Tests

```bash
pytest tests/ -v
```

Expected: 19 tests passing — 10 processing tests, 9 collector tests.

---

## 11. Examination Compliance

| Requirement | Implementation | Status |
|-------------|---------------|--------|
| >= 100 documents | 100+ articles minimum from 31 feeds + 40 Wikipedia topics | PASS |
| >= 3 independent sources | 5 source types: company, news, research, industry, community | PASS |
| Automatic collection | python pipeline.py --collect | PASS |
| Knowledge repository | SQLite + ChromaDB | PASS |
| Clean / dedup / embed | cleaner.py + MD5 + embedder.py MiniLM | PASS |
| Opportunities / Risks / Trends | 3 separate RAG->LLM tasks, typed dataclasses | PASS |
| AI CEO Agent | 5 tasks: analyse + reason + prioritise + recommend + justify | PASS |
| Evidence-based recommendations | Evidence dataclass on every finding, clickable links | PASS |
| Risk assessment 3 dimensions | RiskAssessment: financial + operational + strategic | PASS |
| Open-source LLM only | llama-3.1-8b (Groq) / qwen3:8b (Ollama) / Mistral-7B (HF) | PASS |
| Embedding model | all-MiniLM-L6-v2 (exam recommended list) | PASS |
| RAG + Semantic + Hybrid Search | semantic_search() + hybrid_search() BM25+cosine+RRF | PASS |
| Streamlit dashboard | 3 tabs, 7 sections, CEO chat | PASS |
| Section 1 Company Overview | Name, industry, documents, sources, timestamp | PASS |
| Section 2 Market Intelligence | Recent news, competitors, emerging tech, announcements | PASS |
| Section 3 Opportunity Monitor | Title, impact level, evidence, confidence score | PASS |
| Section 4 Risk Monitor | Title, category, severity, evidence, confidence score | PASS |
| Section 5 Sentiment Analysis | News sentiment, public sentiment, visualisations | PASS |
| Section 6 Recommendations | Recommendation, priority, evidence, impact, risk level | PASS |
| Section 7 CEO Briefing | What happened, why it matters, what to do next | PASS |
| Unit tests | 19 tests: chunking, sentiment, deduplication, relevance | PASS |
| Architecture documentation | System diagram, data flow, stack, design decisions, AI pipeline | PASS |

---

*NVIDIA Strategic Intelligence Agent — NLP Final Examination — SRH University Heidelberg — June 2026*
