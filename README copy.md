# NVIDIA Strategic Intelligence Agent

> **NLP Final Examination — M.Sc. Data Science & AI**
> SRH University Heidelberg · June 2026
> Autonomous ReAct agent for CEO-level strategic decision-making

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Quick Start](#2-quick-start)
3. [System Architecture](#3-system-architecture)
4. [Data Flow](#4-data-flow)
5. [Technology Stack](#5-technology-stack)
6. [Design Decisions](#6-design-decisions)
7. [The Autonomous Agent](#7-the-autonomous-agent)
8. [Project Structure](#8-project-structure)
9. [Dashboard Sections](#9-dashboard-sections)
10. [Configuration Reference](#10-configuration-reference)
11. [Examination Compliance](#11-examination-compliance)

---

## 1. Project Overview

The NVIDIA Strategic Intelligence Agent is a **fully autonomous ReAct agent** that
transforms live public information into executive-level strategic intelligence
for NVIDIA Corporation.

The system continuously collects news, company announcements, research, and
community discussions. It processes and embeds this content into a hybrid
vector + keyword knowledge base, then runs an **autonomous agent** that *plans
its own research strategy*, retrieves evidence, extracts findings, validates them
against the evidence using SBERT cosine grounding, and produces a CEO briefing.

**Key distinction from a normal RAG pipeline:** this is not a fixed 5-step
sequence. The agent runs a true ReAct loop where, at every step, the LLM itself
decides which tool to call and what query to run, based on the current state of
its memory. The loop self-terminates the moment it has met the minimum
thresholds for opportunities, risks, and trends.

### Key Metrics

| Metric | Value |
|--------|-------|
| Articles collected | Grows with each pipeline run (100+ minimum required) |
| Data source types | 5 — Company, News, Research, Industry, Community |
| Active RSS feeds | 31 |
| Wikipedia background topics | 40 |
| Vector chunks in ChromaDB | Grows with articles (≈ 2-3 chunks per article) |
| Embedding dimensions | 384 (`all-MiniLM-L6-v2`) |
| Primary LLM (agent / engine) | `qwen3:14b` running locally on Ollama (Apache 2.0) |
| CEO chat LLM | `llama-3.3-70b-versatile` on Groq (Llama 3 Community Licence) |
| Cloud fallback LLM | `mistralai/Mistral-7B-Instruct-v0.3` (HuggingFace, Apache 2.0) |
| Agent tools | 9 non-LLM programmatic tools |
| Dashboard sections | 7 |
| Unit tests | 19 |

---

## 2. Quick Start

### Prerequisites

- Python 3.11 or 3.12
- Free Groq API key — [console.groq.com](https://console.groq.com)
- 2 GB free disk space (model weights + data)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Vinayybg/NATURAL-LANGUAGE-PROCESSING.git
cd nvidia_strategic_intelligence

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add GROQ_API_KEY=...
```

### Running the Pipeline

```bash
# Stage 1 — Collect live data (~3-5 min)
python pipeline.py --collect

# Stage 2 — Process & embed (~2 min, downloads MiniLM on first run)
python pipeline.py --process

# Stage 3 — Run autonomous agent (~2-3 min)
python pipeline.py --analyse

# Or run all three stages back to back
python pipeline.py --all --report

# Launch the dashboard
streamlit run app.py
```

### LLM Backend Configuration

The default setup is **mixed mode**: the agent runs locally on Ollama (no rate
limits, no token cost during long ReAct loops), and the CEO chat uses Groq for
sub-second responses.

```bash
# Default — mixed mode (in .env):
LLM_BACKEND=ollama        # qwen3:14b runs the agent
CHAT_BACKEND=groq         # llama-3.3-70b runs the chat

# All-cloud mode:
LLM_BACKEND=groq          # everything on Groq llama-3.3-70b
# (unset CHAT_BACKEND)

# All-local mode (fully offline after model pull):
ollama pull qwen3:14b
LLM_BACKEND=ollama        # agent + chat both on qwen3:14b
CHAT_BACKEND=ollama       # or just unset

# HuggingFace Inference API fallback:
LLM_BACKEND=huggingface
HF_API_KEY=<your_token>
```

---

## 3. System Architecture

```
+-------------------------------------------------------------------------+
|              NVIDIA STRATEGIC INTELLIGENCE AGENT                        |
|                   5-Layer Pipeline Architecture                         |
+-------------------------------------------------------------------------+

+----------------------------------------------------------------------+
|  LAYER 1 — LIVE DATA COLLECTION                              Task 1  |
|                                                                      |
|  +-------------+  +-------------+  +-------------+  +-----------+    |
|  | NVIDIA      |  | 28 News &   |  |  Wikipedia  |  |  NewsAPI  |    |
|  | Official    |  | Industry    |  |  REST API   |  | optional  |    |
|  | RSS + SEC   |  | RSS Feeds   |  |  40 topics  |  | 8 queries |    |
|  | EDGAR 8-K   |  | + Google    |  |             |  |           |    |
|  |             |  | News RSS    |  |             |  |           |    |
|  +------+------+  +------+------+  +------+------+  +-----+-----+    |
|                                                                      |
|  feedparser / requests / BeautifulSoup4 / lxml                       |
|  MD5 deduplication / NVIDIA keyword relevance filter                 |
+----------------------------------|-----------------------------------+
                                   |
+----------------------------------v-----------------------------------+
|  LAYER 2 — SQLITE STORAGE                                    Task 2  |
|                                                                      |
|  articles.db   WAL mode   B-tree indexes on source, type, date       |
|                                                                      |
|  id (MD5 PK) | title | content | url | source_name                   |
|  source_type | published | collected | sentiment_score               |
+----------------------------------|-----------------------------------+
                                   |
+----------------------------------v-----------------------------------+
|  LAYER 3 — INFORMATION PROCESSING                            Task 3  |
|                                                                      |
|  [1] VADER sentiment        → compound in [-1.0, +1.0]               |
|      Written back to SQLite → powers Section 5 charts                |
|                                                                      |
|  [2] Word-based chunking    → 400 words / 60-word overlap            |
|      step = 340 words       → every sentence appears in ≥ 1 chunk    |
|                                                                      |
|  [3] MiniLM-L6-v2 encoding  → 384-dim vector per chunk               |
|      @lru_cache once        → ~50 ms per chunk on CPU                |
+----------------------------------|-----------------------------------+
                                   |
+----------------------------------v-----------------------------------+
|  LAYER 4 — CHROMADB VECTOR STORE                             Task 2  |
|                                                                      |
|  HNSW cosine index · 384 dimensions · grows with each pipeline run   |
|  metadata: source_name | source_type | url | sentiment | chunk_idx   |
|                                                                      |
|  Retrieval: hybrid_search(query, k=6)                                |
|  [1] semantic_search() → ChromaDB cosine → top 18 candidates         |
|  [2] BM25Okapi over candidates → keyword scores                      |
|  [3] RRF fusion (k=60) → ranked top 6 (or top 12 for chat)           |
+----------------------------------|-----------------------------------+
                                   |
+----------------------------------v-----------------------------------+
|  LAYER 5 — AUTONOMOUS REACT AGENT                     Tasks 4, 5, 6  |
|                                                                      |
|  PHASE 1: Research Loop (up to MAX_STEPS = 20)                       |
|    ┌─── per-step ───────────────────────────────────────────┐        |
|    │  1. PLAN     LLM decides: search / summarise / conclude │        |
|    │  2. ACT      Execute the chosen tool                    │        |
|    │  3. OBSERVE  Update AgentMemory                         │        |
|    │  4. CHECK    Has the agent met all minimum thresholds? │        |
|    │              YES → break;   NO → next step              │        |
|    └────────────────────────────────────────────────────────┘        |
|                                                                      |
|  PHASE 2: Recommend  → 4 prioritised recs with 3-part risk           |
|  PHASE 3: Validate   → SBERT cosine grounding + LLM critic           |
|  PHASE 4: Brief      → Executive what / why / next                   |
+----------------------------------|-----------------------------------+
                                   |
+----------------------------------v-----------------------------------+
|  STREAMLIT DASHBOARD                                  Deliverable 2  |
|                                                                      |
|  3 tabs · 7 sections · Plotly charts · CEO chat (live RAG Q&A)       |
+----------------------------------------------------------------------+
```

---

## 4. Data Flow

```
User runs: python pipeline.py --all --report
│
├─> collector.py :: run_collection()
│     ├─ collect_rss()        — 31 feeds, feedparser, MD5 dedup, NVIDIA filter
│     ├─ collect_newsapi()    — 8 queries × 20 articles (skipped if no key)
│     └─ collect_wikipedia()  — 40 topics, REST summary endpoint
│
├─> processor.py :: run_processing()
│     ├─ get_embed_model()           — load MiniLM (cached)
│     ├─ get_collection()            — open ChromaDB
│     └─ for every unembedded article:
│         ├─ compute_sentiment()     — VADER compound → SQLite
│         ├─ chunk_text()            — 400-word / 60-word overlap
│         ├─ model.encode()          — 384-dim vector per chunk
│         └─ collection.upsert()     — vectors + metadata → ChromaDB
│
└─> ceo_agent.py :: run_intelligence_engine()
      │
      ├─ PHASE 1 — Research Loop (autonomous ReAct)
      │     while not done and step < MAX_STEPS:
      │       ├─ _plan_step(mem)         # LLM picks tool + query
      │       ├─ _tool_search(...)       # hybrid_search → LLM extracts findings
      │       │  or _tool_summarise(...) # consolidate findings so far
      │       │  or conclude             # only if thresholds met
      │       └─ if mem.has_enough(): break
      │
      ├─ PHASE 2 — _generate_recommendations(mem)
      │     LLM synthesises 4 prioritised actions with 3-part risk assessment
      │
      ├─ PHASE 3 — _validate_recommendations(recs, mem)
      │     ├─ SBERT cosine grounding (objective, non-LLM)
      │     └─ LLM critic — per-rec approved / needs_revision / rejected
      │           (SBERT block CANNOT be overridden by the LLM critic)
      │
      ├─ PHASE 4 — _generate_briefing(mem, recs)
      │     LLM writes: executive_summary / what_happened / why_it_matters /
      │                 what_to_do_next
      │
      └─ _save_outputs() — writes outputs/report.json, agent_report.txt,
                           agent_report.md
                                   |
                                   v
User runs: streamlit run app.py
      ├─ Sidebar  → 3 pipeline buttons (collect / process / analyse)
      ├─ Tab 1    → Section 1 (Company Overview) + Section 2 (Market feed)
      ├─ Tab 2    → Sections 3-7 (Opps / Risks / Sentiment / Recs / Brief)
      └─ Tab 3    → CEO Advisor — live RAG chat (k=8)
```

---

## 5. Technology Stack

### Core Libraries

| Category | Library | Version | Purpose |
|----------|---------|---------|---------|
| Dashboard | `streamlit` | ≥ 1.35 | Reactive web UI — all 7 sections + CEO chat |
| Charts | `plotly` | ≥ 5.22 | Gauge, donut, bar, scatter |
| Data frames | `pandas` | ≥ 2.2 | SQLite → DataFrame for filtering / charts |
| RSS parsing | `feedparser` | ≥ 6.0 | RSS 1.0, RSS 2.0, Atom |
| HTTP | `requests` | ≥ 2.31 | Wikipedia REST, NewsAPI, LLM APIs |
| HTML cleaning | `beautifulsoup4` + `lxml` | ≥ 4.12 | Strip HTML from feed bodies |
| Sentiment | `vaderSentiment` | ≥ 3.3 | Rule-based VADER, news-calibrated |
| Embeddings | `sentence-transformers` | 3.x | `all-MiniLM-L6-v2` 384-dim |
| Vector DB | `chromadb` | ≥ 0.5 | Persistent HNSW cosine, local |
| BM25 | `rank_bm25` | ≥ 0.2 | BM25Okapi lexical ranking |
| Groq client | `groq` | ≥ 0.9 | Free-tier cloud API for Llama 3.3 70B |
| Ollama | `ollama` | ≥ 0.2 | Local runner for qwen3:14b |
| Memory (opt) | `langgraph` | latest | `MemorySaver` checkpointing |
| Config | `python-dotenv` | ≥ 1.0 | Load API keys from `.env` |
| Testing | `pytest` | ≥ 8.2 | 19 unit tests |

### Language Models — All Open-Source

| Role | Backend | Model | Access | Typical Latency | Licence |
|------|---------|-------|--------|-----------------|---------|
| Intelligence engine / agent (primary) | Ollama (local) | `qwen3:14b` | Pull once | ~15-45 s per call | Apache 2.0 |
| CEO Advisor chat | Groq (cloud) | `llama-3.3-70b-versatile` | Free API key | ~1-2 s | Llama 3 Community |
| Fallback | HuggingFace Inference | `mistralai/Mistral-7B-Instruct-v0.3` | Free token | ~60 s | Apache 2.0 |

The split is deliberate: the agent does many sequential LLM calls during its
ReAct loop, so running it locally on Ollama means zero rate-limit pressure and
zero token cost. The CEO chat, by contrast, needs fast single-turn responses,
so it routes to Groq for sub-second latency. This is configured via
`LLM_BACKEND=ollama` + `CHAT_BACKEND=groq` in `.env`.

No paid commercial APIs are used. OpenAI, Anthropic, and Google Gemini are
explicitly excluded per the examination specification.

### Storage

| Store | Technology | Purpose |
|-------|-----------|---------|
| Raw articles | SQLite 3 (`articles.db`) | Structured metadata, full text, sentiment scores |
| Vector index | ChromaDB (`chroma_db/`) | 384-dim embeddings, HNSW cosine index |
| Agent memory | dict + LangGraph `MemorySaver` (optional) | Persists last-run summary across runs |

### Data Sources (31 RSS Feeds + Wikipedia + NewsAPI)

| Type | Sources | Count |
|------|---------|-------|
| Company | NVIDIA Newsroom, NVIDIA Blog, SEC EDGAR 8-K | 3 |
| News | TechCrunch, Reuters, BBC, The Verge, Ars Technica, MIT TR, Wired, ZDNet, The Register, Forbes, VentureBeat, Business Insider | 12 |
| News real-time | Google News RSS (NVIDIA, Jensen Huang, NVDA stock, AMD vs NVIDIA) | 4 |
| Industry | Tom's Hardware, AnandTech, ExtremeTech, NotebookCheck, Digital Trends | 5 |
| Research | IEEE Spectrum, SemiAnalysis | 2 |
| Community | r/nvidia, r/MachineLearning, r/hardware, r/artificial (via RSS), Hacker News | 5 |
| Research background | Wikipedia REST API | 40 topics |
| News optional | NewsAPI | 8 targeted queries |

---

## 6. Design Decisions

### 6.1 Why Two Databases — SQLite and ChromaDB?

SQLite answers structured queries (filter by source type, sort by date, group by
source). ChromaDB answers semantic queries (find the k chunks most similar in
meaning to a question). These are fundamentally different query types; using
both gives the dashboard full power: SQLite for the news feed tabs and sentiment
charts, ChromaDB for all RAG retrieval.

### 6.2 Why VADER over a Transformer for Sentiment?

VADER (Valence Aware Dictionary and Sentiment Reasoner, Hutto & Gilbert 2014)
was calibrated specifically for news and social-media text. It scores hundreds
of articles in under a second on CPU. FinBERT or similar would need a GPU and
take minutes for the same data. At this scale, VADER's accuracy is sufficient
and the speed advantage is decisive. Thresholds (`≥ +0.05` positive,
`≤ −0.05` negative) are the empirically-determined values from the original
VADER paper.

### 6.3 Why Hybrid Search Over Pure Semantic or Pure Keyword?

Tech content contains precise named entities — H100, Blackwell B200, NVLink,
TSMC N3P — that are exact tokens. Pure cosine embeddings can rank a generic
"GPU supply chain" chunk above a specific "H100 yield constraint" chunk for the
query "H100 supply problems". BM25 catches exact keyword matches but misses
semantic synonyms entirely.

Hybrid search using Reciprocal Rank Fusion combines both:

    RRF_score = 1 / (cosine_rank + 60) + 1 / (bm25_rank + 60)

The constant `60` (Cormack, Clarke & Buettcher, 2009) prevents any single
top-ranked result from dominating the fusion. This is the standard value
validated across multiple retrieval benchmarks in the original RRF paper.

### 6.4 Why Overlapping Chunks?

A sentence straddling a non-overlapping chunk boundary is split across two
vectors — neither chunk contains the complete sentence, making it potentially
unretrievable. With 60-word overlap (step = 400 − 60 = 340), every sentence is
guaranteed to appear completely within at least one chunk.

### 6.5 Why JSON-Forced LLM Output?

Free-text LLM responses are unpredictable and fragile to parse. By enforcing a
strict JSON schema in the system prompt, the output maps directly to typed
Python dataclasses (`Opportunity`, `Risk`, `Recommendation`, `RiskAssessment`).
The `_parse_json()` function implements a four-tier fallback (direct parse,
fenced `` ```json `` block, first `{…}` substring, first `[…]` substring) so the
system rarely crashes on malformed output.

### 6.6 Why SBERT Grounding Verification?

The LLM critic in Phase 3 is itself an LLM — it can be sycophantic. To prevent
the agent from approving recommendations that aren't actually supported by the
evidence, the verifier (`verifier_agent.py`) embeds each recommendation and its
attached evidence with SBERT (`all-MiniLM-L6-v2`) and computes cosine
similarity. A recommendation scoring below `QUALITY_THRESHOLD` (default 0.5) is
**force-flagged as `needs_revision`**, and the LLM critic cannot override this
decision. This gives the validation step an objective, programmatic anchor.

### 6.7 Why the `src/` Package Layout?

The `src/nvidia_agent/` layout prevents accidental imports from the project
root during testing and follows modern Python packaging standards (PEP 517/518).
Entry-point scripts at the root (`collector.py`, `processor.py`, `pipeline.py`,
`app.py`) add `src/` to `sys.path` at startup, making the package importable
without `pip install -e .`.

### 6.8 Why Exponential Backoff?

The agent does many sequential LLM calls during the research loop,
recommendation, validation, and briefing phases. In the default mixed-mode
setup the agent runs on local Ollama and never hits a rate limit, but
`_llm_groq()` still implements exponential backoff (10 s, 20 s, 40 s, 80 s,
160 s) because the CEO chat goes through Groq, and so does the agent if the
user switches to all-cloud mode. The retry handler also parses the
`try again in ...s` hint from 429 responses to wait exactly the right amount.
Inter-phase sleeps of 10-12 s add an additional cushion.

### 6.9 Why Reddit via RSS Instead of OAuth?

Reddit changed its API policy in June 2023 and now requires OAuth. Reddit's
public RSS feeds still work without authentication and supply the same content,
so four subreddits (r/nvidia, r/MachineLearning, r/hardware, r/artificial) are
collected via RSS with `source_type="community"`.

---

## 7. The Autonomous Agent

This is the heart of the project — and the part most likely to be examined.
The agent lives in `src/nvidia_agent/agent/ceo_agent.py`.

### Workflow

```
Goal → Plan → [Search → Extract → Decide if enough → repeat if not]
     → Recommend → Validate → Brief
```

This is a **true ReAct loop** (Reason + Act). Every step:

1. **Plan** — `_plan_step()` shows the LLM the current memory state (counts of
   opportunities / risks / trends found so far, queries already run, last
   tool result) and asks it to return JSON: which tool to call next, what
   argument to pass, and one sentence of reasoning.
2. **Act** — Execute the chosen tool. Currently `search`, `summarise`, or
   `conclude`.
3. **Observe** — Update `AgentMemory` with new findings.
4. **Check** — If `mem.has_enough()` is True, break out of the loop and move to
   Phase 2.

### AgentMemory

All state lives in a single dataclass — no globals.

```python
@dataclass
class AgentMemory:
    goal:          str
    company_name:  str
    opportunities: list = field(default_factory=list)
    risks:         list = field(default_factory=list)
    trends:        list = field(default_factory=list)
    summaries:     list = field(default_factory=list)
    steps_taken:   int  = 0
    queries_run:   list = field(default_factory=list)
    reasoning_log: list = field(default_factory=list)
    last_tool_result: str = ""
    all_docs:      list = field(default_factory=list)

    def has_enough(self) -> bool:
        return (len(self.opportunities) >= MIN_OPPORTUNITIES
            and len(self.risks)         >= MIN_RISKS
            and len(self.trends)        >= MIN_TRENDS)
```

### Termination Thresholds

The loop self-terminates when **all three** are met:

| Threshold | Default | Source |
|-----------|---------|--------|
| `MIN_OPPORTUNITIES` | 4 | `config.py` (env-overridable) |
| `MIN_RISKS` | 4 | `config.py` (env-overridable) |
| `MIN_TRENDS` | 3 | `config.py` (env-overridable) |
| `MAX_STEPS` | 20 | Cost guard against runaway API spend |
| `K_CHUNKS` | 6 | Chunks per search (Groq 6000 TPM guard) |

### The 9 AgentTools

These satisfy the "tool usage beyond the LLM" requirement.

| Tool | What it does |
|------|--------------|
| `search` | Hybrid ChromaDB search (BM25 + cosine + RRF). Returns top-k chunks. |
| `deduplicate` | Removes duplicate chunks by content fingerprint. |
| `score_relevance` | Sorts chunks by keyword overlap with the query. |
| `extract_evidence` | Builds structured `Evidence` objects (source, excerpt, url, sentiment). |
| `format_context` | Numbers chunks into LLM-ready `[Doc N]` context. |
| `calculator` | Safe arithmetic via regex-validated `eval`. |
| `summarize` | Extractive summariser (first-N sentence split). |
| `memory_read` / `memory_write` | Persistent key-value memory using LangGraph `MemorySaver` if available, falling back to an in-process dict. |

### Evidence by Doc Index (not rotation)

When the LLM extracts findings, it returns `evidence_doc_indices: [1, 2]`
referring to the numbered documents the agent just formatted into the prompt.
`_attach_evidence_by_index()` then pulls those exact source dicts back from
`mem.all_docs`. This is significantly more reliable than rotating evidence from
a global pool — the LLM is telling the agent **which** chunk supports the
finding, not just **a** chunk that was nearby.

### Validation — Two Stages

```
Stage 1 (objective)    SBERT cosine score(rec, evidence)
                       ├─ score >= 0.5  → pass to Stage 2
                       └─ score <  0.5  → force needs_revision (LLM cannot override)

Stage 2 (LLM critic)   For each rec: approved / needs_revision / rejected + reason
```

This pattern — programmatic verification gating an LLM-as-judge — prevents the
LLM from rubber-stamping its own outputs.

### CEO Chat — On-Demand RAG

Beyond the batch pipeline, the CEO Advisor tab provides real-time Q&A:

```
CEO question
→ hybrid_search(question, k=8)
→ _call_llm_safe(context + question + last 4 chat messages)
→ answer with [1][2][3] source citations
→ clickable source chips in the dashboard
```

`_call_llm_safe` is called from `app.py`. It honours the `CHAT_BACKEND` env
variable, so you can run the heavy agent on local Ollama and still serve fast
chat replies from Groq.

---

## 8. Project Structure

```
nvidia_strategic_intelligence/
│
├─ app.py                         # Streamlit dashboard (1,333 lines)
├─ collector.py                   # Entry point — runs all data collection
├─ processor.py                   # Entry point — embedding & indexing
├─ pipeline.py                    # CLI — --collect / --process / --analyse / --all
│
├─ src/nvidia_agent/
│   ├─ __init__.py
│   ├─ config.py                  # Central configuration — all constants
│   │
│   ├─ collectors/
│   │   ├─ base.py                # SQLite setup, MD5 dedup, NVIDIA relevance filter
│   │   ├─ rss.py                 # 31 RSS feeds + NewsAPI
│   │   └─ wikipedia.py           # Wikipedia REST API (40 topics)
│   │
│   ├─ processing/
│   │   ├─ cleaner.py             # HTML stripping, URL fetching
│   │   └─ embedder.py            # Chunking, VADER sentiment, MiniLM encoding
│   │
│   ├─ storage/
│   │   └─ vector_store.py        # ChromaDB HNSW + BM25 + RRF hybrid search
│   │
│   ├─ agent/
│   │   ├─ llm.py                 # LLM routing — Groq / Ollama / HuggingFace
│   │   ├─ models.py              # Typed dataclasses (Opportunity, Risk, …)
│   │   ├─ verifier_agent.py      # SBERT grounding verifier (non-LLM)
│   │   └─ ceo_agent.py           # Autonomous ReAct agent (1,242 lines)
│   │
│   └─ tests/
│       ├─ test_processing.py     # 10 tests: chunking, sentiment, boundaries
│       └─ test_collector.py      # 9 tests: MD5 dedup, NVIDIA relevance
│
├─ data/                          # Generated at runtime (gitignored)
│   ├─ articles.db                # SQLite raw article store
│   └─ chroma_db/                 # ChromaDB vector index
│
├─ outputs/                       # Generated at runtime
│   ├─ report.json                # Latest intelligence report (full JSON)
│   ├─ agent_report.txt           # Human-readable execution log
│   └─ agent_report.md            # Markdown report with all sections
│
├─ scripts/
│   └─ setup.sh                   # One-command environment bootstrap
│
├─ requirements.txt
├─ pyproject.toml
├─ .env.example
├─ .gitignore
├─ nvidia-logo-green-3840x2160-24758.png
└─ README.md
```

### File-to-File Dependency Map

```
                       ┌──────────────────────┐
                       │      config.py        │  (constants, .env loader)
                       └──────────┬────────────┘
                                  │ imported by ALL
       ┌──────────────────────────┼──────────────────────────┐
       │                          │                          │
┌──────v────┐            ┌────────v────────┐         ┌───────v────┐
│ collectors│            │   processing    │         │  storage   │
│  base.py  │            │ cleaner.py      │         │ vector_    │
│  rss.py ──┼────────────►│ embedder.py    │◄────────│  store.py  │
│ wikipedia │            └─────────────────┘         └───────┬────┘
└─────┬─────┘                                                │
      │                                                       │
      │     ┌────────────────────────────────────────────────┘
      │     │
┌─────v─────v─────────────────┐
│        agent/                │
│  ┌───────────────────────┐   │
│  │ llm.py / models.py    │   │ ← typed dataclasses + backend routing
│  │ verifier_agent.py     │   │ ← SBERT grounding
│  │ ceo_agent.py ─────────┼───┼─► uses storage.hybrid_search,
│  └───────────────────────┘   │   processing.embedder
└──────────────────────────────┘
              ▲          ▲
              │          │
   ┌──────────┘          └──────────┐
   │                                │
┌──┴───────┐  ┌──────────┐  ┌───────┴──┐  ┌──────────┐
│collector.│  │processor.│  │pipeline. │  │  app.py  │
│   py     │  │   py     │  │   py     │  │(Streamlit)│
└──────────┘  └──────────┘  └──────────┘  └──────────┘
   (root-level entry points; each does sys.path.insert("src"))
```

---

## 9. Dashboard Sections

| Section | Tab | Data Source | Key Elements |
|---------|-----|-------------|-------------|
| 1 Company Overview | Overview | SQLite COUNT + config | 8 KPI cards, source bar chart |
| 2 Market Intelligence | Overview | SQLite SELECT | 6 sub-tabs: All / Company / News / Competitors / Community / Research |
| 3 Opportunity Monitor | Intelligence | Agent report | Cards: title, impact, confidence bar, evidence block |
| 4 Risk Monitor | Intelligence | Agent report | Cards: title, category badge, severity, mitigation, evidence |
| 5 Sentiment Analysis | Intelligence | SQLite `sentiment_score` | Gauge + donut + news vs public + per-source bar chart |
| 6 Strategic Recommendations | Intelligence | Agent report | Priority-sorted cards: action, impact, 3-part risk, SBERT score, validation verdict, evidence |
| 7 CEO Briefing | Intelligence | Agent report | Executive summary + what happened / why / what next |
| CEO Advisor | CEO Advisor | Live RAG | Chat with 10 suggested questions + clickable source chips |

---

## 10. Configuration Reference

All runtime parameters live in `src/nvidia_agent/config.py`. API keys and
backend choice live in `.env`.

### `.env` Variables

```bash
# ── Default mixed-mode setup ─────────────────────────────────
# Agent runs locally on Ollama, chat uses Groq for fast responses
LLM_BACKEND=ollama           # primary — drives the intelligence engine
CHAT_BACKEND=groq            # CEO Advisor chat override

# Required for the Groq-backed chat (and any all-cloud mode)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Required for the Ollama-backed agent
OLLAMA_MODEL=qwen3:14b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_THINK=false           # disable qwen3 chain-of-thought (breaks JSON parse)

# Optional additional news source
NEWS_API_KEY=your_newsapi_key_here

# HuggingFace fallback (only if LLM_BACKEND=huggingface)
HF_API_KEY=your_hf_token_here
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3

# Agent tuning
MAX_STEPS=20
K_CHUNKS=6
MIN_OPPORTUNITIES=4
MIN_RISKS=4
MIN_TRENDS=3
QUALITY_THRESHOLD=0.5

# Logging
LOG_LEVEL=INFO
```

### Key Pipeline Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `MAX_ARTICLES_PER_SOURCE` | 15 | Articles per RSS feed per run |
| `CHUNK_SIZE` | 400 | Words per embedding chunk |
| `CHUNK_OVERLAP` | 60 | Shared words between consecutive chunks |
| `TOP_K_RETRIEVAL` | 12 | Default chunks per RAG query (chat uses k=8, agent uses K_CHUNKS=6) |
| `LLM_TEMPERATURE` | 0.3 | LLM creativity (0 = deterministic) |
| `LLM_MAX_TOKENS` | 2048 | Maximum response length in tokens |
| `MAX_STEPS` | 20 | Max research loop iterations |
| `K_CHUNKS` | 6 | Chunks per agent search (Groq TPM guard) |
| `MIN_OPPORTUNITIES` | 4 | Loop cannot conclude below this |
| `MIN_RISKS` | 4 | Loop cannot conclude below this |
| `MIN_TRENDS` | 3 | Loop cannot conclude below this |
| `QUALITY_THRESHOLD` | 0.5 | SBERT cosine block threshold |

### Running Tests

```bash
pytest src/nvidia_agent/tests -v
```

Expected: 19 tests pass — 10 processing tests, 9 collector tests.

---

## 11. Examination Compliance

| Requirement | Implementation | Status |
|-------------|---------------|--------|
| ≥ 100 documents | 100+ articles from 31 feeds + 40 Wikipedia topics | PASS |
| ≥ 3 independent sources | 5 source types: company, news, research, industry, community | PASS |
| Automatic collection | `python pipeline.py --collect` | PASS |
| Knowledge repository | SQLite (raw) + ChromaDB (vectors) | PASS |
| Clean / dedup / embed | `cleaner.py` + MD5 + `embedder.py` MiniLM | PASS |
| Opportunities / Risks / Trends | Extracted incrementally each search; typed dataclasses | PASS |
| **AI agent with planning** | `_plan_step()` runs every iteration; LLM picks tool + query | PASS |
| **Autonomous decision-making** | ReAct loop self-terminates when thresholds met | PASS |
| **Tool usage beyond LLM** | 9 `AgentTools`: search, deduplicate, score_relevance, extract_evidence, format_context, calculator, summarize, memory_read, memory_write | PASS |
| **Validation before presenting** | SBERT cosine grounding + LLM critic, two-stage | PASS |
| Evidence-based recommendations | `Evidence` dataclass on every finding, attached by doc index | PASS |
| 3-dimensional risk assessment | `RiskAssessment`: financial / operational / strategic | PASS |
| Open-source LLM only | qwen3:14b (Ollama, primary) / Llama 3.3 70B (Groq, chat) / Mistral-7B (HF) | PASS |
| Embedding model | `all-MiniLM-L6-v2` (exam-recommended list) | PASS |
| RAG + Semantic + Hybrid Search | `semantic_search()` + `hybrid_search()` BM25+cosine+RRF | PASS |
| Streamlit dashboard | 3 tabs, 7 sections, CEO chat | PASS |
| Section 1 Company Overview | Name, industry, documents, sources, timestamp | PASS |
| Section 2 Market Intelligence | Recent news, competitors, emerging tech, announcements | PASS |
| Section 3 Opportunity Monitor | Title, impact, evidence, confidence score | PASS |
| Section 4 Risk Monitor | Title, category, severity, evidence, confidence score | PASS |
| Section 5 Sentiment Analysis | News / public sentiment, visualisations | PASS |
| Section 6 Recommendations | Recommendation, priority, evidence, impact, 3-part risk, validation verdict | PASS |
| Section 7 CEO Briefing | What happened, why it matters, what to do next | PASS |
| Unit tests | 19 tests: chunking, sentiment, dedup, relevance | PASS |
| Architecture documentation | This README + inline docstrings throughout | PASS |

---

*NVIDIA Strategic Intelligence Agent — NLP Final Examination — SRH University Heidelberg — June 2026*
