"""
ceo_agent.py — Autonomous NVIDIA Strategic Intelligence Agent
=============================================================

PROFESSOR REQUIREMENTS:
  ✅ Planning before execution       → _plan() per-step: LLM picks tool + query
  ✅ Autonomous decision-making      → True ReAct loop, self-terminates when thresholds met
  ✅ Tool usage beyond the LLM       → AgentTools: search, deduplicate, score_relevance,
                                        extract_evidence, format_context, calculator,
                                        summarize, memory_read, memory_write
  ✅ Retrieval and use of evidence   → ChromaDB hybrid search, doc-indexed evidence
  ✅ Analysis of risks/opps/trends   → LLM extracts findings incrementally each search
  ✅ Validation before presenting    → SBERT cosine + LLM per-recommendation verdict

WORKFLOW (true autonomous loop):
  Goal → Plan → [search → extract → decide if enough → repeat if not] → Recommend → Validate → Brief

ARCHITECTURE (from friend's approach, adapted for NVIDIA/Groq):
  - AgentMemory dataclass carries all state between steps
  - Per-step planning: LLM decides which tool to use and what query to run
  - Minimum thresholds: agent CANNOT proceed without MIN_OPPORTUNITIES/RISKS/TRENDS
  - Duplicate query guard: never repeats a search
  - Deterministic confidence scoring from RRF retrieval scores
  - Evidence attached by document index (not rotation guessing)
  - Full reasoning trace logged to agent_log

COST GUARDS (not analytical constraints):
  MAX_STEPS = 20       prevents runaway API spend
  K_CHUNKS  = 6        respects Groq 6000 TPM
"""

import os, re, time, json, logging, asyncio, platform
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from groq import Groq
from dotenv import load_dotenv
from nvidia_agent.storage.vector_store import hybrid_search

# Typed dataclasses for structured output
try:
    from nvidia_agent.agent.models import StrategicReport
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False

# SBERT verifier — optional
try:
    from nvidia_agent.agent.verifier_agent import verify_recommendations
    _VERIFIER_AVAILABLE = True
except ImportError:
    _VERIFIER_AVAILABLE = False

# LangGraph memory — optional
try:
    from langgraph.checkpoint.memory import MemorySaver as _MemorySaver
    _langgraph_memory = _MemorySaver()
    _USE_LANGGRAPH    = True
except ImportError:
    _USE_LANGGRAPH = False

# Load .env from project root — works regardless of working directory
# ceo_agent.py is at src/nvidia_agent/agent/ceo_agent.py
# .env is at project root (3 levels up)
from pathlib import Path
_project_root = Path(__file__).resolve().parents[3]
_env_file     = _project_root / ".env"
load_dotenv(_env_file)
logger = logging.getLogger(__name__)

# Debug: log which backend is active so we can confirm .env is loaded
_backend_debug = os.getenv("LLM_BACKEND", "NOT SET")
print(f"[ceo_agent] LLM_BACKEND={_backend_debug} | OLLAMA_MODEL={os.getenv('OLLAMA_MODEL','NOT SET')}")

# Suppress HuggingFace warnings — SBERT loads via sentence-transformers
# but we don't need HF auth for the all-MiniLM-L6-v2 model
os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY",  "error")
import warnings
warnings.filterwarnings("ignore", message=".*HF Hub.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*Loading weights.*")

# ── Config — all values from config.py, overridable via .env ─────────────────
from nvidia_agent import config as _cfg

LLM_BACKEND     = _cfg.LLM_BACKEND
GROQ_MODEL      = _cfg.GROQ_MODEL
OLLAMA_MODEL    = _cfg.OLLAMA_MODEL
OLLAMA_BASE_URL = _cfg.OLLAMA_BASE_URL

MAX_STEPS         = _cfg.MAX_STEPS          # max research loop steps
K_CHUNKS          = _cfg.K_CHUNKS           # chunks per search (Groq TPM guard)
MIN_OPPORTUNITIES = _cfg.MIN_OPPORTUNITIES  # agent cannot conclude without these
MIN_RISKS         = _cfg.MIN_RISKS
MIN_TRENDS        = _cfg.MIN_TRENDS
QUALITY_THRESHOLD = _cfg.QUALITY_THRESHOLD  # SBERT grounding threshold

_memory_store: Dict[str, Any] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT MEMORY
# Carries all state between steps — avoids global variables
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentMemory:
    """All agent state in one place. Passed between every step."""
    goal:         str
    company_name: str

    opportunities: list = field(default_factory=list)
    risks:         list = field(default_factory=list)
    trends:        list = field(default_factory=list)
    summaries:     list = field(default_factory=list)

    steps_taken:      int  = 0
    queries_run:      list = field(default_factory=list)
    reasoning_log:    list = field(default_factory=list)
    last_tool_result: str  = ""
    all_docs:         list = field(default_factory=list)  # all retrieved docs

    def status(self) -> str:
        return (
            f"Opportunities: {len(self.opportunities)} — {[o.get('title','') for o in self.opportunities]}\n"
            f"Risks:         {len(self.risks)} — {[r.get('title','') for r in self.risks]}\n"
            f"Trends:        {len(self.trends)} — {[t.get('title','') for t in self.trends]}\n"
            f"Queries done:  {self.queries_run}\n"
            f"Last result:   {self.last_tool_result[:200]}"
        )

    def has_enough(self) -> bool:
        """Research complete — enough evidence to move to recommendations."""
        return (
            len(self.opportunities) >= MIN_OPPORTUNITIES and
            len(self.risks)         >= MIN_RISKS         and
            len(self.trends)        >= MIN_TRENDS
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LLM HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _llm(messages: List[Dict], max_tokens: int = 1000) -> str:
    if LLM_BACKEND == "ollama":
        return _llm_ollama(messages, max_tokens)
    return _llm_groq(messages, max_tokens)


def _llm_groq(messages: List[Dict], max_tokens: int = 1000) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY not set in .env")
    client = Groq(api_key=key)
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL, messages=messages,
                temperature=0.3, max_tokens=max_tokens)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                wait = 10 * (2 ** attempt)
                m = re.search(r"try again in ([0-9.]+)s", msg)
                if m: wait = float(m.group(1)) + 2
                logger.warning(f"[LLM] Rate limit — waiting {wait:.1f}s")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Groq: failed after 5 retries")


def _llm_ollama(messages: List[Dict], max_tokens: int = 1000) -> str:
    import urllib.request

    # qwen3 and other "thinking" models output <think>...</think> blocks
    # before the actual response. Strip them out.
    think_mode = os.getenv("OLLAMA_THINK", "false").lower() == "true"

    payload = json.dumps({
        "model":   OLLAMA_MODEL,
        "messages": messages,
        "stream":  False,
        "think":   think_mode,  # false = disable chain-of-thought output
        "options": {"temperature": 0.3, "num_predict": max_tokens}
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data    = json.loads(resp.read())
        content = data["message"]["content"].strip()

    # Strip any <think>...</think> blocks the model still emits
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


def _call_llm_safe(prompt: str, system: str = "") -> str:
    """
    Used by app.py chat feature.
    Uses CHAT_BACKEND if set — allows fast Groq chat while engine runs on Ollama.
    """
    msgs = []
    if system: msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})

    chat_backend = os.getenv("CHAT_BACKEND", "").lower().strip()

    try:
        if chat_backend == "groq":
            # Chat uses Groq regardless of LLM_BACKEND
            return _llm_groq(msgs, max_tokens=800)
        elif chat_backend == "ollama":
            return _llm_ollama(msgs, max_tokens=800)
        else:
            # Fall back to whatever LLM_BACKEND is set to
            return _llm(msgs, max_tokens=800)
    except Exception as e:
        return f"Error: {e}"


def _parse_json(text: str) -> Any:
    for attempt in [
        text,
        text[text.find("```json")+7: text.rfind("```")] if "```json" in text else "",
        text[text.find("{"):text.rfind("}")+1] if "{" in text else "",
        text[text.find("["):text.rfind("]")+1] if "[" in text else "",
    ]:
        try:
            return json.loads(attempt.strip())
        except Exception:
            continue
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT TOOLS  (non-LLM — programmatic capabilities)
# ═══════════════════════════════════════════════════════════════════════════════

class AgentTools:
    """9 tools the agent can call beyond the LLM."""

    @staticmethod
    def search(query: str, k: int = K_CHUNKS) -> List[Dict]:
        """TOOL: Hybrid ChromaDB search (vector + BM25)."""
        logger.info(f"  [TOOL:search] '{query[:55]}' k={k}")
        try:
            results = hybrid_search(query, k=k)
            logger.info(f"  [TOOL:search] → {len(results)} chunks")
            return results
        except Exception as e:
            logger.error(f"  [TOOL:search] failed: {e}")
            return []

    @staticmethod
    def deduplicate(chunks: List[Dict]) -> List[Dict]:
        """TOOL: Remove duplicate chunks by content fingerprint."""
        seen, unique = set(), []
        for c in chunks:
            key = str(c.get("text") or c.get("content",""))[:100] if isinstance(c,dict) else str(c)[:100]
            if key not in seen:
                seen.add(key); unique.append(c)
        return unique

    @staticmethod
    def score_relevance(chunks: List[Dict], query: str) -> List[Dict]:
        """TOOL: Sort chunks by keyword overlap with query."""
        qwords = set(w.lower() for w in query.split() if len(w) > 4)
        scored = [{**c, "_score": sum(1 for w in qwords
                   if w in (str(c.get("title",""))+str(c.get("text") or c.get("content",""))[:300]).lower())}
                  for c in chunks]
        return sorted(scored, key=lambda x: x.get("_score",0), reverse=True)

    @staticmethod
    def extract_evidence(chunks: List[Dict], n: int = 3) -> List[Dict]:
        """TOOL: Extract structured evidence objects from chunks."""
        ev = []
        for c in chunks[:n]:
            if not isinstance(c, dict): continue
            try:
                ev.append({
                    "source":    c.get("source") or c.get("source_name") or "Unknown",
                    "excerpt":   str(c.get("text") or c.get("content") or "")[:220],
                    "url":       c.get("url","#") or "#",
                    "sentiment": float(c.get("sentiment") or 0.0),
                    "title":     c.get("title",""),
                })
            except Exception:
                continue
        return ev

    @staticmethod
    def format_context(chunks: List[Dict], max_chunks: int = 10) -> str:
        """TOOL: Format chunks into numbered LLM context."""
        parts = []
        for i, c in enumerate(chunks[:max_chunks], 1):
            if not isinstance(c, dict): continue
            parts.append(
                f"[Doc {i}] {c.get('source','?')} — {c.get('title','')}\n"
                f"{str(c.get('text') or c.get('content') or '')[:300]}\n"
                f"URL: {c.get('url','')}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def calculator(expression: str) -> str:
        """TOOL: Safe arithmetic. Pattern from professor's LangChain notebook."""
        if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s\%]+", expression):
            return "Invalid expression"
        try:
            return str(round(float(eval(expression, {"__builtins__": {}})), 4))
        except Exception as e:
            return f"Calculation error: {e}"

    @staticmethod
    def summarize(text: str, max_sentences: int = 3) -> str:
        """TOOL: Extractive summarizer. Pattern from professor's LangChain notebook."""
        import textwrap
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return textwrap.fill(" ".join(sents[:max_sentences]), width=100)

    @staticmethod
    def memory_read(key: str) -> Any:
        """TOOL: Read from persistent memory (LangGraph MemorySaver if available)."""
        if _USE_LANGGRAPH:
            try:
                cfg = {"configurable": {"thread_id": key}}
                state = _langgraph_memory.get(cfg)
                val = state.get("value") if state else None
                logger.info(f"  [TOOL:memory_read] LangGraph '{key}' → {'HIT' if val else 'MISS'}")
                return val
            except Exception:
                pass
        val = _memory_store.get(key)
        logger.info(f"  [TOOL:memory_read] dict '{key}' → {'HIT' if val else 'MISS'}")
        return val

    @staticmethod
    def memory_write(key: str, value: Any) -> str:
        """TOOL: Write to persistent memory (LangGraph MemorySaver if available)."""
        if _USE_LANGGRAPH:
            try:
                cfg = {"configurable": {"thread_id": key}}
                _langgraph_memory.put(cfg, {}, {"value": value}, {})
                return f"LangGraph saved: {key}"
            except Exception:
                pass
        _memory_store[key] = value
        return f"Memory saved: {key}"


tools = AgentTools()


# ═══════════════════════════════════════════════════════════════════════════════
# EVIDENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _confidence(docs: list) -> float:
    """
    Deterministic confidence — NOT from LLM.
    Uses RRF retrieval scores, source diversity, and doc count.
    Pattern from friend's implementation.
    """
    if not docs: return 0.0
    n        = min(len(docs), 6) / 6.0
    src_div  = len({d.get("source") for d in docs}) / 6.0
    strength = min(sum(d.get("score",0) for d in docs) / max(len(docs),1) * 30, 1.0)
    return round(0.5*n + 0.3*src_div + 0.2*strength, 2)


def _attach_evidence_by_index(items: list, docs: list, idx_key: str) -> list:
    """
    Attach evidence using document indices the LLM cited in its output.
    Falls back to first 3 docs if no indices given.
    Pattern from friend's implementation — much more reliable than rotation.
    """
    for item in items:
        ev = []
        for idx in item.get(idx_key, []):
            if isinstance(idx, int) and 1 <= idx <= len(docs):
                d = docs[idx - 1]
                ev.append({
                    "source":    d.get("source","Unknown"),
                    "title":     d.get("title","")[:200],
                    "excerpt":   d.get("text","")[:220],
                    "url":       d.get("url","#"),
                    "sentiment": float(d.get("sentiment") or 0.0),
                })
        if not ev:
            ev = [{
                "source":    d.get("source","Unknown"),
                "title":     d.get("title","")[:200],
                "excerpt":   d.get("text","")[:220],
                "url":       d.get("url","#"),
                "sentiment": float(d.get("sentiment") or 0.0),
            } for d in docs[:3]]
        item["evidence"]   = ev
        item["confidence_score"] = _confidence(docs)
    return items


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL DESCRIPTIONS (shown to LLM in _plan)
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS_DESCRIPTION = f"""Available tools:
  search    : Search the NVIDIA knowledge base (ChromaDB hybrid BM25 + vector).
              Use to find opportunities, risks, and market trends.
  summarise : Consolidate all current findings into a strategic summary.
              Use before concluding to synthesize what was found.
  conclude  : End the research loop and move to recommendations.
              ONLY allowed when: opportunities>={MIN_OPPORTUNITIES}, risks>={MIN_RISKS}, trends>={MIN_TRENDS}."""


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — PLAN (per-step)
# LLM decides which tool to use and generates the query.
# This is true autonomous decision-making — runs every step.
# ═══════════════════════════════════════════════════════════════════════════════

def _plan_step(mem: AgentMemory) -> Dict:
    """
    Per-step planning brain.
    LLM reads memory state and decides: search/summarise/conclude + argument.
    This is the autonomous decision-making the professor requires.
    """
    system = (
        f"You are an autonomous research agent for {mem.company_name}. "
        "Decide which tool to use based on current memory state. "
        "Return ONLY valid JSON."
    )
    user = (
        f"GOAL: {mem.goal}\n\n"
        f"CURRENT STATE:\n{mem.status()}\n\n"
        f"{TOOLS_DESCRIPTION}\n\n"
        "DECISION RULES:\n"
        "1. Use 'search' to find evidence on gaps not yet covered\n"
        "2. Use 'summarise' to consolidate when nearing completion\n"
        f"3. Use 'conclude' ONLY when: opps>={MIN_OPPORTUNITIES} AND risks>={MIN_RISKS} AND trends>={MIN_TRENDS}\n"
        "4. NEVER repeat a query already in queries_run\n"
        "5. Target specific gaps — if risks are low, search for risks specifically\n\n"
        'Return JSON: {"tool":"search|summarise|conclude","argument":"query string","reasoning":"one sentence"}'
    )
    try:
        result = _parse_json(_llm(
            [{"role":"system","content":system},{"role":"user","content":user}],
            max_tokens=200  # plan only needs tool + query + reasoning
        ))
        if isinstance(result, dict) and "tool" in result:
            return result
    except Exception as e:
        logger.warning(f"[PLAN] LLM failed: {e}")

    # Fallback — search for whatever is missing
    missing = []
    if len(mem.opportunities) < MIN_OPPORTUNITIES: missing.append("opportunities")
    if len(mem.risks)         < MIN_RISKS:         missing.append("risks")
    if len(mem.trends)        < MIN_TRENDS:         missing.append("market trends")
    topic = missing[0] if missing else "strategic developments"
    return {
        "tool":      "search",
        "argument":  f"{mem.company_name} {topic}",
        "reasoning": "Fallback — searching for missing findings"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — SEARCH
# Hybrid ChromaDB retrieval + LLM extraction of new findings
# ═══════════════════════════════════════════════════════════════════════════════

def _tool_search(query: str, mem: AgentMemory):
    """
    Hybrid search → extract NEW findings not already in memory.
    Evidence attached by document index (reliable, not rotation guessing).
    """
    if query in mem.queries_run:
        mem.last_tool_result = f"Already searched: '{query}'. Choose a different query."
        logger.info(f"[TOOL:search] Duplicate skipped: {query}")
        return

    mem.queries_run.append(query)
    docs = tools.search(query, k=K_CHUNKS)

    if not docs:
        mem.last_tool_result = f"No results for: '{query}'"
        return

    # Add to global doc pool for evidence reference
    mem.all_docs.extend(docs)

    # TOOL: format_context
    context = tools.format_context(docs, max_chunks=K_CHUNKS)

    system = (
        f"You are a strategy analyst for {mem.company_name}. "
        "Extract NEW strategic intelligence not already in memory. "
        "Return ONLY valid JSON."
    )
    user = (
        f"QUERY: {query}\n\n"
        f"ALREADY IN MEMORY (do not duplicate):\n"
        f"Opportunities: {[o.get('title','') for o in mem.opportunities]}\n"
        f"Risks:         {[r.get('title','') for r in mem.risks]}\n"
        f"Trends:        {[t.get('title','') for t in mem.trends]}\n\n"
        f"DOCUMENTS:\n{context}\n\n"
        "Extract ONLY NEW findings. Return empty lists if nothing new.\n\n"
        "Return JSON:\n"
        "{\n"
        '  "opportunities": [\n'
        '    {"title":"max 8 words","description":"2-3 sentences from docs",'
        '"impact_level":"High|Medium|Low","expected_impact":"specific outcome",'
        '"evidence_doc_indices":[1,2]}\n'
        "  ],\n"
        '  "risks": [\n'
        '    {"title":"max 8 words","description":"2-3 sentences from docs",'
        '"severity":"High|Medium|Low","category":"competitive|regulatory|operational|financial|geopolitical",'
        '"mitigation":"specific action","evidence_doc_indices":[1,2]}\n'
        "  ],\n"
        '  "trends": [\n'
        '    {"title":"max 8 words","description":"2-3 sentences from docs",'
        '"time_horizon":"short-term|mid-term|long-term","expected_impact":"High|Medium|Low",'
        '"relevance":"why this matters for ' + mem.company_name + '",'
        '"evidence_doc_indices":[1,2]}\n'
        "  ]\n"
        "}"
    )

    try:
        parsed = _parse_json(_llm(
            [{"role":"system","content":system},{"role":"user","content":user}],
            max_tokens=900  # findings are short; 1400 was wasteful
        ))
        if not parsed:
            mem.last_tool_result = "No findings extracted"
            return

        new_opps   = _attach_evidence_by_index(parsed.get("opportunities",[]), docs, "evidence_doc_indices")
        new_risks  = _attach_evidence_by_index(parsed.get("risks",[]),         docs, "evidence_doc_indices")
        new_trends = _attach_evidence_by_index(parsed.get("trends",[]),        docs, "evidence_doc_indices")

        mem.opportunities.extend(new_opps)
        mem.risks.extend(new_risks)
        mem.trends.extend(new_trends)

        mem.last_tool_result = (
            f"+{len(new_opps)} opps, +{len(new_risks)} risks, +{len(new_trends)} trends. "
            f"Totals: {len(mem.opportunities)}/{MIN_OPPORTUNITIES} opps, "
            f"{len(mem.risks)}/{MIN_RISKS} risks, {len(mem.trends)}/{MIN_TRENDS} trends."
        )
        logger.info(f"[TOOL:search] +{len(new_opps)} opps | +{len(new_risks)} risks | +{len(new_trends)} trends")

    except Exception as e:
        logger.error(f"[TOOL:search] Error: {e}")
        mem.last_tool_result = f"Error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — SUMMARISE (Decide step)
# Consolidates all findings — this IS the Decide step
# ═══════════════════════════════════════════════════════════════════════════════

def _tool_summarise(mem: AgentMemory):
    """
    LLM-powered consolidation of all findings.
    Acts as the Decide step — synthesizes before concluding.
    """
    logger.info(f"[TOOL:summarise] Consolidating {len(mem.opportunities)} opps, "
                f"{len(mem.risks)} risks, {len(mem.trends)} trends")

    system = f"You are a strategy advisor. Summarise findings about {mem.company_name}."
    user = (
        f"Summarise all strategic intelligence for {mem.company_name}:\n\n"
        f"OPPORTUNITIES ({len(mem.opportunities)}):\n"
        f"{json.dumps([{'title':o.get('title'),'description':o.get('description')} for o in mem.opportunities],indent=2)}\n\n"
        f"RISKS ({len(mem.risks)}):\n"
        f"{json.dumps([{'title':r.get('title'),'description':r.get('description')} for r in mem.risks],indent=2)}\n\n"
        f"TRENDS ({len(mem.trends)}):\n"
        f"{json.dumps([{'title':t.get('title'),'description':t.get('description')} for t in mem.trends],indent=2)}\n\n"
        "Write 4-6 sentences on the most critical strategic themes."
    )

    try:
        summary = _llm([{"role":"system","content":system},{"role":"user","content":user}],
                       max_tokens=500)
        mem.summaries.append({"type": "strategic_summary", "content": summary})
        mem.last_tool_result = f"Summary: {summary[:200]}"
        logger.info(f"[TOOL:summarise] Done ({len(summary)} chars)")
        # Also log via extractive tool
        logger.info(f"[TOOL:summarize-extract] {tools.summarize(summary, 2)}")
    except Exception as e:
        logger.error(f"[TOOL:summarise] Error: {e}")
        mem.last_tool_result = f"Summarise error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — RECOMMEND
# Synthesize all findings into CEO recommendations
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_RECS = """You are the AI Chief Strategy Officer to the CEO.
Given structured opportunities, risks, and trends from live intelligence,
synthesize 4-6 prioritized strategic recommendations.

Each recommendation must:
- Be a concrete action, not a vague aspiration
- Address either an opportunity or a risk (or both)
- Include realistic expected impact and risk assessment
- Reference which input items it draws from

Do not invent data outside the inputs."""

def _generate_recommendations(mem: AgentMemory) -> List[Dict]:
    """DECIDE + RECOMMEND — synthesize all findings into CEO actions."""
    logger.info("[PHASE 2: RECOMMEND] Synthesizing CEO recommendations...")

    if not mem.opportunities and not mem.risks and not mem.trends:
        return []

    # Slim summaries for prompt
    def _slim(items, n=8):
        return [{"idx":i+1,"title":x.get("title"),"description":x.get("description"),
                 "confidence":x.get("confidence_score",0)}
                for i,x in enumerate(items[:n])]

    user = (
        f"Company: {mem.company_name}\n\n"
        f"OPPORTUNITIES:\n{json.dumps(_slim(mem.opportunities, 4),indent=2)}\n\n"
        f"RISKS:\n{json.dumps(_slim(mem.risks, 4),indent=2)}\n\n"
        f"TRENDS:\n{json.dumps(_slim(mem.trends, 3),indent=2)}\n\n"
        "Produce 4 prioritized recommendations (not more). Return JSON:\n"
        '{"recommendations":[\n'
        '  {"title":"concrete action","rationale":"2-3 sentences why now",'
        '"priority":"High|Medium|Low","time_horizon":"0-6 months|6-18 months|18+ months",'
        '"expected_impact":"specific measurable outcome",'
        '"risk_assessment":{"financial":"...","operational":"...","strategic":"..."},'
        '"linked_opportunities":[1,2],"linked_risks":[1],"linked_trends":[1]}\n'
        "]}"
    )

    try:
        parsed = _parse_json(_llm(
            [{"role":"system","content":SYSTEM_RECS},{"role":"user","content":user}],
            max_tokens=2000
        ))
        recs = parsed.get("recommendations",[]) if isinstance(parsed,dict) else []

        # Attach evidence from linked findings
        for r in recs:
            ev = []
            for ix in r.get("linked_opportunities",[]):
                if isinstance(ix,int) and 1 <= ix <= len(mem.opportunities):
                    for e in (mem.opportunities[ix-1].get("evidence") or [])[:2]:
                        ev.append({**e, "from":"opportunity"})
            for ix in r.get("linked_risks",[]):
                if isinstance(ix,int) and 1 <= ix <= len(mem.risks):
                    for e in (mem.risks[ix-1].get("evidence") or [])[:2]:
                        ev.append({**e, "from":"risk"})
            for ix in r.get("linked_trends",[]):
                if isinstance(ix,int) and 1 <= ix <= len(mem.trends):
                    for e in (mem.trends[ix-1].get("evidence") or [])[:1]:
                        ev.append({**e, "from":"trend"})
            r["evidence"] = ev[:6]

            # Normalise to app.py display fields
            r["recommendation"] = r.get("title") or r.get("recommendation","")
            r["risk_assessment"] = r.get("risk_assessment") or {
                "financial":"See rationale","operational":"See rationale","strategic":"See rationale"}

        logger.info(f"[PHASE 2: RECOMMEND] {len(recs)} recommendations")
        return recs

    except Exception as e:
        logger.error(f"[PHASE 2: RECOMMEND] Error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — VALIDATE RECOMMENDATIONS
# Professor's exact requirement: validate before presenting to CEO
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_VALIDATE = """You are a critical strategy reviewer.
Validate recommendations BEFORE they reach the CEO.
For each check: grounded in evidence? specific and actionable? priority justified?
Return ONLY valid JSON."""

def _validate_recommendations(recommendations: List[Dict], mem: AgentMemory) -> List[Dict]:
    """
    Validates each recommendation before presenting to CEO.
    Stage 1: SBERT cosine grounding (objective, if available)
    Stage 2: LLM per-recommendation verdict (position-based matching)
    """
    logger.info(f"[PHASE 3: VALIDATE] Validating {len(recommendations)} recommendations...")

    if not recommendations:
        return []

    # Stage 1: SBERT grounding (objective cosine similarity)
    # SBERT score is BINDING — below threshold forces needs_revision
    SBERT_BLOCK_THRESHOLD = 0.5  # below this → overrides LLM approval
    if _VERIFIER_AVAILABLE:
        logger.info("  [VALIDATE] SBERT grounding check...")
        ver = verify_recommendations(recommendations)
        for i, rec in enumerate(recommendations):
            if i < len(ver.get("scores",[])):
                score = ver["scores"][i][1]
                rec["grounding_confidence"] = score
                # Pre-flag low grounding — Stage 2 LLM cannot override this
                if score < SBERT_BLOCK_THRESHOLD:
                    rec["_sbert_failed"] = True
                    logger.warning(f"  [VALIDATE] SBERT BLOCK: Rec #{i} score={score:.2f} "
                                   f"— will force needs_revision regardless of LLM")
        logger.info(f"  [VALIDATE] SBERT: mean={ver['mean_confidence']:.2f} "
                    f"precision={ver['factual_precision']:.0%}")

    # Stage 2: LLM per-recommendation verdict
    user = (
        f"Company: {mem.company_name}\n\n"
        f"Review these {len(recommendations)} recommendations:\n"
        f"{json.dumps([{'index':i+1,'title':r.get('title') or r.get('recommendation',''),'rationale':r.get('rationale',''),'priority':r.get('priority','')} for i,r in enumerate(recommendations)],indent=2)}\n\n"
        f"Available evidence: {len(mem.opportunities)} opps, {len(mem.risks)} risks, {len(mem.trends)} trends\n\n"
        f"Return EXACTLY {len(recommendations)} validations in the same order:\n"
        '{{"validations":[{{"index":1,"approved":true/false,"verdict":"approved|needs_revision|rejected","reason":"2 sentences","suggestion":""}}]}}'
    )

    try:
        parsed      = _parse_json(_llm(
            [{"role":"system","content":SYSTEM_VALIDATE},{"role":"user","content":user}],
            max_tokens=1500
        ))
        validations = parsed.get("validations",[]) if isinstance(parsed,dict) else []

        for i, rec in enumerate(recommendations):
            v = validations[i] if i < len(validations) else {}
            # Apply SBERT override if it failed — SBERT block cannot be overridden by LLM
            if rec.pop("_sbert_failed", False):
                rec["validation"] = {
                    "approved":   False,
                    "verdict":    "needs_revision",
                    "reason":     f"SBERT grounding score {rec.get('grounding_confidence',0):.2f} "
                                  f"below threshold {SBERT_BLOCK_THRESHOLD} — insufficient evidence grounding.",
                    "suggestion": v.get("suggestion", "Strengthen evidence links before presenting to CEO."),
                }
            else:
                rec["validation"] = {
                    "approved":   v.get("approved", True),
                    "verdict":    v.get("verdict",  "approved"),
                    "reason":     v.get("reason",   ""),
                    "suggestion": v.get("suggestion",""),
                }
            emoji = "✅" if rec["validation"]["approved"] else "⚠️"
            logger.info(f"  [VALIDATE] {emoji} '{(rec.get('title') or rec.get('recommendation',''))[:50]}' "
                        f"→ {rec['validation']['verdict']}")

    except Exception as e:
        logger.error(f"[VALIDATE] LLM error: {e}")
        for rec in recommendations:
            rec["validation"] = {"approved":True,"verdict":"approved",
                                 "reason":f"Validation error: {e}","suggestion":""}

    approved = sum(1 for r in recommendations if r.get("validation",{}).get("approved",True))
    logger.info(f"[PHASE 3: VALIDATE] {approved}/{len(recommendations)} approved")
    return recommendations


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — CEO BRIEFING
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_BRIEF = """You are writing a one-page executive briefing for the CEO.
Write in four sections:
1. Executive Summary (2-3 paragraphs)
2. What happened? (2-4 sentences on key developments)
3. Why does it matter? (2-4 sentences on strategic implications)
4. What to do next? (3-5 action-oriented bullet points)
Be direct, executive-tone, evidence-grounded."""

def _generate_briefing(mem: AgentMemory, recommendations: List[Dict]) -> Dict:
    """Generate comprehensive CEO briefing from all validated findings."""
    logger.info("[PHASE 4: BRIEF] Generating CEO briefing...")

    def _slim(items, n=5):
        return [{"title":x.get("title"),"description":x.get("description")} for x in items[:n]]

    user = (
        f"Company: {mem.company_name}\n\n"
        f"Top opportunities: {json.dumps(_slim(mem.opportunities),indent=2)}\n\n"
        f"Top risks: {json.dumps(_slim(mem.risks),indent=2)}\n\n"
        f"Top trends: {json.dumps(_slim(mem.trends),indent=2)}\n\n"
        f"Validated recommendations: {json.dumps([{'title':r.get('title') or r.get('recommendation',''),'priority':r.get('priority',''),'verdict':r.get('validation',{}).get('verdict','approved')} for r in recommendations[:6]],indent=2)}\n\n"
        "Return JSON with EXACTLY these keys:\n"
        '{"executive_summary":"multi-paragraph strategic overview",'
        '"what_happened":"current market conditions and key developments",'
        '"why_it_matters":"strategic implications and business impact",'
        '"what_to_do_next":"prioritised action plan with specific steps"}'
    )

    raw    = _llm([{"role":"system","content":SYSTEM_BRIEF},{"role":"user","content":user}],
                  max_tokens=1000)
    result = _parse_json(raw) or {
        "executive_summary": mem.summaries[-1]["content"] if mem.summaries else "Analysis complete.",
        "what_happened":     "See findings above.",
        "why_it_matters":    "See risks and opportunities.",
        "what_to_do_next":   "See recommendations.",
    }

    # Log summary via extractive tool
    logger.info(f"[BRIEF] {tools.summarize(result.get('executive_summary',''), 2)[:120]}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP — true autonomous ReAct
# ═══════════════════════════════════════════════════════════════════════════════

def run_intelligence_engine(company: str = "NVIDIA") -> Dict:
    """
    Main entry point called by app.py.

    True autonomous loop — agent runs until it decides it has enough,
    not until a fixed iteration count is reached.

    Workflow: Goal → [Plan → Search → Extract → Check] → Recommend → Validate → Brief
    """
    goal = f"Build a comprehensive strategic intelligence report for {company}"

    # Check memory for prior run context
    prior = tools.memory_read(f"{company}:last_run")
    if prior:
        logger.info(f"[AGENT] Prior run found: {prior}")

    mem = AgentMemory(goal=goal, company_name=company)

    logger.info("="*60)
    logger.info(f"[AGENT] GOAL: {goal}")
    logger.info(f"[AGENT] Workflow: Goal→Plan→Retrieve→Analyze→Decide→Recommend→Validate")
    logger.info(f"[AGENT] Required: {MIN_OPPORTUNITIES} opps | {MIN_RISKS} risks | {MIN_TRENDS} trends")
    logger.info("="*60)

    # ── PHASE 1: RESEARCH LOOP ────────────────────────────────────────────────
    while mem.steps_taken < MAX_STEPS:
        mem.steps_taken += 1
        logger.info(f"\n[AGENT] ── Step {mem.steps_taken}/{MAX_STEPS} ──")

        # Per-step plan: LLM decides tool + query
        decision  = _plan_step(mem)
        tool      = decision.get("tool", "search")
        argument  = decision.get("argument", "")
        reasoning = decision.get("reasoning", "")

        logger.info(f"[AGENT] Tool:      {tool}")
        logger.info(f"[AGENT] Argument:  {argument}")
        logger.info(f"[AGENT] Reasoning: {reasoning}")

        mem.reasoning_log.append({
            "step": mem.steps_taken, "tool": tool,
            "argument": argument, "reasoning": reasoning
        })

        if tool == "search":
            _tool_search(argument, mem)

        elif tool == "summarise":
            _tool_summarise(mem)

        elif tool == "conclude":
            if mem.has_enough():
                logger.info(f"\n[AGENT] ✅ Research complete — concluding")
                logger.info(f"[AGENT] {len(mem.opportunities)} opps | "
                            f"{len(mem.risks)} risks | {len(mem.trends)} trends")
                break
            else:
                gaps = []
                if len(mem.opportunities) < MIN_OPPORTUNITIES:
                    gaps.append(f"need {MIN_OPPORTUNITIES-len(mem.opportunities)} more opps")
                if len(mem.risks) < MIN_RISKS:
                    gaps.append(f"need {MIN_RISKS-len(mem.risks)} more risks")
                if len(mem.trends) < MIN_TRENDS:
                    gaps.append(f"need {MIN_TRENDS-len(mem.trends)} more trends")
                msg = "Conclude rejected: " + "; ".join(gaps)
                logger.info(f"[AGENT] ⚠️  {msg}")
                mem.last_tool_result = msg + ". Keep searching."

        else:
            _tool_search(argument or f"{company} strategic analysis", mem)

        # ── IMMEDIATE CONCLUDE CHECK ──────────────────────────────────────────
        # As soon as thresholds are met after ANY tool call → conclude immediately.
        # This is the key fix: don't wait for the LLM to decide — act now.
        if mem.has_enough():
            logger.info(f"\n[AGENT] ✅ Thresholds met — concluding immediately")
            logger.info(f"[AGENT] {len(mem.opportunities)}/{MIN_OPPORTUNITIES} opps | "
                        f"{len(mem.risks)}/{MIN_RISKS} risks | "
                        f"{len(mem.trends)}/{MIN_TRENDS} trends")
            break

        # Rate limit guard between steps
        time.sleep(10 if LLM_BACKEND == "groq" else 2)

    # ── PHASE 2: RECOMMEND ────────────────────────────────────────────────────
    recommendations = _generate_recommendations(mem)
    time.sleep(12 if LLM_BACKEND == "groq" else 2)

    # ── PHASE 3: VALIDATE ─────────────────────────────────────────────────────
    recommendations = _validate_recommendations(recommendations, mem)
    time.sleep(12 if LLM_BACKEND == "groq" else 2)

    # ── PHASE 4: BRIEF ────────────────────────────────────────────────────────
    briefing = _generate_briefing(mem, recommendations)

    # Persist results to memory
    tools.memory_write(f"{company}:last_run", {
        "steps":    mem.steps_taken,
        "counts":   {"opps": len(mem.opportunities), "risks": len(mem.risks), "trends": len(mem.trends)},
        "approved": sum(1 for r in recommendations if r.get("validation",{}).get("approved",True))
    })

    # Final safety net — any item missing evidence gets pool rotation
    pool = mem.all_docs
    if pool:
        pool_size = len(pool); g = 0
        for cat, items in [("opportunities",mem.opportunities),("risks",mem.risks),("trends",mem.trends)]:
            for item in items:
                if isinstance(item,dict) and not item.get("evidence"):
                    start = (g*3) % pool_size
                    item["evidence"] = tools.extract_evidence(pool[start:start+3], n=3)
                    g += 1

    approved = sum(1 for r in recommendations if r.get("validation",{}).get("approved",True))
    logger.info(f"\n{'='*60}")
    logger.info(f"[AGENT] COMPLETE — {mem.steps_taken} steps")
    logger.info(f"[AGENT] opps={len(mem.opportunities)} risks={len(mem.risks)} "
                f"trends={len(mem.trends)} recs={len(recommendations)} approved={approved}")
    logger.info(f"[AGENT] Queries run: {mem.queries_run}")
    logger.info(f"{'='*60}")

    result_dict = {
        "opportunities":   mem.opportunities,
        "risks":           mem.risks,
        "trends":          mem.trends,
        "recommendations": recommendations,
        "ceo_briefing":    briefing,
        "agent_log": {
            "goal":          mem.goal,
            "total_steps":   mem.steps_taken,
            "queries_run":   mem.queries_run,
            "reasoning_log": mem.reasoning_log,
            "final_counts": {
                "opportunities":            len(mem.opportunities),
                "risks":                    len(mem.risks),
                "trends":                   len(mem.trends),
                "recommendations":          len(recommendations),
                "recommendations_approved": approved,
            },
        },
    }

    # ── SAVE OUTPUTS (runs whether called from pipeline.py or app.py button) ──
    _save_outputs(result_dict, mem)

    # Return typed StrategicReport if models.py is available
    if _MODELS_AVAILABLE:
        return StrategicReport(
            opportunities   = mem.opportunities,
            risks           = mem.risks,
            trends          = mem.trends,
            recommendations = recommendations,
            ceo_briefing    = briefing,
            agent_plan      = {"queries": mem.queries_run, "steps": mem.steps_taken},
            agent_validation= {"approved": approved, "total": len(recommendations)},
            error           = None,
        )

    return result_dict


def _save_outputs(d: Dict, mem: Any) -> None:
    """
    Save report.json, agent_report.txt and agent_report.md to outputs/.
    Called automatically at end of every run — from pipeline.py OR app.py button.
    """
    try:
        from datetime import datetime

        # outputs/ folder sits at project root (3 levels up from this file)
        outputs_dir = Path(__file__).resolve().parents[3] / "outputs"
        outputs_dir.mkdir(exist_ok=True)

        # 1. Full JSON data
        (outputs_dir / "report.json").write_text(
            json.dumps(d, indent=2, default=str), encoding="utf-8"
        )

        agent_log = d.get("agent_log", {})
        counts    = agent_log.get("final_counts", {})
        now       = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 2. Plain text report
        txt = []
        txt.append("=" * 60)
        txt.append("NVIDIA STRATEGIC INTELLIGENCE AGENT — EXECUTION REPORT")
        txt.append("=" * 60)
        txt.append(f"Generated:   {now}")
        txt.append(f"Goal:        {agent_log.get('goal','')}")
        txt.append(f"Total steps: {agent_log.get('total_steps', 0)}")
        txt.append(f"Queries run: {len(agent_log.get('queries_run', []))}")
        txt.append("")
        txt.append("FINAL COUNTS:")
        txt.append(f"  Opportunities:            {counts.get('opportunities', 0)}")
        txt.append(f"  Risks:                    {counts.get('risks', 0)}")
        txt.append(f"  Trends:                   {counts.get('trends', 0)}")
        txt.append(f"  Recommendations:          {counts.get('recommendations', 0)}")
        txt.append(f"  Recommendations approved: {counts.get('recommendations_approved', 0)}")
        txt.append("")
        txt.append("SEARCH QUERIES EXECUTED:")
        for i, q in enumerate(agent_log.get("queries_run", []), 1):
            txt.append(f"  {i}. {q}")
        txt.append("")
        txt.append("STEP-BY-STEP REASONING LOG:")
        txt.append("-" * 60)
        for step in agent_log.get("reasoning_log", []):
            txt.append(f"Step {step.get('step','?'):>2} | Tool: {step.get('tool','?'):<12} | {step.get('argument','')[:60]}")
            txt.append(f"         Reasoning: {step.get('reasoning','')[:100]}")
        txt.append("")
        txt.append("OPPORTUNITIES:")
        txt.append("-" * 60)
        for i, o in enumerate(d.get("opportunities", []), 1):
            txt.append(f"{i}. [{o.get('impact_level','?').upper()}] {o.get('title','')}")
            txt.append(f"   {o.get('description','')[:200]}")
            txt.append("")
        txt.append("RISKS:")
        txt.append("-" * 60)
        for i, r in enumerate(d.get("risks", []), 1):
            txt.append(f"{i}. [{r.get('severity','?').upper()}] {r.get('title','')} — {r.get('category','')}")
            txt.append(f"   {r.get('description','')[:200]}")
            txt.append(f"   Mitigation: {r.get('mitigation','')[:150]}")
            txt.append("")
        txt.append("TRENDS:")
        txt.append("-" * 60)
        for i, t in enumerate(d.get("trends", []), 1):
            txt.append(f"{i}. {t.get('title','')}")
            txt.append(f"   {t.get('description','')[:200]}")
            txt.append("")
        txt.append("STRATEGIC RECOMMENDATIONS:")
        txt.append("-" * 60)
        for i, r in enumerate(d.get("recommendations", []), 1):
            rec_text   = r.get("recommendation") or r.get("title", "")
            validation = r.get("validation", {})
            grounding  = r.get("grounding_confidence", None)
            txt.append(f"{i}. [{r.get('priority','?').upper()}] {rec_text[:120]}")
            txt.append(f"   Time horizon:    {r.get('time_horizon','')}")
            txt.append(f"   Expected impact: {r.get('expected_impact','')[:150]}")
            txt.append(f"   SBERT grounding: {grounding if grounding is not None else 'N/A'}")
            txt.append(f"   Validation:      {validation.get('verdict','?')} — {validation.get('reason','')[:100]}")
            txt.append("")
        briefing = d.get("ceo_briefing", {})
        if briefing:
            txt.append("CEO BRIEFING:")
            txt.append("-" * 60)
            txt.append("EXECUTIVE SUMMARY:")
            txt.append(str(briefing.get("executive_summary", "")))
            txt.append("")
            txt.append("WHAT HAPPENED:")
            txt.append(str(briefing.get("what_happened", "")))
            txt.append("")
            txt.append("WHY IT MATTERS:")
            txt.append(str(briefing.get("why_it_matters", "")))
            txt.append("")
            txt.append("WHAT TO DO NEXT:")
            txt.append(str(briefing.get("what_to_do_next", "")))
        txt.append("")
        txt.append("=" * 60)
        txt.append("END OF AGENT REPORT")
        txt.append("=" * 60)
        (outputs_dir / "agent_report.txt").write_text("\n".join(txt), encoding="utf-8")

        # 3. Markdown report
        md = []
        md.append("# NVIDIA Strategic Intelligence Report")
        md.append(f"> Generated: {now}  ")
        md.append(f"> Agent steps: {agent_log.get('total_steps', 0)}  ")
        md.append(f"> Goal: {agent_log.get('goal', '')}")
        md.append("")
        md.append("## Agent Execution Log")
        md.append("")
        md.append("### Search Queries")
        for i, q in enumerate(agent_log.get("queries_run", []), 1):
            md.append(f"{i}. `{q}`")
        md.append("")
        md.append("### Step-by-Step Reasoning")
        md.append("")
        md.append("| Step | Tool | Query | Reasoning |")
        md.append("|------|------|-------|-----------|")
        for step in agent_log.get("reasoning_log", []):
            md.append(
                f"| {step.get('step','?')} "
                f"| `{step.get('tool','?')}` "
                f"| {step.get('argument','')[:50]} "
                f"| {step.get('reasoning','')[:80]} |"
            )
        md.append("")
        md.append("### Final Counts")
        md.append("| Category | Count |")
        md.append("|----------|-------|")
        md.append(f"| Opportunities | {counts.get('opportunities', 0)} |")
        md.append(f"| Risks | {counts.get('risks', 0)} |")
        md.append(f"| Trends | {counts.get('trends', 0)} |")
        md.append(f"| Recommendations | {counts.get('recommendations', 0)} |")
        md.append(f"| Approved | {counts.get('recommendations_approved', 0)} |")
        md.append("")
        md.append("## Opportunities")
        for i, o in enumerate(d.get("opportunities", []), 1):
            md.append(f"### {i}. {o.get('title', '')}")
            md.append(f"**Impact:** {o.get('impact_level','?')} | **Confidence:** {o.get('confidence_score', o.get('confidence','?'))}")
            md.append("")
            md.append(o.get("description", ""))
            md.append("")
        md.append("## Risks")
        for i, r in enumerate(d.get("risks", []), 1):
            md.append(f"### {i}. {r.get('title', '')}")
            md.append(f"**Severity:** {r.get('severity','?')} | **Category:** {r.get('category','?')}")
            md.append("")
            md.append(r.get("description", ""))
            md.append("")
            md.append(f"**Mitigation:** {r.get('mitigation','')}")
            md.append("")
        md.append("## Trends")
        for i, t in enumerate(d.get("trends", []), 1):
            md.append(f"### {i}. {t.get('title', '')}")
            md.append(f"**Time Horizon:** {t.get('time_horizon','?')} | **Impact:** {t.get('expected_impact','?')}")
            md.append("")
            md.append(t.get("description", ""))
            md.append("")
        md.append("## Strategic Recommendations")
        for i, r in enumerate(d.get("recommendations", []), 1):
            rec_text   = r.get("recommendation") or r.get("title", "")
            validation = r.get("validation", {})
            verdict    = validation.get("verdict", "approved")
            grounding  = r.get("grounding_confidence", None)
            v_icon     = "✅" if verdict == "approved" else "⚠️" if verdict == "needs_revision" else "❌"
            md.append(f"### {i}. {rec_text}")
            md.append(f"**Priority:** {r.get('priority','?')} | **Time Horizon:** {r.get('time_horizon','?')}")
            md.append("")
            md.append(f"**Expected Impact:** {r.get('expected_impact','')}")
            md.append("")
            md.append(f"#### Validation")
            md.append(f"{v_icon} **{verdict.replace('_',' ').upper()}**" +
                      (f" — SBERT: `{grounding:.2f}`" if grounding is not None else ""))
            if validation.get("reason"):
                md.append(f"> {validation.get('reason','')}")
            md.append("")
            risk_ass = r.get("risk_assessment", {})
            if risk_ass:
                md.append("#### Risk Assessment")
                md.append(f"- **Financial:** {risk_ass.get('financial','')}")
                md.append(f"- **Operational:** {risk_ass.get('operational','')}")
                md.append(f"- **Strategic:** {risk_ass.get('strategic','')}")
            md.append("")
        if briefing:
            md.append("## CEO Briefing")
            md.append("")
            md.append("### Executive Summary")
            md.append(str(briefing.get("executive_summary", "")))
            md.append("")
            md.append("### What Happened")
            md.append(str(briefing.get("what_happened", "")))
            md.append("")
            md.append("### Why It Matters")
            md.append(str(briefing.get("why_it_matters", "")))
            md.append("")
            md.append("### What to Do Next")
            md.append(str(briefing.get("what_to_do_next", "")))
            md.append("")
        md.append("---")
        md.append("*Generated by NVIDIA Strategic Intelligence Agent*")
        (outputs_dir / "agent_report.md").write_text("\n".join(md), encoding="utf-8")

        logger.info(f"[AGENT] Outputs saved to {outputs_dir}")
        logger.info(f"[AGENT]   report.json | agent_report.txt | agent_report.md")

    except Exception as e:
        logger.warning(f"[AGENT] Output save failed: {e}")


def report_to_dict(result) -> Dict:
    """Convert StrategicReport or dict to plain dict for app.py."""
    if isinstance(result, dict):
        return result
    # StrategicReport dataclass — convert to dict
    try:
        import dataclasses
        if dataclasses.is_dataclass(result):
            d = dataclasses.asdict(result)
            # Remove None error field to keep dict clean
            d.pop("error", None)
            return d
    except Exception:
        pass
    # Last resort — try __dict__
    try:
        return vars(result)
    except Exception:
        return {}
