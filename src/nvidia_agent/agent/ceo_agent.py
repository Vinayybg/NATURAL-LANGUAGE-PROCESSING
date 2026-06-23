"""
agent/ceo_agent.py — AI CEO Strategic Intelligence Engine (Tasks 4, 5, 6).

Task 6 compliance — every recommendation contains:
  ✅ Recommendation      → specific CEO action starting with strong verb
  ✅ Supporting Evidence → 3 source documents with excerpt, URL, sentiment
  ✅ Expected Impact     → quantified business outcome
  ✅ Risk Assessment     → Financial + Operational + Strategic risk breakdown
"""

import json
import logging
import re
import time
from dataclasses import asdict

from nvidia_agent import config
from nvidia_agent.storage.vector_store import hybrid_search
from nvidia_agent.agent.llm import call_llm
from nvidia_agent.agent.models import (
    Evidence, Opportunity, Risk, Trend,
    Recommendation, RiskAssessment, StrategicReport
)

log = logging.getLogger(__name__)

# ── System prompts ────────────────────────────────────────────────────────────

CEO_SYSTEM_PROMPT = f"""You are an elite AI Strategic Advisor to the CEO of {config.COMPANY_NAME}.
You combine the analytical rigour of a McKinsey senior partner with deep expertise
in semiconductors, AI infrastructure, and technology strategy.

Your role:
- Analyse live intelligence from news, company announcements, and research
- Identify strategic OPPORTUNITIES and RISKS with specific business implications
- Generate RECOMMENDATIONS that are specific, actionable, and evidence-backed
- Justify every recommendation with Financial, Operational, and Strategic risk assessment

CRITICAL OUTPUT RULE:
Respond with ONLY valid JSON. No text before or after. No markdown fences. Raw JSON only.
"""

CEO_CHAT_SYSTEM = f"""You are an elite AI Strategic Advisor to the CEO of {config.COMPANY_NAME}.
Answer questions based ONLY on the provided intelligence sources.
Be concise, direct, and executive-level. Reference specific sources in your answer.
If the sources do not contain enough information, say so clearly.
Do NOT make up facts. Only use information from the provided sources.
Respond in plain text only — no markdown headers, no ### symbols.
"""


# ── Rate-limit-aware LLM caller ───────────────────────────────────────────────

def _call_llm_safe(prompt: str, system: str, max_retries: int = 6) -> str:
    """
    Wraps call_llm with exponential backoff on Groq 429 rate-limit errors.
    Wait schedule: 30s → 60s → 120s → 240s → 480s → 960s before giving up.
    """
    for attempt in range(max_retries):
        try:
            return call_llm(prompt, system)
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = (
                "429" in str(e)
                or "rate limit" in err
                or "too many requests" in err
                or "rate_limit_exceeded" in err
            )
            if is_rate_limit and attempt < max_retries - 1:
                wait = 30 * (2 ** attempt)   # 30, 60, 120, 240, 480, 960 seconds
                retry_match = re.search(r"retry.after[^\d]*(\d+)", err)
                if retry_match:
                    wait = max(int(retry_match.group(1)) + 2, wait)
                log.warning(
                    f"  Groq rate limit — waiting {wait}s "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Groq rate limit exceeded after {max_retries} retries")


# ── Inter-task delay ──────────────────────────────────────────────────────────

_TASK_DELAY = 15   # seconds between analysis tasks — keeps RPM under free tier

def _task_pause(label: str = "") -> None:
    log.info(f"  Pausing {_TASK_DELAY}s — {label}...")
    time.sleep(_TASK_DELAY)


# ── Private helpers ───────────────────────────────────────────────────────────

def _sentiment_label(score: float) -> str:
    if score >= 0.05:  return "positive"
    if score <= -0.05: return "negative"
    return "neutral"


def _retrieve(query: str, k: int = config.TOP_K_RETRIEVAL) -> tuple[str, list[dict]]:
    """Run hybrid BM25+cosine search, return (formatted_context, raw_docs)."""
    docs = hybrid_search(query, k=k)
    if not docs:
        return "", []
    parts = [
        f"[SOURCE {i}] {d['source']} — {d['title']}\n"
        f"Relevance: {d['score']:.2f} | Sentiment: {d['sentiment']:.2f} "
        f"({_sentiment_label(d['sentiment'])})\n{d['text'][:700]}\nURL: {d['url']}"
        for i, d in enumerate(docs, 1)
    ]
    return "\n\n---\n\n".join(parts), docs


def _evidence_from_docs(docs: list[dict], max_items: int = 3) -> list[Evidence]:
    """Convert retrieved chunks to Evidence dataclass objects."""
    return [
        Evidence(
            source=d["source"],
            excerpt=d["text"][:200].replace("\n", " "),
            url=d["url"],
            sentiment=d["sentiment"],
        )
        for d in docs[:max_items]
    ]


def _parse_json(raw: str) -> dict | list:
    """
    Robustly extract JSON from LLM output.
    Tier 1: strip markdown fences and parse directly.
    Tier 2: regex-extract first {...} or [...] block.
    Tier 3: return empty dict — never crash.
    """
    clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
        m = re.search(pattern, clean)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    log.warning(f"JSON parse failed. First 300 chars:\n{raw[:300]}")
    return {}


# ── Analysis Tasks ────────────────────────────────────────────────────────────

def analyse_opportunities() -> list[Opportunity]:
    """Task 4a — Identify top 3 strategic opportunities from live intelligence."""
    log.info("  Analysing opportunities...")
    context, docs = _retrieve(
        "NVIDIA growth opportunities new markets AI demand partnerships "
        "sovereign AI robotics autonomous vehicles enterprise agentic"
    )
    if not context:
        return []

    prompt = f"""Based on the following live intelligence about {config.COMPANY_NAME},
identify the TOP 3 strategic OPPORTUNITIES.

INTELLIGENCE:
{context}

Return ONLY a JSON array of exactly 3 objects:
[
  {{
    "title": "Short specific opportunity title (max 8 words)",
    "description": "2-3 sentences explaining the opportunity with specific evidence",
    "impact_level": "High",
    "confidence_score": 0.85,
    "expected_impact": "Specific quantified business outcome where possible"
  }}
]

Valid impact_level: High, Medium, Low. confidence_score: 0.0 to 1.0."""

    data = _parse_json(_call_llm_safe(prompt, CEO_SYSTEM_PROMPT))
    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) else []
    evidence = _evidence_from_docs(docs)
    return [
        Opportunity(
            title=i.get("title", ""),
            description=i.get("description", ""),
            impact_level=i.get("impact_level", "Medium"),
            confidence_score=float(i.get("confidence_score", 0.7)),
            evidence=evidence,
            expected_impact=i.get("expected_impact", ""),
        )
        for i in data[:3] if isinstance(i, dict) and "title" in i
    ]


def analyse_risks() -> list[Risk]:
    """Task 4b — Identify top 3 strategic risks from live intelligence."""
    log.info("  Analysing risks...")
    context, docs = _retrieve(
        "NVIDIA risks threats competition AMD Intel custom chips export controls "
        "regulatory China supply chain TSMC Huawei customer concentration"
    )
    if not context:
        return []

    prompt = f"""Based on the following live intelligence about {config.COMPANY_NAME},
identify the TOP 3 strategic RISKS.

INTELLIGENCE:
{context}

Return ONLY a JSON array of exactly 3 objects:
[
  {{
    "title": "Short specific risk title (max 8 words)",
    "description": "2-3 sentences explaining the risk with specific evidence",
    "category": "competitive",
    "severity": "High",
    "confidence_score": 0.8,
    "mitigation": "Specific CEO action to mitigate this risk"
  }}
]

Valid category: competitive | regulatory | supply_chain | sentiment | macro | technology
Valid severity: High, Medium, Low"""

    data = _parse_json(_call_llm_safe(prompt, CEO_SYSTEM_PROMPT))
    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) else []
    evidence = _evidence_from_docs(docs)
    return [
        Risk(
            title=i.get("title", ""),
            description=i.get("description", ""),
            category=i.get("category", "competitive"),
            severity=i.get("severity", "Medium"),
            confidence_score=float(i.get("confidence_score", 0.7)),
            evidence=evidence,
            mitigation=i.get("mitigation", ""),
        )
        for i in data[:3] if isinstance(i, dict) and "title" in i
    ]


def analyse_trends() -> list[Trend]:
    """Task 4c — Detect top 4 technology and industry trends."""
    log.info("  Detecting trends...")
    context, docs = _retrieve(
        "AI technology trends inference training shift custom silicon ASIC "
        "edge computing robotics enterprise AI adoption sovereign AI"
    )
    if not context:
        return []

    prompt = f"""Based on the intelligence, identify TOP 4 technology and industry TRENDS
most relevant to {config.COMPANY_NAME}.

INTELLIGENCE:
{context}

Return ONLY a JSON array of exactly 4 objects:
[
  {{
    "title": "Trend title (max 6 words)",
    "description": "2 sentences describing the trend with specific evidence",
    "relevance": "1-2 sentences on why this matters specifically for NVIDIA"
  }}
]"""

    data = _parse_json(_call_llm_safe(prompt, CEO_SYSTEM_PROMPT))
    if not isinstance(data, list):
        data = []
    return [
        Trend(
            title=i.get("title", ""),
            description=i.get("description", ""),
            relevance=i.get("relevance", ""),
            source_count=len(docs),
        )
        for i in data[:4] if isinstance(i, dict)
    ]


def generate_recommendations(
    opportunities: list[Opportunity],
    risks: list[Risk],
) -> list[Recommendation]:
    """
    Task 5 & 6 — Generate 4 evidence-backed CEO action items.

    Task 6 requires every recommendation to contain:
      ✅ Recommendation      → specific action starting with strong verb
      ✅ Supporting Evidence → from _evidence_from_docs()
      ✅ Expected Impact     → quantified business outcome
      ✅ Risk Assessment     → Financial + Operational + Strategic breakdown
    """
    log.info("  Generating recommendations...")
    context, docs = _retrieve(
        "NVIDIA strategic priorities investment decisions competitive moat "
        "CUDA ecosystem software AI infrastructure market leadership"
    )

    opp_summary = "\n".join(
        f"  [{o.impact_level}] {o.title}: {o.description}"
        for o in opportunities
    ) or "  None identified"

    risk_summary = "\n".join(
        f"  [{r.severity}] {r.title}: {r.description}"
        for r in risks
    ) or "  None identified"

    prompt = f"""You are advising the CEO of {config.COMPANY_NAME}.

OPPORTUNITIES IDENTIFIED:
{opp_summary}

RISKS IDENTIFIED:
{risk_summary}

SUPPORTING INTELLIGENCE:
{context[:3000]}

Generate exactly 4 specific, actionable STRATEGIC RECOMMENDATIONS.
Each must start with a strong verb and be specific enough to act on this quarter.

CRITICAL: For each recommendation, provide a THREE-PART risk assessment:
- financial_risk: what this costs or what revenue is at risk if it fails
- operational_risk: execution difficulty, team requirements, timeline challenges
- strategic_risk: competitive positioning risk if this recommendation is wrong

Return ONLY a JSON array of exactly 4 objects:
[
  {{
    "recommendation": "Specific CEO action starting with a strong verb",
    "priority": "High",
    "expected_impact": "Specific quantified business outcome",
    "time_horizon": "Short-term (0-6 months)",
    "financial_risk": "High — requires $X investment with Y% ROI uncertainty",
    "operational_risk": "Medium — needs new team of Z people and 6-month ramp",
    "strategic_risk": "Low — aligns with existing CUDA moat and partner ecosystem"
  }}
]

Valid priority: High, Medium, Low
Valid time_horizon: Short-term (0-6 months) | Mid-term (6-18 months) | Long-term (18+ months)"""

    data = _parse_json(_call_llm_safe(prompt, CEO_SYSTEM_PROMPT))
    if not isinstance(data, list):
        data = []
    evidence = _evidence_from_docs(docs)

    return [
        Recommendation(
            recommendation=i.get("recommendation", ""),
            priority=i.get("priority", "Medium"),
            evidence=evidence,
            expected_impact=i.get("expected_impact", ""),
            risk_assessment=RiskAssessment(
                financial=i.get("financial_risk", "To be assessed"),
                operational=i.get("operational_risk", "To be assessed"),
                strategic=i.get("strategic_risk", "To be assessed"),
            ),
            time_horizon=i.get("time_horizon", "Mid-term (6-18 months)"),
        )
        for i in data[:4] if isinstance(i, dict) and "recommendation" in i
    ]


def generate_ceo_briefing(
    opportunities: list[Opportunity],
    risks: list[Risk],
    recommendations: list[Recommendation],
) -> dict:
    """Generate the executive CEO briefing: what happened, why, what next."""
    log.info("  Writing CEO briefing...")
    opp_text  = "\n".join(f"• {o.title}: {o.description}" for o in opportunities) or "None"
    risk_text = "\n".join(f"• {r.title}: {r.description}" for r in risks) or "None"
    rec_text  = "\n".join(f"• {r.recommendation}" for r in recommendations) or "None"

    prompt = f"""Write an executive CEO briefing for {config.COMPANY_NAME}.

OPPORTUNITIES: {opp_text}
RISKS: {risk_text}
RECOMMENDED ACTIONS: {rec_text}

Return ONLY a JSON object with exactly these 4 keys:
{{
  "what_happened": "2-3 sentences on the most important recent developments",
  "why_it_matters": "2-3 sentences on strategic significance for {config.COMPANY_NAME}",
  "what_to_do_next": "3-4 specific bullet-point actions the CEO should prioritise this week",
  "executive_summary": "One powerful paragraph (4-5 sentences) suitable for a board presentation"
}}"""

    data = _parse_json(_call_llm_safe(prompt, CEO_SYSTEM_PROMPT))
    if not isinstance(data, dict):
        data = {
            "what_happened":     "Strategic intelligence analysis complete.",
            "why_it_matters":    "Multiple significant developments identified from live sources.",
            "what_to_do_next":   "Review the recommendations section for prioritised action items.",
            "executive_summary": f"Strategic intelligence report for {config.COMPANY_NAME} generated.",
        }
    return data


# ── CEO Chat ──────────────────────────────────────────────────────────────────

def ceo_chat(question: str, chat_history=None) -> dict:
    """
    CEO Chat: answer any question using RAG retrieval.
    Returns dict with 'content' and 'sources' keys.
    """
    log.info(f"  CEO Chat: '{question[:60]}...'")
    docs = hybrid_search(question, k=config.TOP_K_RETRIEVAL)

    if not docs:
        return {
            "content": (
                "No intelligence data available yet. "
                "Run Steps 1 and 2 in the sidebar first."
            ),
            "sources": [],
        }

    parts = [
        f"[{i}] {d['source']} — {d['title']}\n"
        f"Sentiment: {_sentiment_label(d.get('sentiment', 0))} "
        f"({d.get('sentiment', 0):.2f})\n"
        f"{d['text'][:600]}\nURL: {d.get('url', '#')}"
        for i, d in enumerate(docs, 1)
    ]
    context = "\n\n---\n\n".join(parts)

    history_text = ""
    if chat_history:
        for t in (chat_history or [])[-4:]:
            role = "CEO" if t.get("role") == "user" else "Advisor"
            history_text += f"{role}: {t.get('content','')[:300]}\n"

    prompt = (
        f"CONVERSATION HISTORY:\n{history_text or 'No prior conversation.'}\n\n"
        f"LIVE INTELLIGENCE:\n{context}\n\n"
        f"CEO QUESTION:\n{question}\n\n"
        "Be direct, specific, and actionable. Cite sources as [1],[2],[3]."
    )

    answer = _call_llm_safe(prompt, CEO_CHAT_SYSTEM)
    return {"content": answer, "sources": docs[:5]}


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_intelligence_engine() -> StrategicReport:
    """Run all 5 analysis tasks in sequence with rate-limit-safe delays."""
    log.info("=" * 60)
    log.info("AI CEO Intelligence Engine starting")
    log.info("=" * 60)
    report = StrategicReport()

    try:
        report.opportunities = analyse_opportunities()
        _task_pause("before risks")

        report.risks = analyse_risks()
        _task_pause("before trends")

        report.trends = analyse_trends()
        _task_pause("before recommendations")

        report.recommendations = generate_recommendations(
            report.opportunities, report.risks
        )
        _task_pause("before CEO briefing")

        report.ceo_briefing = generate_ceo_briefing(
            report.opportunities, report.risks, report.recommendations
        )

    except Exception as e:
        log.error(f"Intelligence engine error: {e}", exc_info=True)
        report.error = str(e)

    log.info("Intelligence Engine complete")
    return report


def report_to_dict(report: StrategicReport) -> dict:
    """Serialise StrategicReport to plain dict for Streamlit session state."""
    def ev_list(evs):
        return [asdict(e) for e in evs]

    def rec_to_dict(r: Recommendation) -> dict:
        return {
            "recommendation":  r.recommendation,
            "priority":        r.priority,
            "evidence":        ev_list(r.evidence),
            "expected_impact": r.expected_impact,
            "risk_assessment": asdict(r.risk_assessment),
            "time_horizon":    r.time_horizon,
        }

    return {
        "opportunities":   [{**asdict(o), "evidence": ev_list(o.evidence)} for o in report.opportunities],
        "risks":           [{**asdict(r), "evidence": ev_list(r.evidence)} for r in report.risks],
        "trends":          [asdict(t) for t in report.trends],
        "recommendations": [rec_to_dict(r) for r in report.recommendations],
        "ceo_briefing":    report.ceo_briefing,
        "error":           report.error,
    }
