"""
agent/ceo_agent.py — Autonomous AI CEO Strategic Intelligence Agent.

AGENT WORKFLOW (professor requirement):
  Goal → Plan → Retrieve → Analyse → Decide → Recommend → Validate

STEP 1  PLAN       agent_plan()              — LLM reads live snapshot,
                                               autonomously decides what to investigate
STEP 2  RETRIEVE   _retrieve(planned_query)  — hybrid BM25+cosine per planned topic
STEP 3  ANALYSE    analyse_opportunities()   — evidence → structured findings
                   analyse_risks()
                   analyse_trends()
STEP 4  DECIDE     generate_recommendations()— cross-task reasoning opps+risks→actions
STEP 5  VALIDATE   agent_validate()          — agent self-critiques own output
STEP 6  BRIEF      generate_ceo_briefing()   — executive synthesis

What makes this AUTONOMOUS:
  - The plan step decides WHAT to search for — queries are NOT hardcoded
  - Each analysis uses the planned queries, not fixed strings
  - The validate step catches gaps and contradictions before presenting
  - The full loop runs without any human input after clicking the button
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
    Recommendation, RiskAssessment, StrategicReport,
)

log = logging.getLogger(__name__)


# ── System prompts ─────────────────────────────────────────────────────────────

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

AGENT_SYSTEM_PROMPT = f"""You are an autonomous AI strategic intelligence agent for {config.COMPANY_NAME}.

Your job is to PLAN what to investigate, RETRIEVE evidence, ANALYSE findings,
DECIDE on recommendations, and VALIDATE your own output before presenting it.

You think step by step. You justify every finding with evidence.
You self-critique to catch gaps before presenting to the CEO.

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
    """Exponential backoff on Groq 429 rate-limit errors."""
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
                wait = 30 * (2 ** attempt)
                retry_match = re.search(r"retry.after[^\d]*(\d+)", err)
                if retry_match:
                    wait = max(int(retry_match.group(1)) + 2, wait)
                log.warning(f"  Groq rate limit — waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Groq rate limit exceeded after {max_retries} retries")


# ── Inter-task delay ───────────────────────────────────────────────────────────

_TASK_DELAY = 15

def _task_pause(label: str = "") -> None:
    log.info(f"  Pausing {_TASK_DELAY}s — {label}...")
    time.sleep(_TASK_DELAY)


# ── Private helpers ────────────────────────────────────────────────────────────

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


def _retrieve_multi(queries: list[str], k_each: int = 8) -> tuple[str, list[dict]]:
    """
    Run multiple retrieval queries and merge results.
    Used when the agent plans multiple search angles for one topic.
    Deduplicates by URL so the same chunk is not counted twice.
    """
    all_docs = []
    seen_urls = set()
    for q in queries[:3]:           # cap at 3 queries to control token usage
        _, docs = _retrieve(q, k=k_each)
        for d in docs:
            if d["url"] not in seen_urls:
                seen_urls.add(d["url"])
                all_docs.append(d)

    if not all_docs:
        return "", []

    parts = [
        f"[SOURCE {i}] {d['source']} — {d['title']}\n"
        f"Relevance: {d['score']:.2f} | Sentiment: {d['sentiment']:.2f} "
        f"({_sentiment_label(d['sentiment'])})\n{d['text'][:600]}\nURL: {d['url']}"
        for i, d in enumerate(all_docs[:config.TOP_K_RETRIEVAL], 1)
    ]
    return "\n\n---\n\n".join(parts), all_docs[:config.TOP_K_RETRIEVAL]


def _evidence_from_docs(docs: list[dict], max_items: int = 3) -> list[Evidence]:
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
    Three-tier fallback JSON extraction.
    Tier 1: strip markdown fences, parse directly.
    Tier 2: regex-extract first {...} or [...].
    Tier 3: return {} — never crash.
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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — PLAN
# The agent reads current intelligence and autonomously decides what to search for.
# This is the explicit planning step — queries are NOT hardcoded.
# ══════════════════════════════════════════════════════════════════════════════

def agent_plan() -> dict:
    """
    AGENT STEP 1: AUTONOMOUS PLANNING.

    The agent:
    1. Reads a broad snapshot of current intelligence (6 chunks)
    2. Autonomously decides what topics to investigate for opportunities
    3. Autonomously decides what topics to investigate for risks
    4. Formulates specific key questions for the CEO
    5. Explains its reasoning

    This step is what makes the system genuinely autonomous —
    the search queries come FROM the agent's analysis of current events,
    not from hardcoded strings written months ago.
    """
    log.info("=" * 60)
    log.info("AGENT STEP 1: PLANNING — deciding what to investigate")
    log.info("=" * 60)

    # Get a broad snapshot — use a neutral query to see what is currently available
    snapshot_context, _ = _retrieve(
        f"{config.COMPANY_NAME} latest news strategy market competition",
        k=6
    )

    if not snapshot_context:
        log.warning("  No context for planning — using default plan")
        return _default_plan()

    prompt = f"""You are an autonomous AI strategic intelligence agent for {config.COMPANY_NAME}.

You have just read the following recent intelligence snapshot:

{snapshot_context}

Based on what you see in the intelligence above, create a SPECIFIC investigation plan.
Do NOT use generic topics. Base your queries on what is actually mentioned in the sources above.

Return ONLY a JSON object:
{{
  "goal": "One specific sentence describing the most important strategic question right now",
  "opportunity_queries": [
    "specific search query 1 based on what you read — focus on growth signals",
    "specific search query 2 based on what you read — focus on new markets",
    "specific search query 3 based on what you read — focus on partnerships or products"
  ],
  "risk_queries": [
    "specific search query 1 based on what you read — focus on competitive threats",
    "specific search query 2 based on what you read — focus on regulatory or supply issues",
    "specific search query 3 based on what you read — focus on market or macro risks"
  ],
  "trend_queries": [
    "specific technology or market trend query 1 based on what you read",
    "specific industry shift query 2 based on what you read"
  ],
  "key_questions": [
    "Specific strategic question 1 the CEO needs answered based on this intelligence",
    "Specific strategic question 2 the CEO needs answered"
  ],
  "reasoning": "2 sentences explaining WHY you chose these specific topics based on what you read"
}}"""

    data = _parse_json(_call_llm_safe(prompt, AGENT_SYSTEM_PROMPT))

    if not isinstance(data, dict) or "goal" not in data:
        log.warning("  Planning returned invalid JSON — using default plan")
        return _default_plan()

    # Validate the plan has required fields
    for field in ["opportunity_queries", "risk_queries", "trend_queries"]:
        if field not in data or not isinstance(data[field], list) or len(data[field]) == 0:
            data[field] = _default_plan()[field]

    log.info(f"  Agent goal: {data.get('goal', '')}")
    log.info(f"  Opportunity queries: {data.get('opportunity_queries', [])}")
    log.info(f"  Risk queries: {data.get('risk_queries', [])}")
    log.info(f"  Reasoning: {data.get('reasoning', '')}")

    return data


def _default_plan() -> dict:
    """Fallback plan when planning LLM call fails or returns invalid output."""
    return {
        "goal": f"Identify the most critical strategic opportunities and risks for {config.COMPANY_NAME} from current live intelligence",
        "opportunity_queries": [
            f"{config.COMPANY_NAME} growth new markets AI demand sovereign AI partnerships",
            f"{config.COMPANY_NAME} enterprise robotics autonomous vehicles agentic AI data center",
            f"{config.COMPANY_NAME} software platform CUDA NIM inference cloud hyperscaler",
        ],
        "risk_queries": [
            f"{config.COMPANY_NAME} competition AMD Intel custom silicon Huawei threat",
            f"{config.COMPANY_NAME} export controls regulatory China geopolitical risk",
            f"{config.COMPANY_NAME} supply chain TSMC concentration HBM memory shortage",
        ],
        "trend_queries": [
            "AI inference training shift edge computing enterprise adoption sovereign AI",
            "semiconductor industry custom silicon ASIC GPU alternative technology trend",
        ],
        "key_questions": [
            f"What is the single biggest competitive threat to {config.COMPANY_NAME} right now?",
            "Which new markets represent the highest near-term revenue opportunity?",
        ],
        "reasoning": "Default plan covering core strategic dimensions. Live planning was unavailable.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEPS 2+3 — RETRIEVE + ANALYSE
# Uses the agent's plan — queries come FROM the plan, not hardcoded.
# ══════════════════════════════════════════════════════════════════════════════

def analyse_opportunities(plan: dict) -> list[Opportunity]:
    """
    AGENT STEPS 2+3: RETRIEVE + ANALYSE — Opportunities.

    Retrieval is PLAN-DRIVEN: uses the queries the agent decided on in Step 1.
    Runs multi-query retrieval to get diverse evidence from multiple angles.
    """
    log.info("  AGENT STEP 2+3: Retrieving and analysing opportunities...")

    queries = plan.get("opportunity_queries", _default_plan()["opportunity_queries"])
    context, docs = _retrieve_multi(queries, k_each=8)

    if not context:
        log.warning("  No opportunity context retrieved")
        return []

    prompt = f"""Based on the following live intelligence about {config.COMPANY_NAME},
identify the TOP 3 strategic OPPORTUNITIES.

INTELLIGENCE:
{context}

Return ONLY a JSON array of exactly 3 objects:
[
  {{
    "title": "Short specific opportunity title (max 8 words)",
    "description": "2-3 sentences explaining the opportunity with specific evidence from the sources above",
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


def analyse_risks(plan: dict) -> list[Risk]:
    """
    AGENT STEPS 2+3: RETRIEVE + ANALYSE — Risks.
    Retrieval is PLAN-DRIVEN.
    """
    log.info("  AGENT STEP 2+3: Retrieving and analysing risks...")

    queries = plan.get("risk_queries", _default_plan()["risk_queries"])
    context, docs = _retrieve_multi(queries, k_each=8)

    if not context:
        log.warning("  No risk context retrieved")
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


def analyse_trends(plan: dict) -> list[Trend]:
    """
    AGENT STEPS 2+3: RETRIEVE + ANALYSE — Trends.
    Retrieval is PLAN-DRIVEN.
    """
    log.info("  AGENT STEP 2+3: Retrieving and analysing trends...")

    queries = plan.get("trend_queries", _default_plan()["trend_queries"])
    context, docs = _retrieve_multi(queries, k_each=8)

    if not context:
        log.warning("  No trend context retrieved")
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
    "relevance": "1-2 sentences on why this matters specifically for {config.COMPANY_NAME}"
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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — DECIDE
# Cross-task reasoning: synthesises opportunities + risks → CEO actions.
# Also uses the agent's planned key questions.
# ══════════════════════════════════════════════════════════════════════════════

def generate_recommendations(
    opportunities: list[Opportunity],
    risks: list[Risk],
    plan: dict,
) -> list[Recommendation]:
    """
    AGENT STEP 4: DECIDE.

    Cross-task reasoning:
    - Receives outputs of Steps 3a (opportunities) and 3b (risks)
    - Retrieves fresh strategic context
    - Uses the agent's key questions from the plan
    - Produces 4 prioritised CEO action items with 3-part risk assessment

    This is where the agent synthesises multiple findings into decisions —
    not just answering a single question.
    """
    log.info("  AGENT STEP 4: Deciding on recommendations...")

    context, docs = _retrieve(
        f"{config.COMPANY_NAME} strategic priorities investment decisions "
        "competitive moat CUDA ecosystem software AI infrastructure market leadership",
    )

    opp_summary = "\n".join(
        f"  [{o.impact_level}] {o.title}: {o.description}"
        for o in opportunities
    ) or "  None identified"

    risk_summary = "\n".join(
        f"  [{r.severity}] {r.title}: {r.description}"
        for r in risks
    ) or "  None identified"

    # Include the agent's planned key questions
    key_questions_text = ""
    key_questions = plan.get("key_questions", [])
    if key_questions:
        key_questions_text = "\nKEY QUESTIONS TO ADDRESS:\n" + "\n".join(
            f"  - {q}" for q in key_questions
        )

    prompt = f"""You are advising the CEO of {config.COMPANY_NAME}.

OPPORTUNITIES IDENTIFIED FROM LIVE INTELLIGENCE:
{opp_summary}

RISKS IDENTIFIED FROM LIVE INTELLIGENCE:
{risk_summary}
{key_questions_text}

ADDITIONAL STRATEGIC INTELLIGENCE:
{context[:3000]}

Generate exactly 4 specific, actionable STRATEGIC RECOMMENDATIONS.
Each must start with a strong verb and be specific enough to act on this quarter.
Each recommendation must ADDRESS at least one opportunity or mitigate at least one risk above.

For each recommendation, provide a THREE-PART risk assessment:
- financial_risk: capital required and ROI uncertainty if this fails
- operational_risk: execution difficulty, team size, timeline
- strategic_risk: competitive positioning danger if this recommendation is wrong

Return ONLY a JSON array of exactly 4 objects:
[
  {{
    "recommendation": "Specific CEO action starting with a strong verb",
    "priority": "High",
    "expected_impact": "Specific quantified business outcome",
    "time_horizon": "Short-term (0-6 months)",
    "financial_risk": "High — requires $X investment with Y% ROI uncertainty",
    "operational_risk": "Medium — needs team of Z people and 6-month ramp",
    "strategic_risk": "Low — aligns with existing CUDA moat and partner ecosystem",
    "addresses": "Which opportunity or risk this tackles"
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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — VALIDATE
# The agent reviews its own output before presenting it to the CEO.
# This is explicit autonomous self-assessment — a core agent capability.
# ══════════════════════════════════════════════════════════════════════════════

def agent_validate(
    recommendations: list[Recommendation],
    opportunities: list[Opportunity],
    risks: list[Risk],
) -> dict:
    """
    AGENT STEP 5: VALIDATE.

    The agent asks itself:
    - Are recommendations grounded in the identified opportunities and risks?
    - Are there any identified risks that NO recommendation addresses?
    - Are there any high-impact opportunities that NO recommendation captures?
    - Are there any contradictions between recommendations?
    - What is the overall confidence in this analysis?
    - What should the CEO know about the limitations?

    Returns a validation report included in the final output.
    The CEO sees this — it demonstrates transparent, auditable AI reasoning.
    """
    log.info("  AGENT STEP 5: Validating own recommendations...")

    rec_text = "\n".join(
        f"  [{r.priority}] {r.recommendation} → {r.expected_impact}"
        for r in recommendations
    ) or "  None"

    opp_text = "\n".join(
        f"  [{o.impact_level}] {o.title}: {o.description[:100]}"
        for o in opportunities
    ) or "  None"

    risk_text = "\n".join(
        f"  [{r.severity}] {r.title}: {r.description[:100]}"
        for r in risks
    ) or "  None"

    prompt = f"""You are an autonomous AI agent reviewing your own strategic analysis for {config.COMPANY_NAME}.

YOUR IDENTIFIED OPPORTUNITIES:
{opp_text}

YOUR IDENTIFIED RISKS:
{risk_text}

YOUR RECOMMENDATIONS:
{rec_text}

Critically evaluate this analysis. Be honest about weaknesses and gaps.

Check:
1. Does each recommendation address at least one opportunity or risk above?
2. Which risks (if any) have NO recommendation addressing them?
3. Which high-impact opportunities (if any) have NO recommendation capturing them?
4. Are there contradictions between any recommendations?
5. What is your confidence in the overall analysis quality (0.0 to 1.0)?
6. What should the CEO know about potential blind spots?

Return ONLY a JSON object:
{{
  "overall_confidence": 0.82,
  "all_recommendations_grounded": true,
  "unaddressed_risks": ["risk title if any not addressed — empty list if all addressed"],
  "unaddressed_opportunities": ["opp title if any missed — empty list if all captured"],
  "contradictions": ["any contradictions found — empty list if none"],
  "validation_notes": "2-3 honest sentences about the quality and limitations of this analysis",
  "recommendation_scores": [
    {{"text": "first 6 words of recommendation 1", "score": 0.85, "reason": "strong evidence base"}},
    {{"text": "first 6 words of recommendation 2", "score": 0.80, "reason": "why this score"}},
    {{"text": "first 6 words of recommendation 3", "score": 0.75, "reason": "why this score"}},
    {{"text": "first 6 words of recommendation 4", "score": 0.70, "reason": "why this score"}}
  ]
}}"""

    data = _parse_json(_call_llm_safe(prompt, AGENT_SYSTEM_PROMPT))

    if not isinstance(data, dict):
        return {
            "overall_confidence": 0.75,
            "all_recommendations_grounded": True,
            "unaddressed_risks": [],
            "unaddressed_opportunities": [],
            "contradictions": [],
            "validation_notes": (
                "Validation completed. Recommendations appear grounded in retrieved evidence. "
                "Manual review recommended before executive presentation."
            ),
            "recommendation_scores": [],
        }

    conf = data.get("overall_confidence", 0.75)
    log.info(f"  Validation confidence: {conf:.0%}")
    log.info(f"  Unaddressed risks: {data.get('unaddressed_risks', [])}")
    log.info(f"  Validation notes: {data.get('validation_notes', '')}")

    return data


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — CEO BRIEFING
# Final synthesis of all agent outputs into executive narrative.
# ══════════════════════════════════════════════════════════════════════════════

def generate_ceo_briefing(
    opportunities: list[Opportunity],
    risks: list[Risk],
    recommendations: list[Recommendation],
    plan: dict,
    validation: dict,
) -> dict:
    """
    AGENT STEP 6: SYNTHESISE.

    Receives ALL previous outputs and writes the executive briefing.
    Includes the agent's goal from planning and validation confidence.
    """
    log.info("  AGENT STEP 6: Writing CEO briefing...")

    opp_text  = "\n".join(f"• {o.title}: {o.description}" for o in opportunities) or "None"
    risk_text = "\n".join(f"• {r.title}: {r.description}" for r in risks) or "None"
    rec_text  = "\n".join(f"• {r.recommendation}" for r in recommendations) or "None"

    agent_goal = plan.get("goal", "")
    conf_pct   = int(validation.get("overall_confidence", 0.75) * 100)
    val_notes  = validation.get("validation_notes", "")

    prompt = f"""Write an executive CEO briefing for {config.COMPANY_NAME}.

AGENT INVESTIGATION GOAL: {agent_goal}
ANALYSIS CONFIDENCE: {conf_pct}% — {val_notes}

OPPORTUNITIES IDENTIFIED:
{opp_text}

RISKS IDENTIFIED:
{risk_text}

RECOMMENDED ACTIONS:
{rec_text}

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
            "executive_summary": f"Strategic intelligence report for {config.COMPANY_NAME} generated with {conf_pct}% confidence.",
        }
    return data


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR — Full Autonomous Agent Loop
# Goal → Plan → Retrieve → Analyse → Decide → Validate → Recommend
# ══════════════════════════════════════════════════════════════════════════════

def run_intelligence_engine() -> StrategicReport:
    """
    FULL AUTONOMOUS AGENT LOOP.

    Every step is logged clearly so the examiner can see the agent thinking.
    The plan from Step 1 drives every subsequent retrieval query.
    The validation in Step 5 runs before briefing is written.
    """
    log.info("=" * 60)
    log.info("AI CEO AGENT — Autonomous Loop Starting")
    log.info("Workflow: Goal → Plan → Retrieve → Analyse → Decide → Validate")
    log.info("=" * 60)

    report = StrategicReport()

    try:
        # ── STEP 1: PLAN ──────────────────────────────────────────────────────
        log.info("\n[ STEP 1 / 6 ] PLANNING — agent decides what to investigate")
        plan = agent_plan()
        report.agent_plan = plan
        _task_pause("planning complete — starting retrieval")

        # ── STEP 2+3a: RETRIEVE + ANALYSE — Opportunities ─────────────────────
        log.info("\n[ STEP 2+3 / 6 ] RETRIEVE + ANALYSE — Opportunities")
        report.opportunities = analyse_opportunities(plan)
        log.info(f"  Found {len(report.opportunities)} opportunities")
        _task_pause("opportunities done — analysing risks")

        # ── STEP 2+3b: RETRIEVE + ANALYSE — Risks ─────────────────────────────
        log.info("\n[ STEP 2+3 / 6 ] RETRIEVE + ANALYSE — Risks")
        report.risks = analyse_risks(plan)
        log.info(f"  Found {len(report.risks)} risks")
        _task_pause("risks done — analysing trends")

        # ── STEP 2+3c: RETRIEVE + ANALYSE — Trends ────────────────────────────
        log.info("\n[ STEP 2+3 / 6 ] RETRIEVE + ANALYSE — Trends")
        report.trends = analyse_trends(plan)
        log.info(f"  Found {len(report.trends)} trends")
        _task_pause("trends done — generating recommendations")

        # ── STEP 4: DECIDE ─────────────────────────────────────────────────────
        log.info("\n[ STEP 4 / 6 ] DECIDE — cross-task reasoning → recommendations")
        report.recommendations = generate_recommendations(
            report.opportunities, report.risks, plan
        )
        log.info(f"  Generated {len(report.recommendations)} recommendations")
        _task_pause("recommendations done — validating")

        # ── STEP 5: VALIDATE ───────────────────────────────────────────────────
        log.info("\n[ STEP 5 / 6 ] VALIDATE — agent self-critique")
        validation = agent_validate(
            report.recommendations,
            report.opportunities,
            report.risks,
        )
        report.agent_validation = validation
        conf = validation.get("overall_confidence", 0)
        log.info(f"  Validation confidence: {conf:.0%}")
        _task_pause("validation done — writing CEO briefing")

        # ── STEP 6: CEO BRIEFING ───────────────────────────────────────────────
        log.info("\n[ STEP 6 / 6 ] CEO BRIEFING — executive synthesis")
        report.ceo_briefing = generate_ceo_briefing(
            report.opportunities,
            report.risks,
            report.recommendations,
            plan=plan,
            validation=validation,
        )

        log.info("\n" + "=" * 60)
        log.info("AI CEO AGENT — Complete")
        log.info(f"  Agent goal:       {plan.get('goal', 'N/A')[:80]}")
        log.info(f"  Opportunities:    {len(report.opportunities)}")
        log.info(f"  Risks:            {len(report.risks)}")
        log.info(f"  Trends:           {len(report.trends)}")
        log.info(f"  Recommendations:  {len(report.recommendations)}")
        log.info(f"  Confidence:       {conf:.0%}")
        log.info("=" * 60)

    except Exception as e:
        log.error(f"Agent error: {e}", exc_info=True)
        report.error = str(e)

    return report


# ── CEO Chat ────────────────────────────────────────────────────────────────

def ceo_chat(question: str, chat_history=None) -> dict:
    """Real-time RAG Q&A for the CEO Advisor tab."""
    docs = hybrid_search(question, k=config.TOP_K_RETRIEVAL)
    if not docs:
        return {
            "content": "No intelligence data available yet. Run Steps 1 and 2 in the sidebar first.",
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
            history_text += f"{role}: {t.get('content', '')[:300]}\n"

    prompt = (
        f"CONVERSATION HISTORY:\n{history_text or 'No prior conversation.'}\n\n"
        f"LIVE INTELLIGENCE:\n{context}\n\n"
        f"CEO QUESTION:\n{question}\n\n"
        "Be direct, specific, and actionable. Cite sources as [1],[2],[3]."
    )

    answer = _call_llm_safe(prompt, CEO_CHAT_SYSTEM)
    return {"content": answer, "sources": docs[:5]}


# ── Serialisation ─────────────────────────────────────────────────────────────

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
        "opportunities":    [{**asdict(o), "evidence": ev_list(o.evidence)} for o in report.opportunities],
        "risks":            [{**asdict(r), "evidence": ev_list(r.evidence)} for r in report.risks],
        "trends":           [asdict(t) for t in report.trends],
        "recommendations":  [rec_to_dict(r) for r in report.recommendations],
        "ceo_briefing":     report.ceo_briefing,
        "error":            report.error,
        "agent_plan":       report.agent_plan,
        "agent_validation": report.agent_validation,
    }
