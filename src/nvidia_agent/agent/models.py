"""
agent/models.py — Typed dataclasses for all strategic intelligence outputs.

Using dataclasses gives:
  - IDE autocomplete and type safety
  - asdict() serialisation for Streamlit session state
  - Clear schema the LLM is prompted to follow
  - Direct mapping between LLM JSON output and dashboard rendering
"""

from dataclasses import dataclass, field


@dataclass
class Evidence:
    """
    One source document supporting a finding.
    Every Opportunity, Risk, and Recommendation has up to 3 Evidence objects.
    Each links to the exact article chunk the LLM used to make the finding.
    """
    source:    str    # e.g. "NVIDIA Newsroom", "Reuters"
    excerpt:   str    # first 200 chars of the retrieved chunk
    url:       str    # clickable link to original article
    sentiment: float  # VADER compound score [-1.0 to +1.0]


@dataclass
class RiskAssessment:
    """
    Three-dimensional risk assessment for every recommendation.
    Directly maps to Task 6 exam requirement:
      - Financial risk  → what it costs if this recommendation goes wrong
      - Operational risk → execution difficulty and internal challenges
      - Strategic risk   → competitive and market positioning danger
    """
    financial:   str   # e.g. "High — requires $500M+ capital reallocation"
    operational: str   # e.g. "Medium — needs new team and 6-month ramp"
    strategic:   str   # e.g. "Low — aligns with existing CUDA moat strategy"


@dataclass
class Opportunity:
    """A strategic opportunity identified from live intelligence."""
    title:            str
    description:      str
    impact_level:     str             # "High" | "Medium" | "Low"
    confidence_score: float           # 0.0 to 1.0
    evidence:         list[Evidence]
    expected_impact:  str             # specific, quantified business outcome


@dataclass
class Risk:
    """A strategic risk identified from live intelligence."""
    title:            str
    description:      str
    category:         str             # competitive | regulatory | supply_chain | sentiment | macro
    severity:         str             # "High" | "Medium" | "Low"
    confidence_score: float
    evidence:         list[Evidence]
    mitigation:       str             # specific CEO action to mitigate


@dataclass
class Trend:
    """An emerging technology or industry trend."""
    title:        str
    description:  str
    relevance:    str   # why this matters specifically for NVIDIA
    source_count: int   # how many retrieved chunks mentioned this


@dataclass
class Recommendation:
    """
    A specific, evidence-backed strategic recommendation.
    Contains ALL fields required by Task 6:
      - recommendation  → the specific CEO action (starts with strong verb)
      - evidence        → 3 source documents supporting this recommendation
      - expected_impact → quantified business outcome (revenue, market share, etc.)
      - risk_assessment → Financial + Operational + Strategic risk breakdown
      - priority        → High / Medium / Low (used for sorting in dashboard)
      - time_horizon    → when to execute this action
    """
    recommendation:  str
    priority:        str             # "High" | "Medium" | "Low"
    evidence:        list[Evidence]
    expected_impact: str             # quantified: "15% revenue increase in 18 months"
    risk_assessment: RiskAssessment  # three-dimensional risk breakdown
    time_horizon:    str             # "Short-term (0-6 months)" | "Mid-term (6-18 months)" | "Long-term (18+ months)"


@dataclass
class ChatMessage:
    """One exchange in the CEO chat conversation."""
    role:     str              # "user" | "assistant"
    content:  str
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class StrategicReport:
    """The complete strategic intelligence report."""
    opportunities:   list[Opportunity]    = field(default_factory=list)
    risks:           list[Risk]           = field(default_factory=list)
    trends:          list[Trend]          = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    ceo_briefing:    dict                 = field(default_factory=dict)
    error:           str | None           = None
