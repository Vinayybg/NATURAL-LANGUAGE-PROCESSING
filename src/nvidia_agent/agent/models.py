"""
agent/models.py — Typed dataclasses for all strategic intelligence outputs.

Agent fields added to StrategicReport:
  agent_plan       → STEP 1: what the agent decided to investigate
  agent_validation → STEP 5: agent self-critique before presenting
"""

from dataclasses import dataclass, field


@dataclass
class Evidence:
    source:    str    # e.g. "NVIDIA Newsroom", "Reuters"
    excerpt:   str    # first 200 chars of the retrieved chunk
    url:       str    # clickable link to original article
    sentiment: float  # VADER compound score [-1.0 to +1.0]


@dataclass
class RiskAssessment:
    """Three-dimensional risk assessment — Task 6 requirement."""
    financial:   str   # capital risk and ROI uncertainty
    operational: str   # execution difficulty and team requirements
    strategic:   str   # competitive positioning risk


@dataclass
class Opportunity:
    title:            str
    description:      str
    impact_level:     str    # "High" | "Medium" | "Low"
    confidence_score: float  # 0.0 to 1.0
    evidence:         list[Evidence]
    expected_impact:  str


@dataclass
class Risk:
    title:            str
    description:      str
    category:         str    # competitive|regulatory|supply_chain|sentiment|macro|technology
    severity:         str    # "High" | "Medium" | "Low"
    confidence_score: float
    evidence:         list[Evidence]
    mitigation:       str


@dataclass
class Trend:
    title:        str
    description:  str
    relevance:    str
    source_count: int


@dataclass
class Recommendation:
    recommendation:  str
    priority:        str             # "High" | "Medium" | "Low"
    evidence:        list[Evidence]
    expected_impact: str
    risk_assessment: RiskAssessment
    time_horizon:    str


@dataclass
class ChatMessage:
    role:     str
    content:  str
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class StrategicReport:
    opportunities:    list[Opportunity]    = field(default_factory=list)
    risks:            list[Risk]           = field(default_factory=list)
    trends:           list[Trend]          = field(default_factory=list)
    recommendations:  list[Recommendation] = field(default_factory=list)
    ceo_briefing:     dict                 = field(default_factory=dict)
    agent_plan:       dict                 = field(default_factory=dict)
    agent_validation: dict                 = field(default_factory=dict)
    error:            str | None           = None
