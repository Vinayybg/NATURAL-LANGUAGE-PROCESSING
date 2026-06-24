"""
app.py — NVIDIA CEO Strategic Intelligence Platform
Premium executive dashboard — fully meets NLP exam requirements.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sqlite3
import logging
from datetime import datetime, timezone

from nvidia_agent import config
import collector
import processor
from nvidia_agent.agent.ceo_agent import run_intelligence_engine, report_to_dict, _call_llm_safe
from nvidia_agent.storage.vector_store import hybrid_search
from nvidia_agent.collectors.base import get_article_count, get_source_breakdown

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  — load NVIDIA logo from local file (same folder as app.py)
# ─────────────────────────────────────────────────────────────────────────────
import base64 as _b64
from pathlib import Path as _Path

def _load_logo_b64() -> str:
    """Load the NVIDIA logo PNG from the project folder and return as base64 data URI."""
    logo_path = _Path(__file__).parent / "nvidia-logo-green-3840x2160-24758.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return "data:image/png;base64," + _b64.b64encode(f.read()).decode()
    # Fallback: CSS wordmark if file not found
    return ""

NVIDIA_LOGO_B64 = _load_logo_b64()

st.set_page_config(
    page_title="NVIDIA · CEO Strategic Intelligence",
    page_icon="🟩",
    layout="wide",
    initial_sidebar_state="expanded",  # sidebar always starts open
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 2rem !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #060b14 !important;
    border-right: 1px solid #131f35 !important;
}
[data-testid="stSidebar"] * { color: #b8c4d8 !important; }

/* Tabs */
[data-testid="stTabs"] button {
    font-size: 13px !important; font-weight: 600 !important;
    letter-spacing: 0.8px !important; color: #445066 !important;
    padding: 10px 22px !important; text-transform: uppercase !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #76b900 !important;
    border-bottom: 2px solid #76b900 !important;
}

/* Metric cards — force equal size */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0b1628 0%, #0e1c34 100%) !important;
    border: 1px solid #192a45 !important;
    border-radius: 12px !important;
    padding: 20px 22px !important;
    min-height: 100px !important;
}
[data-testid="stMetricLabel"] {
    color: #445066 !important; font-size: 10px !important;
    font-weight: 700 !important; letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #e2eaf8 !important; font-size: 1.7rem !important; font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { display: none !important; }

/* Columns — equal width fix */
[data-testid="column"] { min-width: 0 !important; }

/* Primary button */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #76b900 0%, #5a8f00 100%) !important;
    border: none !important; color: #fff !important;
    font-weight: 700 !important; letter-spacing: 0.3px !important;
    border-radius: 8px !important; padding: 10px 20px !important;
    box-shadow: 0 4px 15px rgba(118,185,0,0.28) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(118,185,0,0.42) !important;
}

/* Secondary buttons */
[data-testid="stButton"] > button:not([kind="primary"]) {
    background: #0b1628 !important; border: 1px solid #192a45 !important;
    color: #b8c4d8 !important; border-radius: 8px !important;
    font-weight: 500 !important; font-size: 12px !important;
}
[data-testid="stButton"] > button:not([kind="primary"]):hover {
    border-color: #76b900 !important; color: #76b900 !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #0b1628 !important; border: 1px solid #192a45 !important;
    border-radius: 12px !important; margin-bottom: 10px !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important; color: #b8c4d8 !important; padding: 14px 18px !important;
}

/* Text input */
[data-testid="stTextInput"] input {
    background: #0b1628 !important; border: 1px solid #192a45 !important;
    color: #e2eaf8 !important; border-radius: 10px !important;
    font-size: 14px !important; padding: 12px 16px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #76b900 !important;
    box-shadow: 0 0 0 2px rgba(118,185,0,0.18) !important;
}

hr { border-color: #131f35 !important; margin: 22px 0 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #060b14; }
::-webkit-scrollbar-thumb { background: #192a45; border-radius: 3px; }
.stPlotlyChart { border-radius: 12px; overflow: hidden; }

/* ── Sidebar toggle arrow — green styling ── */
[data-testid="stSidebarCollapseButton"] button {
    color: #76b900 !important;
    border-color: #192a45 !important;
}

/* ── Hero ── */
.nx-hero {
    background: linear-gradient(135deg, #040a15 0%, #071020 50%, #0a1830 100%);
    border-bottom: 1px solid #131f35;
    padding: 28px 0 22px 0;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 28px;
}
.nx-hero-logo-wrap {
    flex-shrink: 0;
}
.nx-hero-text {
    flex: 1;
    min-width: 0;
}
.nx-pill {
    display: inline-block;
    background: rgba(118,185,0,0.1);
    border: 1px solid rgba(118,185,0,0.28);
    color: #76b900; font-size: 11px; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    border-radius: 20px; padding: 3px 10px; margin-bottom: 8px;
}
.nx-hero-title {
    font-size: 2.4rem; font-weight: 800; color: #fff;
    letter-spacing: -0.5px; margin: 0; line-height: 1.2;
}
.nx-hero-subtitle { font-size: 16px; color: #8a9ab8; margin-top: 4px; letter-spacing: 0.3px; }
.nx-hero-ts { font-size: 13px; color: #445066; margin-top: 8px; }

/* ── Section header ── */
.nx-section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 28px 0 14px 0; padding-bottom: 10px;
    border-bottom: 1px solid #131f35;
}
.nx-section-icon {
    width: 26px; height: 26px;
    background: rgba(118,185,0,0.1);
    border: 1px solid rgba(118,185,0,0.22);
    border-radius: 6px; display: flex;
    align-items: center; justify-content: center; font-size: 12px;
}
.nx-section-title { font-size: 1.1rem; font-weight: 700; color: #e2eaf8; margin: 0; }

/* ── Company Overview card ── */
.nx-overview-card {
    background: linear-gradient(135deg, #071020 0%, #0a1830 100%);
    border: 1px solid #192a45; border-radius: 14px; padding: 22px 26px;
    margin-bottom: 20px; position: relative; overflow: hidden;
}
.nx-overview-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #76b900, #3b82f6, transparent);
}
.nx-overview-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px; margin-top: 16px;
}
.nx-overview-item {}
.nx-overview-label { font-size: 9px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #2a3a55; margin-bottom: 4px; }
.nx-overview-value { font-size: 13px; font-weight: 600; color: #8a9ab8; }

/* ── Intelligence cards ── */
.nx-card {
    background: linear-gradient(135deg, #0b1628 0%, #0d1c35 100%);
    border: 1px solid #192a45; border-radius: 14px;
    padding: 18px 20px; margin-bottom: 12px;
    position: relative; overflow: hidden;
}
.nx-card:hover { border-color: #263d5e; }
.nx-card-green  { border-left: 3px solid #76b900; }
.nx-card-red    { border-left: 3px solid #ef4444; }
.nx-card-amber  { border-left: 3px solid #f59e0b; }
.nx-card-blue   { border-left: 3px solid #3b82f6; }
.nx-card-title  { font-size: 1.05rem; font-weight: 700; color: #e2eaf8; margin-bottom: 8px; }
.nx-card-body   { font-size: 15px; color: #7a8ba8; line-height: 1.65; }
.nx-card-footer { font-size: 11px; color: #2a3a55; margin-top: 12px; padding-top: 12px; border-top: 1px solid #131f35; }

/* ── Badges ── */
.nx-badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 20px;
    font-size: 9px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;
}
.nx-badge-red   { background: rgba(239,68,68,0.1);  border: 1px solid rgba(239,68,68,0.3);  color: #ef4444; }
.nx-badge-amber { background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); color: #f59e0b; }
.nx-badge-green { background: rgba(118,185,0,0.1);  border: 1px solid rgba(118,185,0,0.3);  color: #76b900; }
.nx-badge-blue  { background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.3); color: #3b82f6; }
.nx-badge-gray  { background: rgba(100,116,139,0.1);border: 1px solid rgba(100,116,139,0.3);color: #64748b; }

/* ── Confidence bar ── */
.nx-conf { margin: 12px 0 4px; }
.nx-conf-label { font-size: 11px; color: #2a3a55; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 5px; text-transform: uppercase; }
.nx-conf-track { background: #0e1c34; border-radius: 4px; height: 5px; }
.nx-conf-fill  { height: 100%; border-radius: 4px; }

/* ── Evidence block ── */
.nx-evidence-block {
    background: #071020;
    border: 1px solid #1e2d4a;
    border-left: 3px solid #3b82f6;
    border-radius: 10px;
    padding: 16px 18px;
    margin-top: 16px;
}
.nx-evidence-title {
    font-size: 12px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: #76b900; margin-bottom: 12px;
    display: flex; align-items: center; gap: 6px;
}
.nx-evidence-item {
    display: flex; gap: 12px; align-items: flex-start;
    padding: 10px 0; border-bottom: 1px solid #0e1c34;
}
.nx-evidence-item:last-child { border-bottom: none; padding-bottom: 0; }
.nx-evidence-dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 4px; flex-shrink: 0; }
.nx-evidence-source { font-size: 14px; font-weight: 700; color: #76b900; margin-bottom: 4px; }
.nx-evidence-source a { color: #76b900; text-decoration: none; font-weight: 700; }
.nx-evidence-source a:hover { text-decoration: underline; }
.nx-evidence-excerpt { font-size: 14px; color: #6b7a99; line-height: 1.6; }
.nx-evidence-sent { font-size: 12px; font-weight: 600; margin-top: 4px; }

/* ── News feed ── */
.nx-news-item {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 11px 0; border-bottom: 1px solid #0e1c34;
}
.nx-news-dot { width: 6px; height: 6px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; }
.nx-news-title { font-size: 15px; font-weight: 500; color: #c8d4e8; line-height: 1.5; }
.nx-news-title a { color: #b8c4d8; text-decoration: none; }
.nx-news-title a:hover { color: #76b900; }
.nx-news-meta { font-size: 12px; color: #445066; margin-top: 4px; }

/* ── Briefing ── */
.nx-briefing {
    background: linear-gradient(135deg, #040a15 0%, #071020 100%);
    border: 1px solid #192a45; border-radius: 16px;
    padding: 26px 30px; margin-bottom: 20px; position: relative; overflow: hidden;
}
.nx-briefing::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #76b900 0%, #3b82f6 50%, transparent 100%);
}
.nx-briefing-label { font-size: 9px; font-weight: 700; letter-spacing: 2px; color: #76b900; text-transform: uppercase; margin-bottom: 12px; }
.nx-briefing-text  { font-size: 16px; line-height: 1.85; color: #b8c4d8; }
.nx-briefing-col   { background: #0b1628; border: 1px solid #131f35; border-radius: 12px; padding: 18px 20px; }
.nx-briefing-col-title { font-size: 13px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; color: #6b7a99; margin-bottom: 12px; }
.nx-briefing-col-body  { font-size: 15px; color: #7a8ba8; line-height: 1.7; }

/* ── Trend card ── */
.nx-trend { background: #0b1628; border: 1px solid #131f35; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; }
.nx-trend-title { font-size: 16px; font-weight: 700; color: #e2eaf8; margin-bottom: 8px; }
.nx-trend-body  { font-size: 14px; color: #8a9ab8; line-height: 1.65; }
.nx-trend-rel   { font-size: 13px; color: #76b900; margin-top: 8px; font-weight: 500; }

/* ── Recommendation card ── */
.nx-rec { background: linear-gradient(135deg, #0b1628 0%, #0d1c35 100%); border: 1px solid #192a45; border-radius: 14px; padding: 16px 18px; margin-bottom: 10px; }
.nx-rec-num    { font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: #2a3a55; margin-bottom: 7px; }
.nx-rec-text   { font-size: 16px; font-weight: 600; color: #e2eaf8; line-height: 1.45; margin-bottom: 8px; }
.nx-rec-impact { font-size: 14px; color: #5a6a88; line-height: 1.5; margin-bottom: 10px; }
.nx-rec-meta   { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 10px; padding-top: 10px; border-top: 1px solid #131f35; }
.nx-rec-tag    { font-size: 12px; color: #2a3a55; background: #0e1c34; border-radius: 6px; padding: 3px 8px; font-weight: 500; }

/* ── Chat ── */
.nx-chat-header {
    background: linear-gradient(135deg, #040a15 0%, #071020 100%);
    border: 1px solid #192a45; border-radius: 16px;
    padding: 22px 26px; margin-bottom: 22px;
    display: flex; align-items: center; gap: 16px;
    position: relative; overflow: hidden;
}
.nx-chat-header::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #76b900, #3b82f6, transparent);
}
.nx-chat-icon { width: 46px; height: 46px; background: rgba(118,185,0,0.1); border: 1px solid rgba(118,185,0,0.28); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 20px; flex-shrink: 0; }
.nx-chat-title { font-size: 1.2rem; font-weight: 800; color: #e2eaf8; }
.nx-chat-sub   { font-size: 13px; color: #445066; margin-top: 2px; }
.nx-chat-live  { margin-left: auto; display: flex; align-items: center; gap: 6px; font-size: 10px; color: #76b900; font-weight: 700; letter-spacing: 1px; }
.nx-chat-dot   { width: 7px; height: 7px; background: #76b900; border-radius: 50%; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.35; } }

.nx-sug-label  { font-size: 9px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #2a3a55; margin-bottom: 10px; }
.nx-msg-user   { background: linear-gradient(135deg, #0b1e38 0%, #0d2245 100%); border: 1px solid #1a3860; border-radius: 14px 14px 4px 14px; padding: 14px 18px; margin: 10px 0 10px 12%; color: #b8c4d8; font-size: 15px; line-height: 1.6; }
.nx-msg-user-lbl { font-size: 9px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: #3b82f6; margin-bottom: 6px; }
.nx-msg-ai     { background: linear-gradient(135deg, #0b1628 0%, #0d1c35 100%); border: 1px solid #192a45; border-radius: 14px 14px 14px 4px; padding: 14px 18px; margin: 10px 12% 4px 0; color: #b8c4d8; font-size: 15px; line-height: 1.7; }
.nx-msg-ai-lbl { font-size: 9px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: #76b900; margin-bottom: 6px; }
.nx-src-chip   { display: inline-block; background: rgba(118,185,0,0.07); border: 1px solid rgba(118,185,0,0.18); color: #76b900; border-radius: 20px; padding: 3px 12px; font-size: 13px; font-weight: 500; margin: 3px 3px 3px 0; text-decoration: none; }
.nx-src-chip:hover { background: rgba(118,185,0,0.14); }
.nx-chat-empty { text-align: center; padding: 60px 40px; color: #2a3a55; }
.nx-chat-empty-icon { font-size: 3rem; margin-bottom: 16px; }
.nx-chat-empty-text { font-size: 13px; line-height: 1.7; color: #2a3a55; max-width: 400px; margin: 0 auto; }

/* Sidebar brand */
.nx-sidebar-brand { padding: 18px 0 14px; border-bottom: 1px solid #131f35; margin-bottom: 18px; }
.nx-sidebar-logo  { height: 22px; margin-bottom: 6px; }
.nx-sidebar-tagline { font-size: 11px !important; letter-spacing: 2px !important; text-transform: uppercase !important; color: #2a3a55 !important; }
.nx-step-label { font-size: 12px !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 1px !important; color: #2a3a55 !important; margin-bottom: 3px; }
.nx-step-desc  { font-size: 12px !important; color: #2a3a55 !important; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)


# Inject NVIDIA favicon into browser tab
_favicon_svg = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>
  <rect width='100' height='100' rx='12' fill='#76b900'/>
  <text x='50' y='72' font-family='Arial Black,Arial,sans-serif' font-weight='900' 
    font-size='60' fill='white' text-anchor='middle'>N</text>
</svg>"""
import base64 as _fav_b64
_fav_b64_str = _fav_b64.b64encode(_favicon_svg.encode()).decode()
st.markdown(
    f'''<link rel="shortcut icon" href="data:image/svg+xml;base64,{_fav_b64_str}">''',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "report"         not in st.session_state: st.session_state.report         = None
if "last_analysis"  not in st.session_state: st.session_state.last_analysis  = None
if "chat_history"   not in st.session_state: st.session_state.chat_history   = []
if "chat_input_key" not in st.session_state: st.session_state.chat_input_key = 0

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
COMPETITOR_KEYWORDS = ["amd", "intel", "qualcomm", "google tpu", "cerebras",
                       "huawei", "custom silicon", "competitor", "alternative chip",
                       "arm", "apple silicon", "aws trainium", "microsoft maia"]

def _is_competitor(title: str, source: str) -> bool:
    t = (title or "").lower()
    return any(k in t for k in COMPETITOR_KEYWORDS)

def _sev_badge(level: str) -> str:
    lvl = (level or "").strip().lower()
    if lvl == "high":   return f'<span class="nx-badge nx-badge-red">● {level}</span>'
    if lvl == "medium": return f'<span class="nx-badge nx-badge-amber">● {level}</span>'
    return f'<span class="nx-badge nx-badge-green">● {level}</span>'

def _hor_badge(h: str) -> str:
    if "short" in (h or "").lower(): return '<span class="nx-badge nx-badge-green">Short-term</span>'
    if "long"  in (h or "").lower(): return '<span class="nx-badge nx-badge-blue">Long-term</span>'
    return '<span class="nx-badge nx-badge-amber">Mid-term</span>'

def _conf_bar(score: float) -> str:
    pct = int((score or 0) * 100)
    col = "#76b900" if pct >= 75 else "#f59e0b" if pct >= 50 else "#ef4444"
    return (
        f'<div class="nx-conf">'
        f'<div class="nx-conf-label">Confidence — {pct}%</div>'
        f'<div class="nx-conf-track">'
        f'<div class="nx-conf-fill" style="width:{pct}%;background:{col}"></div>'
        f'</div></div>'
    )

def _evidence_block(evidence: list) -> str:
    """Render a prominent evidence block with excerpt, source, and sentiment."""
    if not evidence:
        return ""
    items_html = ""
    for ev in evidence[:3]:
        if isinstance(ev, dict):
            src     = ev.get("source", "Unknown source")
            url     = ev.get("url", "#")
            excerpt = (ev.get("excerpt") or "")[:180]
            sent    = ev.get("sentiment", 0) or 0
            if sent >= 0.05:   sent_label, sent_col = "Positive", "#76b900"
            elif sent <= -0.05: sent_label, sent_col = "Negative", "#ef4444"
            else:               sent_label, sent_col = "Neutral",  "#f59e0b"
            dot_col = sent_col
            items_html += f"""
            <div class="nx-evidence-item">
              <div class="nx-evidence-dot" style="background:{dot_col};"></div>
              <div style="flex:1;min-width:0;">
                <div class="nx-evidence-source">
                  <a href="{url}" target="_blank">↗ {src}</a>
                </div>
                <div class="nx-evidence-excerpt">{excerpt}{"…" if len(ev.get("excerpt","")) > 180 else ""}</div>
                <div class="nx-evidence-sent" style="color:{sent_col};">
                  {sent_label} · {sent:.2f}
                </div>
              </div>
            </div>"""
    return f'<div class="nx-evidence-block"><div class="nx-evidence-title">📎 Supporting Evidence</div>{items_html}</div>'

def _sent_dot(score) -> str:
    if score is None: return "#2a3a55"
    if score >= 0.05:  return "#76b900"
    if score <= -0.05: return "#ef4444"
    return "#f59e0b"

def _sent_label(score: float) -> str:
    if score >= 0.05:  return "positive"
    if score <= -0.05: return "negative"
    return "neutral"

def get_articles_df(limit=300):
    try:
        conn = sqlite3.connect(config.SQLITE_DB_PATH, timeout=10)
        df   = pd.read_sql_query(
            f"SELECT * FROM articles ORDER BY collected DESC LIMIT {limit}", conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def get_last_update() -> str:
    try:
        conn = sqlite3.connect(config.SQLITE_DB_PATH, timeout=10)
        row  = conn.execute(
            "SELECT MAX(collected) FROM articles"
        ).fetchone()
        conn.close()
        if row and row[0]:
            dt = row[0][:19].replace("T", " ")
            return f"{dt} UTC"
    except Exception:
        pass
    return "—"

# ─────────────────────────────────────────────────────────────────────────────
# CHAT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
CEO_CHAT_SYSTEM = f"""You are an elite AI Strategic Advisor to the CEO of {config.COMPANY_NAME}.
RULES:
- Answer concisely but with executive depth — 3-5 paragraphs max
- Always ground claims in the provided intelligence sources
- Reference sources as [1], [2], [3] when citing
- If evidence is limited, say so honestly
- Use strategic CEO-level language — no filler phrases
- CRITICAL: Respond in plain text only — no markdown headers, no ### symbols
"""

def _build_prompt(question, context, history):
    hist = ""
    if history:
        for t in history[-4:]:
            role = "CEO" if t["role"] == "user" else "Advisor"
            hist += f"{role}: {t['content'][:300]}\n"
    return (
        f"CONVERSATION HISTORY:\n{hist or 'No prior conversation.'}\n\n"
        f"LIVE INTELLIGENCE:\n{context}\n\n"
        f"CEO QUESTION:\n{question}\n\n"
        "Be direct, specific, and actionable. Cite sources as [1],[2],[3]."
    )

def get_chat_response(question, history):
    docs = hybrid_search(question, k=8)
    if not docs:
        return "No intelligence data available yet. Run Steps 1 and 2 in the sidebar first.", []
    parts = [
        f"[{i}] {d['source']} — {d['title']}\n"
        f"Sentiment: {_sent_label(d.get('sentiment', 0))} ({d.get('sentiment', 0):.2f})\n"
        f"{d['text'][:600]}\nURL: {d.get('url','#')}"
        for i, d in enumerate(docs, 1)
    ]
    prompt = _build_prompt(question, "\n\n---\n\n".join(parts), history)
    try:
        answer = _call_llm_safe(prompt, CEO_CHAT_SYSTEM)
    except Exception as e:
        answer = f"An error occurred: {e}"
    return answer, docs[:5]

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
total_docs  = get_article_count()
source_data = get_source_breakdown()
last_update = get_last_update()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="nx-sidebar-brand">
      {f'<img src="{NVIDIA_LOGO_B64}" style="width:110px;height:auto;margin-bottom:6px;" alt="NVIDIA">' if NVIDIA_LOGO_B64 else '<div style="font-family:Arial Black,Arial,sans-serif;font-weight:900;font-size:26px;letter-spacing:4px;color:#76b900;">NVIDIA</div>'}
      <div class="nx-sidebar-tagline">CEO Strategic Intelligence</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-bottom:18px;">
      <div style="font-size:10px;color:#2a3a55;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">Knowledge Base</div>
      <div style="font-size:1.35rem;font-weight:800;color:#e2eaf8;">{total_docs:,} <span style="font-size:12px;color:#445066;font-weight:400;">documents</span></div>
      <div style="font-size:12px;color:#6b7a99;margin-top:6px;line-height:1.8;">&#128240; News &nbsp;&middot;&nbsp; &#127970; Company &nbsp;&middot;&nbsp; &#128300; Research<br>&#128172; Community (Reddit RSS) &nbsp;&middot;&nbsp; &#127981; Industry</div>
      <div style="font-size:10px;color:#2a3a55;margin-top:4px;">Updated: {last_update}</div>
    </div>
    <div style="height:1px;background:#131f35;margin-bottom:18px;"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nx-step-label">01 — Collect Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-step-desc">RSS feeds · Wikipedia · NewsAPI</div>', unsafe_allow_html=True)
    if st.button("Run Data Collection", use_container_width=True):
        with st.spinner("Collecting from sources…"):
            try:
                r = collector.run_collection()
                st.success(f"✓ {r['new']} new  ·  {r['total']} total")
            except Exception as e:
                st.error(f"{e}")

    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-step-label">02 — Process & Embed</div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-step-desc">VADER sentiment + MiniLM → ChromaDB</div>', unsafe_allow_html=True)
    if st.button("Process & Embed", use_container_width=True):
        with st.spinner("Embedding documents…"):
            try:
                r = processor.run_processing()
                st.success(f"✓ {r['new_chunks']} chunks  ·  {r['total_vectors']} vectors")
            except Exception as e:
                st.error(f"{e}")

    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-step-label">03 — AI Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="nx-step-desc">RAG + LLM reasoning  (~2 min)</div>', unsafe_allow_html=True)
    if st.button("Run Intelligence Engine", use_container_width=True, type="primary"):
        with st.spinner("AI engine analysing…"):
            try:
                rpt = run_intelligence_engine()
                st.session_state.report        = report_to_dict(rpt)
                st.session_state.last_analysis = datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M UTC")
                if rpt.error:
                    st.warning(f"Partial — {rpt.error}")
                else:
                    st.success("Analysis complete")
            except Exception as e:
                st.error(f"{e}")

    st.markdown('<div style="height:1px;background:#131f35;margin:18px 0;"></div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history   = []
            st.session_state.chat_input_key += 1
            st.rerun()

    bk_map = {
        "groq":        f"Groq · {config.GROQ_MODEL}",
        "ollama":      f"Ollama · {config.OLLAMA_MODEL}",
        "huggingface": f"HuggingFace · {config.HF_MODEL}",
    }
    st.markdown(f"""
    <div style="font-size:9px;color:#2a3a55;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:4px;">LLM Backend</div>
    <div style="font-size:11px;color:#445066;font-weight:500;">{bk_map.get(config.LLM_BACKEND, config.LLM_BACKEND)}</div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER — NVIDIA logo + title
# ─────────────────────────────────────────────────────────────────────────────
# Hero: logo LEFT + title RIGHT side by side
_logo_html = (
    f'<img src="{NVIDIA_LOGO_B64}" style="width:180px;height:auto;'
    'filter:drop-shadow(0 0 20px rgba(118,185,0,0.18));" alt="NVIDIA">'
    if NVIDIA_LOGO_B64 else
    '<div style="font-family:Arial Black,Arial,sans-serif;font-weight:900;'
    'font-size:32px;letter-spacing:5px;color:#76b900;">NVIDIA</div>'
)
_ts_html = (
    "Last analysis: " + st.session_state.last_analysis
    if st.session_state.last_analysis
    else "Run the Intelligence Engine (Step 3) to generate your strategic briefing"
)
st.markdown(f"""
<div class="nx-hero">
  <div class="nx-hero-logo-wrap">
    {_logo_html}
  </div>
  <div class="nx-hero-text">
    <h1 class="nx-hero-title">CEO Strategic Intelligence Agent</h1>
    <div class="nx-hero-subtitle">
      Real-time market analysis &nbsp;·&nbsp; AI-powered risk &amp; opportunity detection &nbsp;·&nbsp; CEO advisory
    </div>
    <div class="nx-hero-ts">{_ts_html}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_overview, tab_intel, tab_chat = st.tabs([
    "  Overview  ",
    "  Intelligence  ",
    "  CEO Advisor  ",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW  (Sections 1 + 2)
# ═════════════════════════════════════════════════════════════════════════════
with tab_overview:

    # ── Section 1: Company Overview ───────────────────────────────────────────
    st.markdown('<div class="nx-section-header"><div class="nx-section-icon">🏢</div><h3 class="nx-section-title">Section 1 — Company Overview</h3></div>', unsafe_allow_html=True)

    kpi_vectors = 0
    try:
        from nvidia_agent.storage.vector_store import get_collection
        kpi_vectors = get_collection().count()
    except Exception:
        pass

    # 4 equal-width metric cards — Analysis Status uses same st.metric as others
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Company", config.COMPANY_NAME)
    with c2:
        st.metric("Industry", config.COMPANY_SECTOR)
    with c3:
        st.metric("Documents Collected", f"{total_docs:,}")
    with c4:
        st.metric("Data Sources", "5 Types · 30+ Feeds", help="Company, News, Research, Industry, Community")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("Vector Chunks", f"{kpi_vectors:,}")
    with c6:
        st.metric("Headquarters", config.COMPANY_HQ)
    with c7:
        st.metric("Last Updated", last_update)
    with c8:
        status = "✓ Ready" if st.session_state.report else "Pending"
        st.metric("Analysis Status", status)

    st.markdown("---")

    # Source bar chart
    if source_data:
        df_src = pd.DataFrame(source_data).sort_values("count", ascending=False)
        TOP_N  = 10
        if len(df_src) > TOP_N:
            top    = df_src.head(TOP_N).copy()
            others = pd.DataFrame([{"source": f"Others ({len(df_src)-TOP_N})", "count": int(df_src.tail(len(df_src)-TOP_N)["count"].sum())}])
            df_src = pd.concat([top, others], ignore_index=True)
        df_src = df_src.sort_values("count", ascending=True)
        bar_c  = ["#1e2d4a" if "Others" in str(s) else "#76b900" for s in df_src["source"]]
        fig    = go.Figure(go.Bar(
            x=df_src["count"], y=df_src["source"], orientation="h",
            marker_color=bar_c, marker_line=dict(color="rgba(0,0,0,0)"),
            text=df_src["count"], textposition="outside",
            textfont=dict(color="#445066", size=11, family="Inter"),
            hovertemplate="%{y}<br><b>%{x} articles</b><extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="Knowledge Base — Articles by Source", font=dict(color="#8a9ab8", size=14, family="Inter"), x=0),
            height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#445066", family="Inter"),
            margin=dict(l=0, r=60, t=36, b=0),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#7a8ba8")),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Section 2: Market Intelligence ───────────────────────────────────────
    st.markdown('<div class="nx-section-header"><div class="nx-section-icon">📰</div><h3 class="nx-section-title">Section 2 — Market Intelligence</h3></div>', unsafe_allow_html=True)

    df_all = get_articles_df(300)

    def _feed(df_f, n=25):
        if df_f.empty:
            st.markdown('<div style="padding:28px;text-align:center;color:#2a3a55;font-size:13px;">No articles in this category yet. Run Step 1 — Data Collection.</div>', unsafe_allow_html=True)
            return
        html = ""
        for _, row in df_f.head(n).iterrows():
            score  = row.get("sentiment_score") or 0.0
            dot_c  = _sent_dot(score)
            title  = (row.get("title") or "No title")[:100]
            url    = row.get("url") or "#"
            source = row.get("source_name") or ""
            pub    = (row.get("published") or "")[:16]
            html  += f"""
            <div class="nx-news-item">
              <div class="nx-news-dot" style="background:{dot_c};"></div>
              <div>
                <div class="nx-news-title"><a href="{url}" target="_blank">{title}</a></div>
                <div class="nx-news-meta">{source}{' · ' + pub if pub else ''}</div>
              </div>
            </div>"""
        st.markdown(html, unsafe_allow_html=True)

    if not df_all.empty:
        # Fix: Company tab shows "company" AND Wikipedia research articles about NVIDIA
        # Also include research/industry in their own tab
        df_company    = df_all[df_all["source_type"] == "company"]
        df_news       = df_all[df_all["source_type"] == "news"]
        df_community  = df_all[df_all["source_type"] == "community"]
        df_research   = df_all[df_all["source_type"].isin(["research", "industry"])]

        # Competitor articles — any source_type containing competitor keywords in title
        df_competitor = df_all[df_all["title"].apply(lambda t: _is_competitor(str(t), ""))]

        # If company tab is still empty, include Wikipedia NVIDIA articles as fallback
        if df_company.empty:
            df_company = df_all[
                df_all["source_name"].str.contains("Wikipedia", case=False, na=False) |
                df_all["source_type"].str.contains("company", case=False, na=False)
            ]

        t_all, t_co, t_news, t_comp, t_comm, t_res = st.tabs([
            "All", "Company", "News", "Competitors", "Community", "Research"
        ])
        with t_all:  _feed(df_all)
        with t_co:   _feed(df_company)
        with t_news: _feed(df_news)
        with t_comp:
            if df_competitor.empty:
                st.info("No competitor-specific articles detected yet. More articles may appear after collection.")
            else:
                _feed(df_competitor)
        with t_comm:
            if df_community.empty:
                reddit_info = (
                    '<div style="background:#071020;border:1px solid #192a45;'
                    'border-left:3px solid #76b900;border-radius:10px;padding:16px 20px;margin:12px 0;">'
                    '<div style="font-size:14px;font-weight:700;color:#76b900;margin-bottom:10px;">'
                    'ℹ️ Reddit — Collected via RSS Feeds</div>'
                    '<div style="font-size:14px;color:#8a9ab8;line-height:1.8;">'
                    "Reddit's direct API requires OAuth (policy change 2023). "
                    'Reddit IS collected via public RSS feeds:<br>'
                    '<span style="color:#e2eaf8;">r/nvidia</span> · '
                    '<span style="color:#e2eaf8;">r/MachineLearning</span> · '
                    '<span style="color:#e2eaf8;">r/hardware</span> · '
                    '<span style="color:#e2eaf8;">r/artificial</span><br>'
                    '<span style="font-size:13px;color:#6b7a99;margin-top:8px;display:block;">'
                    'Run Step 1 to fetch the latest community posts.</span>'
                    '</div></div>'
                )
                st.markdown(reddit_info, unsafe_allow_html=True)
            else:
                _feed(df_community)
        with t_res:  _feed(df_research)
    else:
        st.markdown('<div style="padding:40px;text-align:center;color:#2a3a55;font-size:13px;">No articles yet. Click <strong>Run Data Collection</strong> in the sidebar.</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — INTELLIGENCE  (Sections 3, 4, 6, 7)
# ═════════════════════════════════════════════════════════════════════════════
with tab_intel:

    df_all = get_articles_df(300)  # needed for Section 5 sentiment
    report = st.session_state.report

    if report is None:
        st.markdown("""
        <div style="text-align:center;padding:80px 40px;color:#2a3a55;">
          <div style="font-size:2.5rem;margin-bottom:16px;">🧠</div>
          <div style="font-size:15px;font-weight:600;color:#445066;margin-bottom:8px;">No Intelligence Report Yet</div>
          <div style="font-size:13px;line-height:1.7;color:#2a3a55;">
            Run <strong style="color:#76b900;">Step 1 → Step 2 → Step 3</strong> in the sidebar<br>
            to generate your AI-powered strategic intelligence report.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        # ── AGENT PLAN (Step 1 output) ────────────────────────────────────────
        agent_plan = report.get("agent_plan", {})
        agent_val  = report.get("agent_validation", {})

        if agent_plan:
            st.markdown(
                '<div class="nx-section-header">'
                '<div class="nx-section-icon">🎯</div>'
                '<h3 class="nx-section-title">Agent Plan — What the AI Decided to Investigate</h3>'
                '</div>',
                unsafe_allow_html=True
            )
            goal      = agent_plan.get("goal", "")
            reasoning = agent_plan.get("reasoning", "")
            key_qs    = agent_plan.get("key_questions", [])
            opp_qs    = agent_plan.get("opportunity_queries", [])
            risk_qs   = agent_plan.get("risk_queries", [])

            pc1, pc2, pc3 = st.columns([2, 1, 1])

            with pc1:
                st.markdown(
                    '<div style="background:#071020;border:1px solid #192a45;'
                    'border-left:3px solid #76b900;border-radius:10px;padding:16px 18px;margin-bottom:12px;">'
                    '<div style="font-size:11px;font-weight:700;color:#76b900;text-transform:uppercase;'
                    'letter-spacing:1px;margin-bottom:8px;">Agent Goal</div>'
                    f'<div style="font-size:15px;color:#e2eaf8;line-height:1.6;">{goal}</div>'
                    f'<div style="font-size:13px;color:#6b7a99;margin-top:10px;line-height:1.5;">{reasoning}</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                if key_qs:
                    qs_items = "".join(
                        f'<div style="font-size:13px;color:#8a9ab8;padding:5px 0;'
                        f'border-bottom:1px solid #0e1c34;">→ {q}</div>'
                        for q in key_qs
                    )
                    st.markdown(
                        '<div style="background:#071020;border:1px solid #192a45;border-radius:10px;padding:14px 16px;">'
                        '<div style="font-size:11px;font-weight:700;color:#445066;text-transform:uppercase;margin-bottom:8px;">Key Questions</div>'
                        f'{qs_items}</div>',
                        unsafe_allow_html=True
                    )

            with pc2:
                if opp_qs:
                    items = "".join(
                        f'<div style="font-size:12px;color:#76b900;padding:4px 0;line-height:1.4;">✓ {q[:60]}</div>'
                        for q in opp_qs
                    )
                    st.markdown(
                        '<div style="background:#071020;border:1px solid #192a45;border-radius:10px;padding:14px 16px;">'
                        '<div style="font-size:11px;font-weight:700;color:#445066;text-transform:uppercase;margin-bottom:8px;">Opportunity Queries</div>'
                        f'{items}</div>',
                        unsafe_allow_html=True
                    )

            with pc3:
                if risk_qs:
                    items = "".join(
                        f'<div style="font-size:12px;color:#ef4444;padding:4px 0;line-height:1.4;">✓ {q[:60]}</div>'
                        for q in risk_qs
                    )
                    st.markdown(
                        '<div style="background:#071020;border:1px solid #192a45;border-radius:10px;padding:14px 16px;">'
                        '<div style="font-size:11px;font-weight:700;color:#445066;text-transform:uppercase;margin-bottom:8px;">Risk Queries</div>'
                        f'{items}</div>',
                        unsafe_allow_html=True
                    )

            st.markdown("---")

        # ── AGENT VALIDATION (Step 5 output) ─────────────────────────────────
        if agent_val:
            conf     = agent_val.get("overall_confidence", 0)
            conf_pct = int(conf * 100)
            conf_col = "#76b900" if conf_pct >= 75 else "#f59e0b" if conf_pct >= 50 else "#ef4444"
            val_notes        = agent_val.get("validation_notes", "")
            unaddressed_risks = agent_val.get("unaddressed_risks", [])
            unaddressed_opps  = agent_val.get("unaddressed_opportunities", [])
            contradictions    = agent_val.get("contradictions", [])

            st.markdown(
                '<div class="nx-section-header">'
                '<div class="nx-section-icon">✅</div>'
                '<h3 class="nx-section-title">Agent Validation — Self-Assessment Before Presenting</h3>'
                '</div>',
                unsafe_allow_html=True
            )

            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Analysis Confidence",       f"{conf_pct}%")
            v2.metric("Unaddressed Risks",          str(len(unaddressed_risks)))
            v3.metric("Unaddressed Opportunities",  str(len(unaddressed_opps)))
            v4.metric("Contradictions Found",       str(len(contradictions)))

            if val_notes:
                st.markdown(
                    f'<div style="background:#071020;border:1px solid #192a45;'
                    f'border-left:3px solid {conf_col};border-radius:10px;'
                    f'padding:14px 18px;margin:12px 0;font-size:14px;color:#8a9ab8;line-height:1.6;">'
                    f'{val_notes}</div>',
                    unsafe_allow_html=True
                )

            rec_scores = agent_val.get("recommendation_scores", [])
            if rec_scores:
                rows_html = ""
                for rs in rec_scores:
                    sc     = rs.get("score", 0)
                    sc_col = "#76b900" if sc >= 0.75 else "#f59e0b" if sc >= 0.5 else "#ef4444"
                    reason = rs.get("reason", "")
                    text   = rs.get("text", "")
                    rows_html += (
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'padding:8px 0;border-bottom:1px solid #0e1c34;">'
                        f'<div style="flex:1;font-size:13px;color:#b8c4d8;">{text}</div>'
                        f'<div style="font-size:12px;color:#6b7a99;flex:1;padding:0 12px;">{reason}</div>'
                        f'<div style="font-size:14px;font-weight:700;color:{sc_col};white-space:nowrap;">{int(sc*100)}%</div>'
                        f'</div>'
                    )
                st.markdown(
                    '<div style="background:#071020;border:1px solid #192a45;border-radius:10px;padding:14px 18px;margin:12px 0;">'
                    '<div style="font-size:11px;font-weight:700;color:#445066;text-transform:uppercase;margin-bottom:10px;">Recommendation Confidence Scores</div>'
                    f'{rows_html}</div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")

        # ── Sections 3 + 4: Opportunities + Risks ────────────────────────────
        col_opp, col_risk = st.columns(2)

        with col_opp:
            st.markdown('<div class="nx-section-header"><div class="nx-section-icon">🚀</div><h3 class="nx-section-title">Section 3 — Opportunity Monitor</h3></div>', unsafe_allow_html=True)
            opps = report.get("opportunities", [])
            if not opps:
                st.info("No opportunities identified. Re-run Step 3.")
            else:
                for opp in opps:
                    impact  = opp.get("impact_level", "Medium")
                    i_lower = impact.lower()
                    accent  = "green" if i_lower == "high" else "amber" if i_lower == "medium" else "blue"
                    ev_html = _evidence_block(opp.get("evidence", []))
                    st.markdown(f"""
                    <div class="nx-card nx-card-{accent}">
                      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:8px;">
                        <div class="nx-card-title">{opp.get("title","")}</div>
                        {_sev_badge(impact)}
                      </div>
                      <div class="nx-card-body">{opp.get("description","")}</div>
                      {_conf_bar(opp.get("confidence_score", 0.7))}
                      <div class="nx-card-footer">
                        <div style="font-size:11px;color:#2a3a55;margin-bottom:4px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;">Expected Impact</div>
                        <div style="font-size:14px;color:#5a6a88;">{opp.get("expected_impact","")}</div>
                      </div>
                      {ev_html}
                    </div>""", unsafe_allow_html=True)

        with col_risk:
            st.markdown('<div class="nx-section-header"><div class="nx-section-icon">⚠️</div><h3 class="nx-section-title">Section 4 — Risk Monitor</h3></div>', unsafe_allow_html=True)
            risks = report.get("risks", [])
            if not risks:
                st.info("No risks identified. Re-run Step 3.")
            else:
                for risk in risks:
                    sev     = risk.get("severity", "Medium")
                    s_lower = sev.lower()
                    accent  = "red" if s_lower == "high" else "amber" if s_lower == "medium" else "green"
                    cat     = risk.get("category", "")
                    ev_html = _evidence_block(risk.get("evidence", []))
                    st.markdown(f"""
                    <div class="nx-card nx-card-{accent}">
                      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:8px;">
                        <div class="nx-card-title">{risk.get("title","")}</div>
                        <div style="display:flex;gap:5px;flex-wrap:wrap;">
                          {_sev_badge(sev)}
                          <span class="nx-badge nx-badge-blue" style="font-size:9px;">{cat}</span>
                        </div>
                      </div>
                      <div class="nx-card-body">{risk.get("description","")}</div>
                      {_conf_bar(risk.get("confidence_score", 0.7))}
                      <div class="nx-card-footer">
                        <div style="font-size:11px;color:#2a3a55;margin-bottom:4px;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;">Mitigation</div>
                        <div style="font-size:14px;color:#5a6a88;">{risk.get("mitigation","")}</div>
                      </div>
                      {ev_html}
                    </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Section 5: Sentiment Analysis (moved here for correct sequence) ──────
        st.markdown('---')
        st.markdown('<div class="nx-section-header"><div class="nx-section-icon">💡</div><h3 class="nx-section-title">Section 5 — Sentiment Analysis</h3></div>', unsafe_allow_html=True)


        df_s = df_all.dropna(subset=["sentiment_score"]).copy()
        df_s["label"] = df_s["sentiment_score"].apply(
            lambda s: "Positive" if (s or 0) >= 0.05 else ("Negative" if (s or 0) <= -0.05 else "Neutral")
        )
        avg = df_s["sentiment_score"].mean()
        tot = len(df_s)
        pos = int((df_s["label"] == "Positive").sum())
        neg = int((df_s["label"] == "Negative").sum())

        ov_col  = "#76b900" if avg >= 0.05 else "#ef4444" if avg <= -0.05 else "#f59e0b"
        ov_text = "POSITIVE"  if avg >= 0.05 else "NEGATIVE" if avg <= -0.05 else "NEUTRAL"

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Overall Sentiment", ov_text)
        k2.metric("Avg VADER Score",   f"{avg:.3f}")
        k3.metric("Positive Articles", f"{pos}  ({int(pos/tot*100)}%)" if tot else "0")
        k4.metric("Negative Articles", f"{neg}  ({int(neg/tot*100)}%)" if tot else "0")

        st.markdown("---")

        # Gauge + Donut
        g1, g2 = st.columns(2)
        with g1:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=round(avg, 3),
                number={"valueformat": ".3f", "font": {"color": ov_col, "size": 38, "family": "Inter"}},
                title={"text": f"Market Sentiment Score<br><span style='font-size:11px;color:{ov_col}'>{ov_text}</span>",
                       "font": {"color": "#445066", "size": 12}},
                gauge={
                    "axis": {"range": [-1, 1], "tickvals": [-1, -0.5, 0, 0.5, 1],
                             "tickfont": {"color": "#2a3a55", "size": 10}, "tickcolor": "#131f35"},
                    "bar": {"color": ov_col, "thickness": 0.24},
                    "bgcolor": "#0b1628", "borderwidth": 0,
                    "steps": [
                        {"range": [-1, -0.05], "color": "#160909"},
                        {"range": [-0.05, 0.05], "color": "#0e1c34"},
                        {"range": [0.05, 1],  "color": "#081508"},
                    ],
                    "threshold": {"line": {"color": ov_col, "width": 2}, "thickness": 0.78, "value": avg},
                },
            ))
            fig_g.update_layout(height=290, paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Inter"), margin=dict(t=80, b=20, l=40, r=40))
            st.plotly_chart(fig_g, use_container_width=True)

        with g2:
            counts = df_s["label"].value_counts().reset_index()
            counts.columns = ["label", "count"]
            cmap   = {"Positive": "#76b900", "Negative": "#ef4444", "Neutral": "#f59e0b"}
            fig_p  = go.Figure(go.Pie(
                labels=counts["label"], values=counts["count"],
                marker_colors=[cmap.get(l, "#2a3a55") for l in counts["label"]],
                hole=0.62, textinfo="percent", textposition="inside",
                insidetextorientation="horizontal",
                textfont=dict(size=12, color="#fff", family="Inter"),
                hovertemplate="%{label}: %{value} articles (%{percent})<extra></extra>",
            ))
            fig_p.update_layout(
                title=dict(text="Sentiment Distribution", font=dict(color="#8a9ab8", size=14, family="Inter"), x=0),
                height=290, paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#445066"),
                margin=dict(t=36, b=10, l=10, r=10),
                legend=dict(orientation="v", x=1, y=0.5, font=dict(size=14, color="#8a9ab8")),
            )
            st.plotly_chart(fig_p, use_container_width=True)

        st.markdown("---")

        # ── News sentiment vs Public (community) sentiment ─────────────────────
        st.markdown('<div style="font-size:15px;font-weight:700;color:#b8c4d8;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid #192a45;">News Sentiment vs Public Sentiment</div>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)

        def _mini_sent_kpi(df_sub, label):
            if df_sub.empty:
                st.markdown(f'<div style="padding:20px;color:#2a3a55;font-size:13px;">No {label} articles yet.</div>', unsafe_allow_html=True)
                return
            avg_sub = df_sub["sentiment_score"].mean()
            pos_sub = int((df_sub["label"] == "Positive").sum())
            neg_sub = int((df_sub["label"] == "Negative").sum())
            tot_sub = len(df_sub)
            col = "#76b900" if avg_sub >= 0.05 else "#ef4444" if avg_sub <= -0.05 else "#f59e0b"
            lbl = "POSITIVE" if avg_sub >= 0.05 else "NEGATIVE" if avg_sub <= -0.05 else "NEUTRAL"
            st.markdown(f"""
            <div class="nx-card" style="border-left:3px solid {col};">
              <div style="font-size:13px;font-weight:700;color:#8a9ab8;margin-bottom:8px;">{label}</div>
              <div style="font-size:1.5rem;font-weight:800;color:{col};">{lbl}</div>
              <div style="font-size:12px;color:#445066;margin-top:4px;">Avg score: {avg_sub:.3f} · {tot_sub} articles</div>
              <div style="font-size:12px;color:#445066;margin-top:2px;">
                🟢 {pos_sub} positive  &nbsp;  🔴 {neg_sub} negative
              </div>
            </div>""", unsafe_allow_html=True)

        with s1:
            df_news_s = df_s[df_s["source_type"] == "news"]
            _mini_sent_kpi(df_news_s, "News Sentiment")
        with s2:
            df_pub_s = df_s[df_s["source_type"] == "community"]
            _mini_sent_kpi(df_pub_s, "Public / Community Sentiment")

        st.markdown("---")

        # ── Sentiment by source — top 12 sources bar chart ──────────────────
        st.markdown('<div style="font-size:15px;font-weight:700;color:#b8c4d8;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid #192a45;">Sentiment Score by Source (Top 12 Most Active)</div>', unsafe_allow_html=True)
        src_sent = (
            df_s.groupby("source_name")
                .agg(avg_sent=("sentiment_score", "mean"), count=("sentiment_score", "count"))
                .reset_index()
                .sort_values("count", ascending=False)
                .head(12)
                .sort_values("avg_sent", ascending=True)
        )
        if not src_sent.empty:
            bar_colors = [
                "#ef4444" if v < -0.05 else "#f59e0b" if v < 0.05 else "#76b900"
                for v in src_sent["avg_sent"]
            ]
            fig_src_sent = go.Figure(go.Bar(
                x=src_sent["avg_sent"],
                y=src_sent["source_name"],
                orientation="h",
                marker_color=bar_colors,
                marker_line=dict(color="rgba(0,0,0,0)"),
                text=[f"{v:+.3f}" for v in src_sent["avg_sent"]],
                textposition="outside",
                textfont=dict(color="#8a9ab8", size=12, family="Inter"),
                customdata=src_sent["count"],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Avg Sentiment: %{x:.3f}<br>"
                    "Articles: %{customdata}<extra></extra>"
                ),
            ))
            fig_src_sent.add_vline(x=0,     line_color="#2a3a55", line_dash="solid", line_width=1)
            fig_src_sent.add_vline(x=0.05,  line_color="rgba(118,185,0,0.35)",  line_dash="dot", line_width=1)
            fig_src_sent.add_vline(x=-0.05, line_color="rgba(239,68,68,0.35)",  line_dash="dot", line_width=1)
            # Add annotation labels for the zones
            fig_src_sent.add_annotation(x=0.5,  y=0, text="POSITIVE ZONE", showarrow=False,
                font=dict(size=9, color="rgba(118,185,0,0.4)", family="Inter"),
                xanchor="center", yanchor="bottom", yref="paper")
            fig_src_sent.add_annotation(x=-0.5, y=0, text="NEGATIVE ZONE", showarrow=False,
                font=dict(size=9, color="rgba(239,68,68,0.4)", family="Inter"),
                xanchor="center", yanchor="bottom", yref="paper")
            fig_src_sent.update_layout(
                height=420,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#445066"),
                margin=dict(t=10, b=10, l=10, r=80),
                xaxis=dict(
                    showgrid=False, zeroline=False,
                    range=[-1, 1],
                    tickfont=dict(size=13, color="#8a9ab8"),
                    tickvals=[-1, -0.5, -0.05, 0, 0.05, 0.5, 1],
                    ticktext=["-1.0", "-0.5", "-0.05", "0", "+0.05", "+0.5", "+1.0"],
                ),
                yaxis=dict(
                    showgrid=False,
                    tickfont=dict(size=13, color="#b8c4d8"),
                ),
                bargap=0.3,
            )
            st.plotly_chart(fig_src_sent, use_container_width=True)
            st.markdown(
                '<div style="font-size:11px;color:#2a3a55;margin-top:-8px;">'
                '🟢 &gt; +0.05 Positive &nbsp;&nbsp; 🟡 -0.05 to +0.05 Neutral &nbsp;&nbsp; 🔴 &lt; -0.05 Negative'
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Avg sentiment by source type bar
        by_type = df_s.groupby("source_type")["sentiment_score"].mean().reset_index()
        by_type.columns = ["Type", "Avg"]
        by_type["Type"] = by_type["Type"].str.capitalize()
        by_type = by_type.sort_values("Avg", ascending=True)
        bar_c   = ["#ef4444" if v < -0.05 else "#f59e0b" if v < 0.05 else "#76b900" for v in by_type["Avg"]]
        fig_b   = go.Figure(go.Bar(
            x=by_type["Avg"], y=by_type["Type"], orientation="h",
            marker_color=bar_c, marker_line=dict(color="rgba(0,0,0,0)"),
            text=[f"{v:.3f}" for v in by_type["Avg"]], textposition="outside",
            textfont=dict(color="#445066", size=11, family="Inter"),
        ))
        fig_b.add_vline(x=0,     line_color="#131f35", line_dash="dash")
        fig_b.add_vline(x=0.05,  line_color="rgba(118,185,0,0.22)", line_dash="dot")
        fig_b.add_vline(x=-0.05, line_color="rgba(239,68,68,0.22)", line_dash="dot")
        fig_b.update_layout(
            title=dict(text="Average Sentiment Score by Source Type", font=dict(color="#8a9ab8", size=14, family="Inter"), x=0),
            height=220, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#445066"),
            margin=dict(t=36, b=10, l=10, r=80),
            xaxis=dict(showgrid=False, range=[-0.6, 0.8], zeroline=False, tickfont=dict(size=12, color="#6b7a99")),
            yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#7a8ba8")),
        )
        st.plotly_chart(fig_b, use_container_width=True)



        # ── Section 6: Strategic Recommendations ─────────────────────────────
        st.markdown('<div class="nx-section-header"><div class="nx-section-icon">🎯</div><h3 class="nx-section-title">Section 6 — Strategic Recommendations</h3></div>', unsafe_allow_html=True)
        # Trends — full width, 2-column grid
        st.markdown('<div style="font-size:15px;font-weight:700;color:#b8c4d8;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid #192a45;">Emerging Trends</div>', unsafe_allow_html=True)
        trends = report.get("trends", [])
        if not trends:
            st.info("Trends appear after Step 3.")
        else:
            tr_cols = st.columns(2)
            for idx, t in enumerate(trends):
                with tr_cols[idx % 2]:
                    st.markdown(f'''
                    <div class="nx-trend">
                      <div class="nx-trend-title">{t.get("title","")}</div>
                      <div class="nx-trend-body">{t.get("description","")}</div>
                      <div class="nx-trend-rel">↳ {t.get("relevance","")}</div>
                    </div>''', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div style="font-size:15px;font-weight:700;color:#b8c4d8;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid #192a45;">CEO Action Items</div>', unsafe_allow_html=True)
        recs = report.get("recommendations", [])
        if not recs:
            st.info("Recommendations appear after Step 3.")
        else:
            priority_order = {"High": 0, "Medium": 1, "Low": 2}
            for i, rec in enumerate(sorted(recs, key=lambda r: priority_order.get(r.get("priority","Low"), 2)), 1):
                pri      = rec.get("priority", "Medium")
                hor      = rec.get("time_horizon", "")
                ev_html  = _evidence_block(rec.get("evidence", []))
                risk_ass = rec.get("risk_assessment", {})
                fin_risk = risk_ass.get("financial",   "To be assessed")
                ops_risk = risk_ass.get("operational", "To be assessed")
                str_risk = risk_ass.get("strategic",   "To be assessed")
                # Determine overall risk level from text
                fin_lvl = "red"   if "high" in fin_risk.lower() else "amber" if "medium" in fin_risk.lower() else "green"
                ops_lvl = "red"   if "high" in ops_risk.lower() else "amber" if "medium" in ops_risk.lower() else "green"
                str_lvl = "red"   if "high" in str_risk.lower() else "amber" if "medium" in str_risk.lower() else "green"
                # Build all dynamic parts first — avoids f-string conflict with CSS {}
                rec_text_val   = rec.get("recommendation", "")
                rec_impact_val = rec.get("expected_impact", "")
                hor_badge_html = _hor_badge(hor)
                pri_badge_html = _sev_badge(pri)

                rec_html = (
                    '<div class="nx-rec">'
                    '<div class="nx-rec-num">Action #' + str(i) + ' &nbsp;&middot;&nbsp; ' + hor_badge_html + '</div>'
                    '<div class="nx-rec-text">' + rec_text_val + '</div>'

                    '<div style="margin:12px 0 6px 0;font-size:10px;font-weight:700;'
                    'letter-spacing:1.2px;text-transform:uppercase;color:#76b900;">'
                    '&#128200; Expected Impact</div>'
                    '<div class="nx-rec-impact">' + rec_impact_val + '</div>'

                    '<div style="margin:14px 0 8px 0;font-size:10px;font-weight:700;'
                    'letter-spacing:1.2px;text-transform:uppercase;color:#ef4444;">'
                    '&#9888; Risk Assessment</div>'

                    '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px;">'

                    '<div style="background:#071020;border:1px solid #1a2540;'
                    'border-top:2px solid #ef4444;border-radius:8px;padding:10px 12px;">'
                    '<div style="font-size:12px;font-weight:700;letter-spacing:0.5px;color:#ef4444;'
                    'text-transform:uppercase;margin-bottom:5px;">Financial Risk</div>'
                    '<div style="font-size:14px;color:#8a9ab8;line-height:1.6;">' + fin_risk + '</div>'
                    '</div>'

                    '<div style="background:#071020;border:1px solid #1a2540;'
                    'border-top:2px solid #f59e0b;border-radius:8px;padding:10px 12px;">'
                    '<div style="font-size:12px;font-weight:700;letter-spacing:0.5px;color:#f59e0b;'
                    'text-transform:uppercase;margin-bottom:5px;">Operational Risk</div>'
                    '<div style="font-size:14px;color:#8a9ab8;line-height:1.6;">' + ops_risk + '</div>'
                    '</div>'

                    '<div style="background:#071020;border:1px solid #1a2540;'
                    'border-top:2px solid #3b82f6;border-radius:8px;padding:10px 12px;">'
                    '<div style="font-size:12px;font-weight:700;letter-spacing:0.5px;color:#3b82f6;'
                    'text-transform:uppercase;margin-bottom:5px;">Strategic Risk</div>'
                    '<div style="font-size:14px;color:#8a9ab8;line-height:1.6;">' + str_risk + '</div>'
                    '</div>'

                    '</div>'
                    + ev_html +
                    '<div class="nx-rec-meta">' + pri_badge_html + '</div>'
                    '</div>'
                )
                st.markdown(rec_html, unsafe_allow_html=True)


        st.markdown('---')
        # ── Section 7: CEO Briefing ───────────────────────────────────────────
        briefing = report.get("ceo_briefing", {})
        if briefing:
            st.markdown('<div class="nx-section-header"><div class="nx-section-icon">📋</div><h3 class="nx-section-title">Section 7 — CEO Briefing</h3></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="nx-briefing">
              <div class="nx-briefing-label">Executive Summary</div>
              <div class="nx-briefing-text">{briefing.get("executive_summary","")}</div>
            </div>""", unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            with b1:
                raw = briefing.get("what_happened","")
                st.markdown(f'<div class="nx-briefing-col"><div class="nx-briefing-col-title">What Happened</div><div class="nx-briefing-col-body">{raw}</div></div>', unsafe_allow_html=True)
            with b2:
                raw = briefing.get("why_it_matters","")
                st.markdown(f'<div class="nx-briefing-col"><div class="nx-briefing-col-title">Why It Matters</div><div class="nx-briefing-col-body">{raw}</div></div>', unsafe_allow_html=True)
            with b3:
                raw = briefing.get("what_to_do_next","")
                if isinstance(raw, list):
                    raw = "".join(f"<div style='margin-bottom:6px;'>→ {item}</div>" for item in raw)
                st.markdown(f'<div class="nx-briefing-col"><div class="nx-briefing-col-title">What to Do Next</div><div class="nx-briefing-col-body">{raw}</div></div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — CEO ADVISOR
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:

    st.markdown(f"""
    <div class="nx-chat-header">
      <div class="nx-chat-icon">🧠</div>
      <div>
        <div class="nx-chat-title">AI Strategic Advisor</div>
        <div class="nx-chat-sub">
          Live RAG over {total_docs:,} documents &nbsp;·&nbsp; {config.LLM_BACKEND.upper()} &nbsp;·&nbsp; {config.GROQ_MODEL if config.LLM_BACKEND == "groq" else "local model"}
        </div>
      </div>
      <div class="nx-chat-live">
        <div class="nx-chat-dot"></div>LIVE
      </div>
    </div>""", unsafe_allow_html=True)

    # Suggested questions
    st.markdown('<div class="nx-sug-label">Quick Queries — click to ask</div>', unsafe_allow_html=True)
    suggested = [
        "What are NVIDIA's biggest strategic opportunities right now?",
        "What risks should the CEO be most concerned about?",
        "How is AMD positioned against NVIDIA in the AI chip market?",
        "What is the status of NVIDIA's China export situation?",
        "Which technology trends should management prioritise?",
        "How strong is NVIDIA's CUDA software moat?",
        "What should be our top 3 actions this quarter?",
        "What is the current market sentiment around NVIDIA?",
        "How is NVIDIA positioned in sovereign AI and government contracts?",
        "What are hyperscalers doing to reduce NVIDIA dependency?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(suggested):
        if cols[i % 2].button(q, key=f"sq_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.spinner("Searching knowledge base…"):
                ans, srcs = get_chat_response(q, st.session_state.chat_history[:-1])
            st.session_state.chat_history.append({"role": "assistant", "content": ans, "sources": srcs})
            st.rerun()

    st.markdown("---")

    # Chat history
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="nx-chat-empty">
          <div class="nx-chat-empty-icon">💬</div>
          <div class="nx-chat-empty-text">
            Ask any strategic question and your AI advisor will search the live
            knowledge base to deliver evidence-backed insights.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        for turn in st.session_state.chat_history:
            if turn["role"] == "user":
                st.markdown(f"""
                <div class="nx-msg-user">
                  <div class="nx-msg-user-lbl">CEO</div>
                  {turn["content"]}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="nx-msg-ai">
                  <div class="nx-msg-ai-lbl">Strategic Advisor</div>
                  {turn["content"]}
                </div>""", unsafe_allow_html=True)
                srcs = turn.get("sources", [])
                if srcs:
                    seen = set()
                    chips = ""
                    for s in srcs:
                        src = s.get("source","") if isinstance(s, dict) else ""
                        url = s.get("url","#") if isinstance(s, dict) else "#"
                        if src and src not in seen:
                            seen.add(src)
                            chips += f'<a href="{url}" target="_blank" class="nx-src-chip">↗ {src}</a>'
                    if chips:
                        st.markdown(f'<div style="margin:-2px 0 12px 0;">{chips}</div>', unsafe_allow_html=True)

    st.markdown("---")

    ci, cs, cc = st.columns([7, 1, 1])
    with ci:
        user_input = st.text_input(
            "q", placeholder="Ask your strategic advisor anything about NVIDIA…",
            label_visibility="collapsed",
            key=f"chat_input_{st.session_state.chat_input_key}",
        )
    with cs:
        send = st.button("Send", use_container_width=True, type="primary")
    with cc:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history   = []
            st.session_state.chat_input_key += 1
            st.rerun()

    if send and user_input and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        with st.spinner("Analysing…"):
            ans, srcs = get_chat_response(user_input.strip(), st.session_state.chat_history[:-1])
        st.session_state.chat_history.append({"role": "assistant", "content": ans, "sources": srcs})
        st.session_state.chat_input_key += 1
        st.rerun()

    st.markdown(f"""
    <div style="font-size:10px;color:#2a3a55;text-align:center;margin-top:10px;">
      Hybrid RAG (BM25 + cosine + RRF) over {total_docs:,} documents &nbsp;·&nbsp;
      LLM: {config.LLM_BACKEND.upper()} &nbsp;·&nbsp;
      Embedding: {config.EMBEDDING_MODEL}
    </div>""", unsafe_allow_html=True)