"""
pipeline.py — Command-line runner for all pipeline stages.

USAGE:
  python pipeline.py --collect          # Stage 1: collect live data
  python pipeline.py --process          # Stage 2: embed into ChromaDB
  python pipeline.py --analyse          # Stage 3: AI CEO analysis
  python pipeline.py --all              # All three stages
  python pipeline.py --all --report     # All stages + save data/report.json

Run this BEFORE launching the dashboard.
Then launch: streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="NVIDIA Strategic Intelligence Pipeline")
    parser.add_argument("--collect",  action="store_true", help="Run data collection")
    parser.add_argument("--process",  action="store_true", help="Run processing and embedding")
    parser.add_argument("--analyse",  action="store_true", help="Run AI CEO analysis")
    parser.add_argument("--all",      action="store_true", help="Run all stages")
    parser.add_argument("--report",   action="store_true", help="Save report.json")
    args = parser.parse_args()

    if not any([args.collect, args.process, args.analyse, args.all]):
        parser.print_help()
        return

    if args.all or args.collect:
        print("\n" + "="*60)
        print("STAGE 1 — Live Data Collection")
        print("="*60)
        from collector import run_collection
        result = run_collection()
        print(f"\n✅ Stage 1 complete | new={result['new']} | total={result['total']}")

        if result["total"] < 100:
            print(f"\n⚠️  Only {result['total']} articles collected.")
            print("   Run collection again to pick up more articles.")

    if args.all or args.process:
        print("\n" + "="*60)
        print("STAGE 2 — Processing & Embedding")
        print("(First run downloads embedding model ~80MB — be patient)")
        print("="*60)
        from processor import run_processing
        result = run_processing()
        print(f"\n✅ Stage 2 complete | chunks={result['new_chunks']} | vectors={result['total_vectors']}")

    if args.all or args.analyse:
        print("\n" + "="*60)
        print("STAGE 3 — AI CEO Intelligence Engine")
        print("Running RAG retrieval + LLM reasoning...")
        print("="*60)
        from nvidia_agent.agent.ceo_agent import run_intelligence_engine, report_to_dict
        report = run_intelligence_engine()
        d      = report_to_dict(report)

        # report is now a StrategicReport dataclass — .error works directly
        if report.error:
            print(f"\n⚠️  Partial results — {report.error}")

        print(f"\n📊 Opportunities : {len(d.get('opportunities', []))}")
        for o in d.get("opportunities", []):
            print(f"   [{o.get('impact_level','?')}] {o.get('title','')}")

        print(f"\n⚠️  Risks         : {len(d.get('risks', []))}")
        for r in d.get("risks", []):
            print(f"   [{r.get('severity','?')}] {r.get('title','')} — {r.get('category','')}")

        print(f"\n🔭 Trends         : {len(d.get('trends', []))}")
        for t in d.get("trends", []):
            print(f"   • {t.get('title','')}")

        print(f"\n🎯 Recommendations: {len(d.get('recommendations', []))}")
        for i, r in enumerate(d.get("recommendations", []), 1):
            rec_text = r.get('recommendation') or r.get('title','')
            print(f"   #{i} [{r.get('priority','?')}] {rec_text[:80]}")

        briefing = d.get("ceo_briefing", {})
        if briefing:
            print(f"\n📋 CEO Briefing:")
            print(f"   {str(briefing.get('executive_summary', ''))[:300]}")

        # Always save outputs — use absolute path based on script location
        script_dir = Path(__file__).resolve().parent
        outputs_dir = script_dir / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        print(f"\n📁 Saving outputs to: {outputs_dir}")

        # 1. Full data report
        (outputs_dir / "report.json").write_text(json.dumps(d, indent=2))
        print("💾 Report saved to outputs/report.json")

        # 2. Agent reasoning log — what the agent did, step by step
        agent_log = d.get("agent_log", {})
        if agent_log:
            log_lines = []
            log_lines.append("=" * 60)
            log_lines.append("NVIDIA STRATEGIC INTELLIGENCE AGENT — EXECUTION REPORT")
            log_lines.append("=" * 60)
            log_lines.append(f"Goal:        {agent_log.get('goal','')}")
            log_lines.append(f"Total steps: {agent_log.get('total_steps', 0)}")
            log_lines.append(f"Queries run: {len(agent_log.get('queries_run', []))}")
            log_lines.append("")

            # Counts
            counts = agent_log.get("final_counts", {})
            log_lines.append("FINAL COUNTS:")
            log_lines.append(f"  Opportunities:            {counts.get('opportunities', 0)}")
            log_lines.append(f"  Risks:                    {counts.get('risks', 0)}")
            log_lines.append(f"  Trends:                   {counts.get('trends', 0)}")
            log_lines.append(f"  Recommendations:          {counts.get('recommendations', 0)}")
            log_lines.append(f"  Recommendations approved: {counts.get('recommendations_approved', 0)}")
            log_lines.append("")

            # Queries
            log_lines.append("SEARCH QUERIES EXECUTED:")
            for i, q in enumerate(agent_log.get("queries_run", []), 1):
                log_lines.append(f"  {i}. {q}")
            log_lines.append("")

            # Step-by-step reasoning
            log_lines.append("STEP-BY-STEP REASONING LOG:")
            log_lines.append("-" * 60)
            for step in agent_log.get("reasoning_log", []):
                log_lines.append(f"Step {step.get('step','?'):>2} | Tool: {step.get('tool','?'):<12} | Query: {step.get('argument','')[:60]}")
                log_lines.append(f"         Reasoning: {step.get('reasoning','')[:100]}")
            log_lines.append("")

            # Opportunities
            log_lines.append("OPPORTUNITIES IDENTIFIED:")
            log_lines.append("-" * 60)
            for i, o in enumerate(d.get("opportunities", []), 1):
                log_lines.append(f"{i}. [{o.get('impact_level','?').upper()}] {o.get('title','')}")
                log_lines.append(f"   {o.get('description','')[:200]}")
                log_lines.append(f"   Confidence: {o.get('confidence_score', o.get('confidence', '?'))}")
                log_lines.append("")

            # Risks
            log_lines.append("RISKS IDENTIFIED:")
            log_lines.append("-" * 60)
            for i, r in enumerate(d.get("risks", []), 1):
                log_lines.append(f"{i}. [{r.get('severity','?').upper()}] {r.get('title','')}")
                log_lines.append(f"   Category: {r.get('category','')}")
                log_lines.append(f"   {r.get('description','')[:200]}")
                log_lines.append(f"   Mitigation: {r.get('mitigation','')[:150]}")
                log_lines.append("")

            # Trends
            log_lines.append("TRENDS IDENTIFIED:")
            log_lines.append("-" * 60)
            for i, t in enumerate(d.get("trends", []), 1):
                log_lines.append(f"{i}. {t.get('title','')}")
                log_lines.append(f"   {t.get('description','')[:200]}")
                log_lines.append(f"   Time horizon: {t.get('time_horizon','')}")
                log_lines.append("")

            # Recommendations
            log_lines.append("STRATEGIC RECOMMENDATIONS:")
            log_lines.append("-" * 60)
            for i, r in enumerate(d.get("recommendations", []), 1):
                rec_text = r.get('recommendation') or r.get('title','')
                validation = r.get('validation', {})
                log_lines.append(f"{i}. [{r.get('priority','?').upper()}] {rec_text[:120]}")
                log_lines.append(f"   Time horizon:    {r.get('time_horizon','')}")
                log_lines.append(f"   Expected impact: {r.get('expected_impact','')[:150]}")
                log_lines.append(f"   SBERT grounding: {r.get('grounding_confidence', 'N/A')}")
                log_lines.append(f"   Validation:      {validation.get('verdict','?')} — {validation.get('reason','')[:100]}")
                log_lines.append("")

            # CEO Briefing
            briefing = d.get("ceo_briefing", {})
            if briefing:
                log_lines.append("CEO BRIEFING:")
                log_lines.append("-" * 60)
                log_lines.append("EXECUTIVE SUMMARY:")
                log_lines.append(briefing.get("executive_summary",""))
                log_lines.append("")
                log_lines.append("WHAT HAPPENED:")
                log_lines.append(briefing.get("what_happened",""))
                log_lines.append("")
                log_lines.append("WHY IT MATTERS:")
                log_lines.append(briefing.get("why_it_matters",""))
                log_lines.append("")
                log_lines.append("WHAT TO DO NEXT:")
                log_lines.append(briefing.get("what_to_do_next",""))
                log_lines.append("")

            log_lines.append("=" * 60)
            log_lines.append("END OF AGENT REPORT")
            log_lines.append("=" * 60)

            report_text = "\n".join(log_lines)
            (outputs_dir / "agent_report.txt").write_text(report_text, encoding="utf-8")
            print(f"📋 Agent report saved to {outputs_dir / 'agent_report.txt'}")

            # ── Markdown report ───────────────────────────────────────────────
            from datetime import datetime
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            md = []
            md.append("# NVIDIA Strategic Intelligence Report")
            md.append(f"> Generated: {now}  ")
            md.append(f"> Agent steps: {agent_log.get('total_steps', 0)}  ")
            md.append(f"> Goal: {agent_log.get('goal', '')}")
            md.append("")

            # Agent execution summary
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

            # Counts summary
            counts = agent_log.get("final_counts", {})
            md.append("### Final Counts")
            md.append("")
            md.append(f"| Category | Count |")
            md.append(f"|----------|-------|")
            md.append(f"| Opportunities | {counts.get('opportunities', 0)} |")
            md.append(f"| Risks | {counts.get('risks', 0)} |")
            md.append(f"| Trends | {counts.get('trends', 0)} |")
            md.append(f"| Recommendations | {counts.get('recommendations', 0)} |")
            md.append(f"| Recommendations Approved | {counts.get('recommendations_approved', 0)} |")
            md.append("")

            # Opportunities
            md.append("## Opportunities")
            md.append("")
            for i, o in enumerate(d.get("opportunities", []), 1):
                level = o.get("impact_level", "?")
                md.append(f"### {i}. {o.get('title', '')}")
                md.append(f"**Impact Level:** {level}  ")
                md.append(f"**Confidence:** {o.get('confidence_score', o.get('confidence', '?'))}  ")
                md.append("")
                md.append(o.get("description", ""))
                md.append("")

            # Risks
            md.append("## Risks")
            md.append("")
            for i, r in enumerate(d.get("risks", []), 1):
                md.append(f"### {i}. {r.get('title', '')}")
                md.append(f"**Severity:** {r.get('severity', '?')}  ")
                md.append(f"**Category:** {r.get('category', '?')}  ")
                md.append("")
                md.append(r.get("description", ""))
                md.append("")
                md.append(f"**Mitigation:** {r.get('mitigation', '')}")
                md.append("")

            # Trends
            md.append("## Trends")
            md.append("")
            for i, t in enumerate(d.get("trends", []), 1):
                md.append(f"### {i}. {t.get('title', '')}")
                md.append(f"**Time Horizon:** {t.get('time_horizon', '?')}  ")
                md.append(f"**Expected Impact:** {t.get('expected_impact', '?')}  ")
                md.append("")
                md.append(t.get("description", ""))
                md.append("")

            # Recommendations
            md.append("## Strategic Recommendations")
            md.append("")
            for i, r in enumerate(d.get("recommendations", []), 1):
                rec_text   = r.get("recommendation") or r.get("title", "")
                validation = r.get("validation", {})
                verdict    = validation.get("verdict", "approved")
                grounding  = r.get("grounding_confidence", None)
                v_icon     = "✅" if verdict == "approved" else "⚠️" if verdict == "needs_revision" else "❌"
                md.append(f"### {i}. {rec_text}")
                md.append(f"**Priority:** {r.get('priority', '?')}  ")
                md.append(f"**Time Horizon:** {r.get('time_horizon', '?')}  ")
                md.append(f"**Expected Impact:** {r.get('expected_impact', '')}  ")
                md.append("")
                md.append(f"#### Validation")
                md.append(f"{v_icon} **{verdict.replace('_', ' ').upper()}**" +
                          (f"  — SBERT grounding: `{grounding:.2f}`" if grounding else ""))
                if validation.get("reason"):
                    md.append(f"> {validation.get('reason', '')}")
                md.append("")
                risk_ass = r.get("risk_assessment", {})
                if risk_ass:
                    md.append("#### Risk Assessment")
                    md.append(f"- **Financial:** {risk_ass.get('financial', '')}")
                    md.append(f"- **Operational:** {risk_ass.get('operational', '')}")
                    md.append(f"- **Strategic:** {risk_ass.get('strategic', '')}")
                md.append("")

            # CEO Briefing
            briefing = d.get("ceo_briefing", {})
            if briefing:
                md.append("## CEO Briefing")
                md.append("")
                md.append("### Executive Summary")
                md.append(briefing.get("executive_summary", ""))
                md.append("")
                md.append("### What Happened")
                md.append(briefing.get("what_happened", ""))
                md.append("")
                md.append("### Why It Matters")
                md.append(briefing.get("why_it_matters", ""))
                md.append("")
                md.append("### What to Do Next")
                md.append(briefing.get("what_to_do_next", ""))
                md.append("")

            md.append("---")
            md.append("*Generated by NVIDIA Strategic Intelligence Agent*")

            md_text = "\n".join(md)
            (outputs_dir / "agent_report.md").write_text(md_text, encoding="utf-8")
            print(f"📋 Markdown report saved to {outputs_dir / 'agent_report.md'}")

        if args.report:
            # Legacy: also save to data/report.json if --report flag used
            Path("data").mkdir(exist_ok=True)
            Path("data/report.json").write_text(json.dumps(d, indent=2))
            print("💾 Report also saved to data/report.json")

    print("\n" + "="*60)
    print("✅ Pipeline complete!")
    print("Launch dashboard with:  streamlit run app.py")
    print("="*60)


if __name__ == "__main__":
    main()
