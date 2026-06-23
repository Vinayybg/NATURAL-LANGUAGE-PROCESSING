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
        print("(Takes ~2 minutes due to Groq rate limits)")
        print("="*60)
        from nvidia_agent.agent.ceo_agent import run_intelligence_engine, report_to_dict
        report = run_intelligence_engine()
        d      = report_to_dict(report)

        if report.error:
            print(f"\n⚠️  Partial results — {report.error}")

        print(f"\n📊 Opportunities : {len(d['opportunities'])}")
        for o in d["opportunities"]:
            print(f"   [{o['impact_level']}] {o['title']}")

        print(f"\n⚠️  Risks         : {len(d['risks'])}")
        for r in d["risks"]:
            print(f"   [{r['severity']}] {r['title']} — {r['category']}")

        print(f"\n🔭 Trends         : {len(d['trends'])}")
        for t in d["trends"]:
            print(f"   • {t['title']}")

        print(f"\n🎯 Recommendations: {len(d['recommendations'])}")
        for i, r in enumerate(d["recommendations"], 1):
            print(f"   #{i} [{r['priority']}] {r['recommendation'][:80]}")

        briefing = d.get("ceo_briefing", {})
        if briefing:
            print(f"\n📋 CEO Briefing:")
            print(f"   {str(briefing.get('executive_summary', ''))[:300]}")

        if args.report:
            Path("data").mkdir(exist_ok=True)
            Path("data/report.json").write_text(json.dumps(d, indent=2))
            print("\n💾 Report saved to data/report.json")

    print("\n" + "="*60)
    print("✅ Pipeline complete!")
    print("Launch dashboard with:  streamlit run app.py")
    print("="*60)


if __name__ == "__main__":
    main()
