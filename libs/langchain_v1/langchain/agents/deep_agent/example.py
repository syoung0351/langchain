"""Example showing how to use the deep research agent.

Run this script to see the agent in action::

    cd libs/langchain_v1
    uv run python langchain/agents/deep_agent/example.py

The example uses mock tools so no external API keys are needed for the tool
calls themselves.  A real Anthropic API key is required for the LLM calls;
set the ANTHROPIC_API_KEY environment variable before running.
"""

from __future__ import annotations

import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Add the libs/langchain_v1 directory to sys.path when running as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from langchain.agents.deep_agent import create_deep_research_agent  # noqa: E402


def run_example() -> None:
    """Run a demonstration of the deep research agent."""
    print("=" * 70)
    print("Deep Research Agent — Example Run")
    print("=" * 70)

    research_question = (
        "What are the main approaches to making large language models more "
        "efficient, and what trade-offs does each approach involve?"
    )

    print(f"\nResearch question:\n  {research_question}\n")
    print("Creating agent...")

    agent = create_deep_research_agent(
        model="anthropic:claude-sonnet-4-6",
        max_iterations=2,
    )

    print("Invoking agent (this may take a moment)...")
    result = agent.invoke(
        {
            "research_question": research_question,
            "max_iterations": 2,
        }
    )

    print("\n" + "=" * 70)
    print("RESEARCH PLAN")
    print("=" * 70)
    plan = result.get("research_plan", {})
    print(json.dumps(plan, indent=2))

    print("\n" + "=" * 70)
    print(f"GATHERED FINDINGS ({len(result.get('gathered_findings', []))} total)")
    print("=" * 70)
    for i, finding in enumerate(result.get("gathered_findings", []), 1):
        print(f"\n--- Finding {i} ---")
        print(finding[:500] + ("..." if len(finding) > 500 else ""))

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(result.get("final_report", "No report generated."))

    print("\n" + "=" * 70)
    print(f"Completed in {result.get('iteration_count', 0)} research iteration(s).")
    print("=" * 70)


if __name__ == "__main__":
    run_example()
