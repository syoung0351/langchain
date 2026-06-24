"""Mock and utility tools for the deep research agent.

These tools provide research capabilities without requiring external API keys.
In production, replace the mock implementations with real integrations
(e.g., Tavily, SerpAPI, or a vector store retriever).
"""

from __future__ import annotations

import textwrap
from typing import Any

from langchain_core.tools import BaseTool, tool


@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic.

    Returns a summary of relevant web results for the given query.
    In production, wire this to a real search provider such as Tavily or SerpAPI.

    Args:
        query: The search query string.

    Returns:
        A string summarising the mock search results for the query.
    """
    # Mock implementation — replace with a real search provider in production.
    return textwrap.dedent(f"""\
        Search results for: "{query}"

        [Result 1] Overview of {query}
        Topic summary: {query} is a widely studied subject with multiple dimensions.
        Key aspects include foundational concepts, recent developments, and practical
        applications across various domains.

        [Result 2] Recent research on {query}
        Researchers have identified several important factors related to {query}.
        Studies show measurable impacts on related fields, with ongoing work exploring
        edge cases and nuanced interactions.

        [Result 3] Practical applications of {query}
        Industry practitioners have applied insights from {query} in real-world
        settings, achieving notable outcomes in efficiency and effectiveness.
    """)


@tool
def fetch_document(url: str) -> str:
    """Fetch and extract the text content of a document at the given URL.

    In production, replace this with a real HTTP fetch + HTML-to-text extractor.

    Args:
        url: The URL of the document to fetch.

    Returns:
        The extracted text content of the document.
    """
    # Mock implementation.
    return textwrap.dedent(f"""\
        Document content from: {url}

        Introduction
        This document provides a detailed examination of the subject matter.
        The author presents evidence gathered from multiple primary sources,
        building a case for the main thesis through logical argumentation.

        Key Findings
        - Finding 1: The primary mechanism operates through iterative refinement.
        - Finding 2: Context dependency plays a significant role in outcomes.
        - Finding 3: Cross-domain transfer is possible but requires adaptation.

        Conclusion
        The evidence supports a nuanced view that avoids oversimplification.
        Future research should address the identified gaps in current understanding.
    """)


@tool
def summarize_findings(
    findings: list[str],
    *,
    focus: str = "general",
) -> str:
    """Synthesize a list of research findings into a coherent summary.

    Args:
        findings: Individual research findings or excerpts to be synthesized.
        focus: The thematic focus for the synthesis (e.g., 'technical', 'historical').

    Returns:
        A synthesized summary integrating all provided findings.
    """
    if not findings:
        return "No findings provided to summarize."

    joined = "\n".join(f"- {f}" for f in findings)
    return textwrap.dedent(f"""\
        Synthesized summary (focus: {focus}):

        The following {len(findings)} finding(s) were integrated:
        {joined}

        Synthesis:
        Across the collected findings, several consistent themes emerge.
        The evidence collectively points toward a coherent understanding of the topic,
        with key patterns observable across independent sources. Areas of apparent
        contradiction merit further investigation and may reflect context-specific
        conditions rather than fundamental disagreements.
    """)


def get_default_tools() -> list[BaseTool]:
    """Return the default tool set for the deep research agent.

    Returns:
        List of `BaseTool` instances available to the agent.
    """
    return [search_web, fetch_document, summarize_findings]  # type: ignore[list-item]


__all__ = [
    "fetch_document",
    "get_default_tools",
    "search_web",
    "summarize_findings",
]


def _get_tool_by_name(name: str) -> BaseTool | None:
    """Look up a default tool by name.

    Args:
        name: The tool name to look up.

    Returns:
        The matching `BaseTool`, or `None` if not found.
    """
    return next((t for t in get_default_tools() if t.name == name), None)


def describe_tools() -> dict[str, Any]:
    """Return a machine-readable description of all default tools.

    Returns:
        Mapping of tool name to a dict with 'description' and 'args_schema' keys.
    """
    result: dict[str, Any] = {}
    for t in get_default_tools():
        result[t.name] = {
            "description": t.description,
            "args_schema": t.args_schema.model_json_schema() if t.args_schema else None,
        }
    return result
