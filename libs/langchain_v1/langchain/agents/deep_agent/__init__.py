"""Deep research agent for multi-step iterative information gathering and synthesis.

This module provides `create_deep_research_agent`, a LangGraph-based agent that:

1. Decomposes a research question into focused sub-questions (planning phase).
2. Iteratively calls tools to gather evidence for each sub-question (research phase).
3. Synthesizes all findings into a structured final report (synthesis phase).

Basic usage::

    from langchain.agents.deep_agent import create_deep_research_agent

    agent = create_deep_research_agent(model="anthropic:claude-sonnet-4-6")
    result = agent.invoke({
        "research_question": "How do transformer models learn contextual representations?",
        "max_iterations": 3,
    })
    print(result["final_report"])

The agent ships with lightweight mock tools so it runs without external API keys.
Swap in real integrations (Tavily, SerpAPI, etc.) by passing a `tools` list::

    from langchain_community.tools.tavily_search import TavilySearchResults

    agent = create_deep_research_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=[TavilySearchResults(max_results=5)],
    )
"""

from langchain.agents.deep_agent.agent import create_deep_research_agent
from langchain.agents.deep_agent.state import (
    DeepResearchInput,
    DeepResearchOutput,
    DeepResearchState,
    ResearchPlan,
)
from langchain.agents.deep_agent.tools import (
    fetch_document,
    get_default_tools,
    search_web,
    summarize_findings,
)

__all__ = [
    "DeepResearchInput",
    "DeepResearchOutput",
    "DeepResearchState",
    "ResearchPlan",
    "create_deep_research_agent",
    "fetch_document",
    "get_default_tools",
    "search_web",
    "summarize_findings",
]
