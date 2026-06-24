"""Deep research agent implementation using LangGraph.

The agent operates in three phases:

1. **Planning** — the LLM decomposes the research question into sub-questions
   and produces a structured `ResearchPlan`.

2. **Researching** — the LLM iteratively calls tools (`search_web`,
   `fetch_document`, `summarize_findings`) to gather evidence for each
   sub-question. After each iteration the agent checks whether enough
   evidence has been collected or whether another iteration is needed.

3. **Synthesizing** — once research is complete, the LLM writes a
   comprehensive final report from all gathered findings.

Example usage::

    from langchain.agents.deep_agent import create_deep_research_agent

    agent = create_deep_research_agent()
    result = agent.invoke({
        "research_question": "How does transformer attention work?",
        "max_iterations": 3,
    })
    print(result["final_report"])
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.prebuilt.tool_node import ToolNode

from langchain.agents.deep_agent.prompts import (
    ITERATION_CHECK_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
)
from langchain.agents.deep_agent.state import DeepResearchState, ResearchPlan
from langchain.agents.deep_agent.tools import get_default_tools
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"
_DEFAULT_MAX_ITERATIONS = 3


def _parse_research_plan(content: str) -> ResearchPlan:
    """Parse an LLM response into a `ResearchPlan`.

    Attempts JSON extraction; falls back to a minimal plan if parsing fails.

    Args:
        content: Raw text content from the planning LLM response.

    Returns:
        A `ResearchPlan` with at least one sub-question.
    """
    text = content.strip()
    # Extract JSON from markdown code fences if present.
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.startswith("{") or stripped.startswith("json\n{"):
                text = stripped.lstrip("json").strip()
                break

    try:
        data = json.loads(text)
        return ResearchPlan(
            main_question=str(data.get("main_question", "")),
            sub_questions=[str(q) for q in data.get("sub_questions", [])],
            search_strategy=str(data.get("search_strategy", "")),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("Failed to parse research plan JSON; using fallback plan.")
        return ResearchPlan(
            main_question=content[:200],
            sub_questions=["What are the key aspects of this topic?"],
            search_strategy="Search for general information and then drill into specifics.",
        )


def _extract_findings(messages: list[Any]) -> list[str]:
    """Pull text from the most recent tool messages as new findings.

    Args:
        messages: The full message history.

    Returns:
        List of non-empty string findings from recent tool responses.
    """
    from langchain_core.messages import ToolMessage

    findings: list[str] = []
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            break
        if isinstance(msg.content, str) and msg.content.strip():
            findings.append(msg.content.strip())
    return findings


def _build_planning_node(
    model: BaseChatModel,
) -> Any:
    """Return a LangGraph node function that produces the research plan.

    The node calls the model once with the planner system prompt and parses the
    resulting JSON into a `ResearchPlan` stored in the graph state.

    Args:
        model: The language model to use for planning.

    Returns:
        A callable that accepts `DeepResearchState` and returns a state update dict.
    """

    def planning_node(state: DeepResearchState) -> dict[str, Any]:
        """Execute the planning phase."""
        question = state.get("research_question", "")
        if not question:
            # Fall back to the last human message if no explicit question.
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    question = str(msg.content)
                    break

        planning_messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Please create a research plan for the following question:\n\n{question}"
            ),
        ]

        response = model.invoke(planning_messages)
        plan = _parse_research_plan(str(response.content))

        logger.debug(
            "Research plan created: %d sub-questions",
            len(plan["sub_questions"]),
        )

        plan_summary = (
            f"Research plan created:\n"
            f"Main question: {plan['main_question']}\n"
            f"Sub-questions:\n"
            + "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(plan["sub_questions"]))
            + f"\nStrategy: {plan['search_strategy']}"
        )

        return {
            "research_plan": plan,
            "research_question": question,
            "gathered_findings": [],
            "iteration_count": 0,
            "phase": "researching",
            "messages": [AIMessage(content=plan_summary)],
        }

    return planning_node


def _build_research_node(
    model: BaseChatModel,
    tools: list[BaseTool],
) -> Any:
    """Return a LangGraph node function that runs one research iteration.

    The node gives the model access to the provided tools and asks it to
    address the next open sub-question from the plan.

    Args:
        model: The language model with tool-calling capability.
        tools: The tools available to the researcher.

    Returns:
        A callable that accepts `DeepResearchState` and returns a state update dict.
    """
    tool_names = ", ".join(t.name for t in tools)
    bound_model = model.bind_tools(tools)

    def research_node(state: DeepResearchState) -> dict[str, Any]:
        """Execute one research iteration."""
        plan = state.get("research_plan", {})
        sub_questions = plan.get("sub_questions", [])
        findings_so_far = state.get("gathered_findings", [])
        iteration = state.get("iteration_count", 0)

        answered_count = min(iteration, len(sub_questions))
        remaining = sub_questions[answered_count:]
        current_question = remaining[0] if remaining else plan.get("main_question", "")

        research_prompt = (
            f"{RESEARCHER_SYSTEM_PROMPT}\n\n"
            f"Available tools: {tool_names}\n\n"
            f"Overall research question: {plan.get('main_question', '')}\n\n"
            f"Current sub-question to address: {current_question}\n\n"
            f"Findings gathered so far ({len(findings_so_far)}):\n"
            + (
                "\n".join(f"- {f[:200]}..." if len(f) > 200 else f"- {f}" for f in findings_so_far)
                or "None yet."
            )
        )

        messages = [
            SystemMessage(content=research_prompt),
            *state["messages"],
        ]

        response = bound_model.invoke(messages)

        return {
            "messages": [response],
            "iteration_count": iteration + 1,
            "phase": "researching",
        }

    return research_node


def _build_tool_result_collector_node() -> Any:
    """Return a node that collects tool outputs into `gathered_findings`.

    Runs after the tool node so that each tool response is stored in the
    `gathered_findings` list for use during synthesis.

    Returns:
        A callable that accepts `DeepResearchState` and returns a state update dict.
    """

    def collect_tool_results(state: DeepResearchState) -> dict[str, Any]:
        """Extract tool call results and append them to gathered_findings."""
        new_findings = _extract_findings(state["messages"])
        existing = list(state.get("gathered_findings", []))
        combined = existing + [f for f in new_findings if f not in existing]
        return {"gathered_findings": combined}

    return collect_tool_results


def _build_iteration_check_node(model: BaseChatModel) -> Any:
    """Return a node that decides whether to continue researching or synthesize.

    The model is given the research plan, the count of iterations already
    performed, and the findings gathered so far.  It returns either
    "CONTINUE" or "SYNTHESIZE".

    Args:
        model: The language model to use for the decision.

    Returns:
        A callable that accepts `DeepResearchState` and returns a state update dict.
    """

    def iteration_check_node(state: DeepResearchState) -> dict[str, Any]:
        """Decide whether to continue researching or move to synthesis."""
        plan = state.get("research_plan", {})
        findings = state.get("gathered_findings", [])
        iteration = state.get("iteration_count", 0)
        max_iters = state.get("max_iterations", _DEFAULT_MAX_ITERATIONS)

        if iteration >= max_iters:
            logger.debug(
                "Reached max iterations (%d); proceeding to synthesis.",
                max_iters,
            )
            return {"phase": "synthesizing"}

        check_messages = [
            SystemMessage(content=ITERATION_CHECK_PROMPT),
            HumanMessage(
                content=(
                    f"Research plan: {json.dumps(plan)}\n\n"
                    f"Iterations completed: {iteration}/{max_iters}\n\n"
                    f"Findings gathered so far ({len(findings)}):\n"
                    + (
                        "\n".join(
                            f"- {f[:300]}..." if len(f) > 300 else f"- {f}"
                            for f in findings
                        )
                        or "None yet."
                    )
                )
            ),
        ]

        response = model.invoke(check_messages)
        decision = str(response.content).strip().upper()

        if "SYNTHESIZE" in decision:
            logger.debug("Model decided to synthesize after %d iteration(s).", iteration)
            return {"phase": "synthesizing"}

        logger.debug("Model decided to continue researching (iteration %d).", iteration)
        return {"phase": "researching"}

    return iteration_check_node


def _build_synthesis_node(model: BaseChatModel) -> Any:
    """Return a node that synthesizes all findings into the final report.

    Args:
        model: The language model to use for synthesis.

    Returns:
        A callable that accepts `DeepResearchState` and returns a state update dict.
    """

    def synthesis_node(state: DeepResearchState) -> dict[str, Any]:
        """Write the final research report from all gathered findings."""
        question = state.get("research_question", "")
        plan = state.get("research_plan", {})
        findings = state.get("gathered_findings", [])
        iterations = state.get("iteration_count", 0)

        synthesis_prompt = (
            f"{SYNTHESIZER_SYSTEM_PROMPT}\n\n"
            f"Original research question: {question}\n\n"
            f"Research plan:\n{json.dumps(plan, indent=2)}\n\n"
            f"Research iterations completed: {iterations}\n\n"
            f"All gathered findings ({len(findings)}):\n"
            + "\n\n".join(f"Finding {i + 1}:\n{f}" for i, f in enumerate(findings))
        )

        response = model.invoke([SystemMessage(content=synthesis_prompt)])
        report = str(response.content)

        logger.debug("Synthesis complete; report length: %d characters.", len(report))

        return {
            "final_report": report,
            "phase": "complete",
            "messages": [
                AIMessage(content=f"Research complete. Final report:\n\n{report}")
            ],
        }

    return synthesis_node


def _make_routing_edge(
    *,
    research_node_name: str,
    check_node_name: str,
    synthesis_node_name: str,
    tool_node_name: str,
) -> Any:
    """Return a conditional-edge function that routes after the research node.

    If the last AI message has tool calls, route to the tool executor.
    Otherwise go directly to the iteration-check node.

    Args:
        research_node_name: Name of the research node (unused but kept for clarity).
        check_node_name: Name of the iteration-check node.
        synthesis_node_name: Name of the synthesis node (unused but kept for clarity).
        tool_node_name: Name of the tool node.

    Returns:
        A callable suitable for use as a LangGraph conditional edge.
    """

    def route_after_research(state: DeepResearchState) -> str:
        """Route to tools if there are pending tool calls, else check iterations."""
        messages = state["messages"]
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    return tool_node_name
                break
        return check_node_name

    return route_after_research


def _make_phase_edge(
    *,
    research_node_name: str,
    synthesis_node_name: str,
) -> Any:
    """Return a conditional-edge function that routes based on the current phase.

    Routes from the iteration-check node to either another research iteration
    or the synthesis node.

    Args:
        research_node_name: Destination when more research is needed.
        synthesis_node_name: Destination when synthesis should begin.

    Returns:
        A callable suitable for use as a LangGraph conditional edge.
    """

    def route_by_phase(state: DeepResearchState) -> str:
        """Route based on the phase field set by the iteration-check node."""
        phase = state.get("phase", "researching")
        if phase == "synthesizing":
            return synthesis_node_name
        return research_node_name

    return route_by_phase


def create_deep_research_agent(
    model: str | BaseChatModel = _DEFAULT_MODEL,
    tools: list[BaseTool] | None = None,
    *,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Create a deep research agent that iteratively gathers and synthesizes information.

    The agent follows a three-phase loop:

    1. **Planning** — decomposes the research question into focused sub-questions.
    2. **Researching** — calls tools to gather evidence (repeats up to `max_iterations`).
    3. **Synthesizing** — writes a structured final report from all findings.

    Args:
        model: The language model to use. Accepts either a model identifier string
            (e.g., `"anthropic:claude-sonnet-4-6"`) or a pre-built `BaseChatModel`
            instance.
        tools: Tools available during the research phase. Defaults to
            `[search_web, fetch_document, summarize_findings]` if not provided.
        max_iterations: Maximum number of research iterations before the agent
            is forced into the synthesis phase regardless of the model's decision.

    Returns:
        A compiled `StateGraph` ready to invoke with a `DeepResearchInput` dict.

    Example:
        ```python
        from langchain.agents.deep_agent import create_deep_research_agent

        agent = create_deep_research_agent(
            model="anthropic:claude-sonnet-4-6",
            max_iterations=3,
        )
        result = agent.invoke({
            "research_question": "What are the key challenges in quantum computing?",
        })
        print(result["final_report"])
        ```
    """
    if isinstance(model, str):
        model = init_chat_model(model)

    research_tools = tools if tools is not None else get_default_tools()

    # Build nodes
    planning_node = _build_planning_node(model)
    research_node = _build_research_node(model, research_tools)
    tool_node = ToolNode(tools=research_tools)
    collector_node = _build_tool_result_collector_node()
    check_node = _build_iteration_check_node(model)
    synthesis_node = _build_synthesis_node(model)

    # Node names
    PLAN = "plan"
    RESEARCH = "research"
    TOOLS = "tools"
    COLLECT = "collect_findings"
    CHECK = "check_iteration"
    SYNTHESIZE = "synthesize"

    # Build graph
    graph: StateGraph[DeepResearchState, Any, Any, Any] = StateGraph(DeepResearchState)

    graph.add_node(PLAN, planning_node)
    graph.add_node(RESEARCH, research_node)
    graph.add_node(TOOLS, tool_node)
    graph.add_node(COLLECT, collector_node)
    graph.add_node(CHECK, check_node)
    graph.add_node(SYNTHESIZE, synthesis_node)

    # Edges
    graph.add_edge(START, PLAN)
    graph.add_edge(PLAN, RESEARCH)

    # After research: either call tools or check whether to continue
    graph.add_conditional_edges(
        RESEARCH,
        _make_routing_edge(
            research_node_name=RESEARCH,
            check_node_name=CHECK,
            synthesis_node_name=SYNTHESIZE,
            tool_node_name=TOOLS,
        ),
        [TOOLS, CHECK],
    )

    # After tools: collect findings, then check whether to continue
    graph.add_edge(TOOLS, COLLECT)
    graph.add_edge(COLLECT, CHECK)

    # After check: either loop back to research or proceed to synthesis
    graph.add_conditional_edges(
        CHECK,
        _make_phase_edge(research_node_name=RESEARCH, synthesis_node_name=SYNTHESIZE),
        [RESEARCH, SYNTHESIZE],
    )

    graph.add_edge(SYNTHESIZE, END)

    return graph.compile()


__all__ = [
    "create_deep_research_agent",
]
