"""State schema definitions for the deep research agent."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ResearchPlan(TypedDict):
    """A structured research plan produced in the planning phase.

    Attributes:
        main_question: The primary research question to answer.
        sub_questions: A list of focused sub-questions derived from the main question.
        search_strategy: A brief description of the intended search approach.
    """

    main_question: str
    sub_questions: list[str]
    search_strategy: str


class DeepResearchState(TypedDict):
    """Graph state for the deep research agent.

    This schema extends the base message-list state with research-specific
    fields that persist across the planning, iteration, and synthesis phases.

    Attributes:
        messages: Full conversation history (auto-merged via `add_messages`).
        research_question: The original question provided by the user.
        research_plan: The plan generated during the planning step.
        gathered_findings: Accumulated findings collected across all iterations.
        iteration_count: Number of research iterations completed so far.
        max_iterations: Maximum allowed iterations before forcing synthesis.
        final_report: The synthesized research report (populated at the end).
        phase: Current agent phase: 'planning', 'researching', or 'synthesizing'.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    research_question: str
    research_plan: ResearchPlan
    gathered_findings: list[str]
    iteration_count: int
    max_iterations: int
    final_report: str
    phase: str


class DeepResearchInput(TypedDict, total=False):
    """Input schema for invoking the deep research agent.

    Attributes:
        messages: Initial messages (typically a single `HumanMessage`).
        research_question: The research topic or question to investigate.
        max_iterations: Maximum research iterations before forcing synthesis.
    """

    messages: list[Any]
    research_question: str
    max_iterations: int


class DeepResearchOutput(TypedDict):
    """Output schema returned when the deep research agent finishes.

    Attributes:
        messages: Complete conversation history.
        final_report: The finished research report.
        gathered_findings: All findings accumulated during research.
        iteration_count: Total research iterations that were completed.
    """

    messages: list[AnyMessage]
    final_report: str
    gathered_findings: list[str]
    iteration_count: int


__all__ = [
    "DeepResearchInput",
    "DeepResearchOutput",
    "DeepResearchState",
    "ResearchPlan",
]
