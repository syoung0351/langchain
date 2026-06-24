"""Unit tests for the deep research agent graph (no network calls).

All LLM calls are replaced with `FakeToolCallingModel` or plain mock objects
so these tests run entirely offline.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain.agents.deep_agent.agent import (
    _build_iteration_check_node,
    _build_planning_node,
    _build_synthesis_node,
    _build_tool_result_collector_node,
    _extract_findings,
    _make_phase_edge,
    _make_routing_edge,
    _parse_research_plan,
    create_deep_research_agent,
)
from langchain.agents.deep_agent.state import DeepResearchState, ResearchPlan
from langchain.agents.deep_agent.tools import get_default_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides: Any) -> DeepResearchState:
    """Build a minimal `DeepResearchState` for testing.

    Args:
        **overrides: Fields to override in the base state.

    Returns:
        A fully populated `DeepResearchState` dict.
    """
    base: DeepResearchState = {
        "messages": [],
        "research_question": "What is deep learning?",
        "research_plan": {
            "main_question": "What is deep learning?",
            "sub_questions": ["How do neural networks learn?", "What are common architectures?"],
            "search_strategy": "Search academic sources.",
        },
        "gathered_findings": [],
        "iteration_count": 0,
        "max_iterations": 3,
        "final_report": "",
        "phase": "researching",
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def _fake_model_response(content: str) -> MagicMock:
    """Create a mock `BaseChatModel` that always returns `content`.

    Args:
        content: Text content to return on every `invoke` call.

    Returns:
        A mock object with an `invoke` method and `bind_tools` that returns itself.
    """
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content=content)
    mock.bind_tools.return_value = mock
    return mock


# ---------------------------------------------------------------------------
# Tests: _parse_research_plan
# ---------------------------------------------------------------------------


class TestParseResearchPlan:
    """Tests for the JSON parsing helper."""

    def test_valid_json(self) -> None:
        """Valid JSON is parsed into a `ResearchPlan`."""
        payload = {
            "main_question": "What is X?",
            "sub_questions": ["Sub A", "Sub B"],
            "search_strategy": "Use academic search.",
        }
        plan = _parse_research_plan(json.dumps(payload))
        assert plan["main_question"] == "What is X?"
        assert plan["sub_questions"] == ["Sub A", "Sub B"]
        assert plan["search_strategy"] == "Use academic search."

    def test_json_inside_code_fence(self) -> None:
        """JSON wrapped in a markdown code fence is extracted and parsed."""
        payload = json.dumps({"main_question": "Q", "sub_questions": ["S1"], "search_strategy": "S"})
        fenced = f"```json\n{payload}\n```"
        plan = _parse_research_plan(fenced)
        assert plan["main_question"] == "Q"

    def test_malformed_json_returns_fallback(self) -> None:
        """Malformed JSON produces a valid fallback `ResearchPlan`."""
        plan = _parse_research_plan("Not valid JSON at all!")
        assert isinstance(plan["main_question"], str)
        assert isinstance(plan["sub_questions"], list)
        assert len(plan["sub_questions"]) >= 1

    def test_missing_fields_do_not_raise(self) -> None:
        """JSON missing optional fields is handled without raising."""
        plan = _parse_research_plan(json.dumps({"main_question": "Only this"}))
        assert plan["main_question"] == "Only this"
        assert isinstance(plan["sub_questions"], list)


# ---------------------------------------------------------------------------
# Tests: _extract_findings
# ---------------------------------------------------------------------------


class TestExtractFindings:
    """Tests for the tool-result extraction helper."""

    def test_extracts_tool_messages_after_last_ai_message(self) -> None:
        """Tool messages following the last AIMessage are returned as findings."""
        messages = [
            HumanMessage(content="Start"),
            AIMessage(content="Thinking...", tool_calls=[]),
            ToolMessage(content="Finding 1", tool_call_id="c1"),
            ToolMessage(content="Finding 2", tool_call_id="c2"),
        ]
        findings = _extract_findings(messages)
        assert "Finding 1" in findings
        assert "Finding 2" in findings

    def test_stops_at_non_tool_message(self) -> None:
        """Traversal stops when a non-ToolMessage is encountered before the scan ends."""
        messages = [
            HumanMessage(content="Start"),
            ToolMessage(content="Old finding", tool_call_id="c0"),
            AIMessage(content="Middle response", tool_calls=[]),
            ToolMessage(content="New finding", tool_call_id="c1"),
        ]
        findings = _extract_findings(messages)
        assert "New finding" in findings
        # "Old finding" is before the last AIMessage so should not appear
        assert "Old finding" not in findings

    def test_empty_messages_returns_empty_list(self) -> None:
        """An empty message list returns an empty findings list."""
        assert _extract_findings([]) == []

    def test_empty_tool_message_content_is_skipped(self) -> None:
        """Tool messages with empty content are not added to findings."""
        messages = [
            AIMessage(content="Response", tool_calls=[]),
            ToolMessage(content="   ", tool_call_id="c1"),
        ]
        findings = _extract_findings(messages)
        assert findings == []


# ---------------------------------------------------------------------------
# Tests: routing edge functions
# ---------------------------------------------------------------------------


class TestMakeRoutingEdge:
    """Tests for the research-phase routing edge."""

    def test_routes_to_tools_when_tool_calls_present(self) -> None:
        """Routes to the tool node when the last AIMessage has tool calls."""
        route = _make_routing_edge(
            research_node_name="research",
            check_node_name="check",
            synthesis_node_name="synthesize",
            tool_node_name="tools",
        )
        state = _make_state(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_web", "args": {"query": "x"}, "id": "c1", "type": "tool_call"}],
                )
            ]
        )
        assert route(state) == "tools"

    def test_routes_to_check_when_no_tool_calls(self) -> None:
        """Routes to the check node when the last AIMessage has no tool calls."""
        route = _make_routing_edge(
            research_node_name="research",
            check_node_name="check",
            synthesis_node_name="synthesize",
            tool_node_name="tools",
        )
        state = _make_state(messages=[AIMessage(content="Plain response")])
        assert route(state) == "check"

    def test_routes_to_check_with_empty_messages(self) -> None:
        """Routes to the check node when the message list is empty."""
        route = _make_routing_edge(
            research_node_name="research",
            check_node_name="check",
            synthesis_node_name="synthesize",
            tool_node_name="tools",
        )
        state = _make_state(messages=[])
        assert route(state) == "check"


class TestMakePhaseEdge:
    """Tests for the iteration-check phase-routing edge."""

    def test_routes_to_synthesis_when_phase_is_synthesizing(self) -> None:
        """Routes to the synthesis node when phase == 'synthesizing'."""
        route = _make_phase_edge(research_node_name="research", synthesis_node_name="synthesize")
        state = _make_state(phase="synthesizing")
        assert route(state) == "synthesize"

    def test_routes_to_research_when_phase_is_researching(self) -> None:
        """Routes to the research node when phase == 'researching'."""
        route = _make_phase_edge(research_node_name="research", synthesis_node_name="synthesize")
        state = _make_state(phase="researching")
        assert route(state) == "research"

    def test_routes_to_research_for_unknown_phase(self) -> None:
        """Routes to the research node for any unrecognised phase value."""
        route = _make_phase_edge(research_node_name="research", synthesis_node_name="synthesize")
        state = _make_state(phase="unknown_phase")
        assert route(state) == "research"


# ---------------------------------------------------------------------------
# Tests: planning node
# ---------------------------------------------------------------------------


class TestPlanningNode:
    """Tests for the planning graph node."""

    def test_returns_research_plan_field(self) -> None:
        """The planning node returns a 'research_plan' key in the update dict."""
        plan_json = json.dumps({
            "main_question": "What is AI?",
            "sub_questions": ["How does AI learn?"],
            "search_strategy": "Search widely.",
        })
        model = _fake_model_response(plan_json)
        node = _build_planning_node(model)
        state = _make_state(research_question="What is AI?")
        result = node(state)
        assert "research_plan" in result
        assert result["research_plan"]["main_question"] == "What is AI?"

    def test_sets_phase_to_researching(self) -> None:
        """The planning node always sets phase to 'researching'."""
        model = _fake_model_response(
            json.dumps({"main_question": "Q", "sub_questions": ["S"], "search_strategy": "st"})
        )
        node = _build_planning_node(model)
        result = node(_make_state(research_question="Q"))
        assert result["phase"] == "researching"

    def test_initialises_iteration_count(self) -> None:
        """The planning node initialises iteration_count to 0."""
        model = _fake_model_response(
            json.dumps({"main_question": "Q", "sub_questions": [], "search_strategy": ""})
        )
        node = _build_planning_node(model)
        result = node(_make_state())
        assert result["iteration_count"] == 0

    def test_falls_back_to_last_human_message_when_no_question(self) -> None:
        """The node extracts the question from the last HumanMessage if not set."""
        model = _fake_model_response(
            json.dumps({"main_question": "fallback", "sub_questions": [], "search_strategy": ""})
        )
        node = _build_planning_node(model)
        state = _make_state(
            research_question="",
            messages=[HumanMessage(content="My fallback question")],
        )
        result = node(state)
        assert result["research_question"] == "My fallback question"


# ---------------------------------------------------------------------------
# Tests: tool result collector node
# ---------------------------------------------------------------------------


class TestToolResultCollectorNode:
    """Tests for the finding-collection node."""

    def test_appends_new_findings(self) -> None:
        """New tool-message findings are appended to existing findings."""
        node = _build_tool_result_collector_node()
        state = _make_state(
            gathered_findings=["Old finding"],
            messages=[
                AIMessage(content="", tool_calls=[]),
                ToolMessage(content="New finding", tool_call_id="c1"),
            ],
        )
        result = node(state)
        assert "Old finding" in result["gathered_findings"]
        assert "New finding" in result["gathered_findings"]

    def test_deduplicates_findings(self) -> None:
        """Findings that already exist are not added again."""
        node = _build_tool_result_collector_node()
        existing = "Already collected"
        state = _make_state(
            gathered_findings=[existing],
            messages=[
                AIMessage(content="", tool_calls=[]),
                ToolMessage(content=existing, tool_call_id="c1"),
            ],
        )
        result = node(state)
        assert result["gathered_findings"].count(existing) == 1

    def test_handles_no_tool_messages(self) -> None:
        """The collector is a no-op when there are no tool messages."""
        node = _build_tool_result_collector_node()
        state = _make_state(
            gathered_findings=["Existing finding"],
            messages=[AIMessage(content="No tools here")],
        )
        result = node(state)
        assert result["gathered_findings"] == ["Existing finding"]


# ---------------------------------------------------------------------------
# Tests: iteration check node
# ---------------------------------------------------------------------------


class TestIterationCheckNode:
    """Tests for the iteration-decision node."""

    def test_forces_synthesis_at_max_iterations(self) -> None:
        """Phase is set to 'synthesizing' when max_iterations is reached."""
        model = _fake_model_response("SYNTHESIZE")
        node = _build_iteration_check_node(model)
        state = _make_state(iteration_count=3, max_iterations=3)
        result = node(state)
        assert result["phase"] == "synthesizing"

    def test_synthesize_when_model_says_synthesize(self) -> None:
        """Phase is 'synthesizing' when the model responds with SYNTHESIZE."""
        model = _fake_model_response("SYNTHESIZE")
        node = _build_iteration_check_node(model)
        state = _make_state(iteration_count=1, max_iterations=5)
        result = node(state)
        assert result["phase"] == "synthesizing"

    def test_continue_when_model_says_continue(self) -> None:
        """Phase remains 'researching' when the model responds with CONTINUE."""
        model = _fake_model_response("CONTINUE")
        node = _build_iteration_check_node(model)
        state = _make_state(iteration_count=1, max_iterations=5)
        result = node(state)
        assert result["phase"] == "researching"

    def test_defaults_to_continue_for_ambiguous_response(self) -> None:
        """An ambiguous model response defaults to 'researching'."""
        model = _fake_model_response("I am not sure")
        node = _build_iteration_check_node(model)
        state = _make_state(iteration_count=0, max_iterations=5)
        result = node(state)
        assert result["phase"] == "researching"


# ---------------------------------------------------------------------------
# Tests: synthesis node
# ---------------------------------------------------------------------------


class TestSynthesisNode:
    """Tests for the synthesis graph node."""

    def test_populates_final_report(self) -> None:
        """The synthesis node sets the 'final_report' field in the update dict."""
        expected_report = "This is the final research report."
        model = _fake_model_response(expected_report)
        node = _build_synthesis_node(model)
        state = _make_state(gathered_findings=["Finding 1", "Finding 2"])
        result = node(state)
        assert result["final_report"] == expected_report

    def test_sets_phase_to_complete(self) -> None:
        """The synthesis node sets phase to 'complete'."""
        model = _fake_model_response("Report text")
        node = _build_synthesis_node(model)
        result = node(_make_state())
        assert result["phase"] == "complete"

    def test_adds_ai_message_with_report(self) -> None:
        """The synthesis node adds an AIMessage containing the report."""
        model = _fake_model_response("Report content")
        node = _build_synthesis_node(model)
        result = node(_make_state())
        messages = result.get("messages", [])
        assert any(isinstance(m, AIMessage) and "Report content" in str(m.content) for m in messages)


# ---------------------------------------------------------------------------
# Tests: create_deep_research_agent (graph structure)
# ---------------------------------------------------------------------------


class TestCreateDeepResearchAgent:
    """Tests for the `create_deep_research_agent` factory."""

    def test_creates_compiled_graph(self) -> None:
        """The factory returns a compiled graph with an `invoke` method."""
        model = _fake_model_response("{}")
        agent = create_deep_research_agent(model=model)
        assert hasattr(agent, "invoke")

    def test_accepts_custom_tools(self) -> None:
        """The factory accepts a custom tool list without error."""
        model = _fake_model_response("{}")
        tools = get_default_tools()
        agent = create_deep_research_agent(model=model, tools=tools)
        assert hasattr(agent, "invoke")

    def test_accepts_max_iterations(self) -> None:
        """The factory accepts a max_iterations parameter."""
        model = _fake_model_response("{}")
        agent = create_deep_research_agent(model=model, max_iterations=2)
        assert agent is not None

    def test_agent_invoke_with_fake_model(self) -> None:
        """The agent can be invoked end-to-end with a fake model (no network calls).

        This test wires the fake model to produce:
        1. A valid plan JSON (planning phase)
        2. An AI message with no tool calls (research phase, skips tools)
        3. 'SYNTHESIZE' (iteration check)
        4. A final report (synthesis phase)
        """
        plan_json = json.dumps({
            "main_question": "Test question",
            "sub_questions": ["Sub 1"],
            "search_strategy": "Direct search.",
        })

        # The model's invoke is called multiple times; cycle through canned responses.
        responses = [
            AIMessage(content=plan_json),           # planning node
            AIMessage(content="Research done."),    # research node (no tool calls)
            AIMessage(content="SYNTHESIZE"),        # iteration check
            AIMessage(content="Final report text"), # synthesis node
        ]
        call_count = [0]

        def side_effect(*args: Any, **kwargs: Any) -> AIMessage:
            idx = call_count[0] % len(responses)
            call_count[0] += 1
            return responses[idx]

        model = MagicMock()
        model.invoke.side_effect = side_effect
        model.bind_tools.return_value = model

        agent = create_deep_research_agent(model=model, max_iterations=2)
        result = agent.invoke({"research_question": "Test question"})

        # The final_report field should be populated.
        assert "final_report" in result
        assert result["final_report"] != ""
