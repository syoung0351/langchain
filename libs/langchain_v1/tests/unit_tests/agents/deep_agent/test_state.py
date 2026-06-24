"""Unit tests for deep_agent state schema (no network calls)."""

from __future__ import annotations

from langchain.agents.deep_agent.state import (
    DeepResearchInput,
    DeepResearchOutput,
    DeepResearchState,
    ResearchPlan,
)


class TestResearchPlan:
    """Tests for the `ResearchPlan` TypedDict."""

    def test_can_instantiate(self) -> None:
        """A `ResearchPlan` can be constructed from explicit fields."""
        plan: ResearchPlan = {
            "main_question": "What is X?",
            "sub_questions": ["Why does X happen?", "When did X emerge?"],
            "search_strategy": "Search academic sources first.",
        }
        assert plan["main_question"] == "What is X?"
        assert len(plan["sub_questions"]) == 2
        assert plan["search_strategy"] != ""

    def test_empty_sub_questions(self) -> None:
        """A `ResearchPlan` with no sub-questions is a valid value object."""
        plan: ResearchPlan = {
            "main_question": "Open question",
            "sub_questions": [],
            "search_strategy": "Free-form exploration.",
        }
        assert plan["sub_questions"] == []


class TestDeepResearchState:
    """Tests for the `DeepResearchState` TypedDict structure."""

    def test_all_expected_keys_present(self) -> None:
        """The TypedDict exposes all expected fields."""
        # TypedDict keys are available via __annotations__
        keys = set(DeepResearchState.__annotations__)
        expected = {
            "messages",
            "research_question",
            "research_plan",
            "gathered_findings",
            "iteration_count",
            "max_iterations",
            "final_report",
            "phase",
        }
        assert expected.issubset(keys)

    def test_can_build_minimal_state(self) -> None:
        """A minimal state dict can be created without error."""
        state: DeepResearchState = {
            "messages": [],
            "research_question": "What is Y?",
            "research_plan": {
                "main_question": "What is Y?",
                "sub_questions": [],
                "search_strategy": "",
            },
            "gathered_findings": [],
            "iteration_count": 0,
            "max_iterations": 3,
            "final_report": "",
            "phase": "planning",
        }
        assert state["iteration_count"] == 0
        assert state["phase"] == "planning"


class TestDeepResearchInput:
    """Tests for the `DeepResearchInput` TypedDict."""

    def test_all_fields_are_optional(self) -> None:
        """All `DeepResearchInput` fields are optional (`total=False`)."""
        # Should not raise
        minimal: DeepResearchInput = {}
        assert isinstance(minimal, dict)

    def test_can_set_research_question(self) -> None:
        """The `research_question` field is settable."""
        payload: DeepResearchInput = {"research_question": "How does X work?"}
        assert payload["research_question"] == "How does X work?"

    def test_can_set_max_iterations(self) -> None:
        """The `max_iterations` field is settable."""
        payload: DeepResearchInput = {"max_iterations": 5}
        assert payload["max_iterations"] == 5


class TestDeepResearchOutput:
    """Tests for the `DeepResearchOutput` TypedDict."""

    def test_all_expected_keys_present(self) -> None:
        """The `DeepResearchOutput` TypedDict exposes the right keys."""
        keys = set(DeepResearchOutput.__annotations__)
        expected = {"messages", "final_report", "gathered_findings", "iteration_count"}
        assert expected == keys

    def test_can_instantiate(self) -> None:
        """A `DeepResearchOutput` can be constructed from explicit fields."""
        output: DeepResearchOutput = {
            "messages": [],
            "final_report": "Report content here.",
            "gathered_findings": ["Finding 1", "Finding 2"],
            "iteration_count": 2,
        }
        assert output["iteration_count"] == 2
        assert len(output["gathered_findings"]) == 2
