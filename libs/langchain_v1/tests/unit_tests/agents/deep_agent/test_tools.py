"""Unit tests for deep_agent tools (no network calls)."""

from __future__ import annotations

import pytest

from langchain.agents.deep_agent.tools import (
    describe_tools,
    fetch_document,
    get_default_tools,
    search_web,
    summarize_findings,
)


class TestSearchWebTool:
    """Tests for the `search_web` mock tool."""

    def test_returns_string(self) -> None:
        """search_web returns a non-empty string."""
        result = search_web.invoke({"query": "quantum computing"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_result_contains_query(self) -> None:
        """The result references the search query."""
        query = "transformer attention mechanism"
        result = search_web.invoke({"query": query})
        assert query in result

    def test_different_queries_produce_different_results(self) -> None:
        """Different queries yield distinct results."""
        result_a = search_web.invoke({"query": "topic A"})
        result_b = search_web.invoke({"query": "topic B"})
        assert result_a != result_b


class TestFetchDocumentTool:
    """Tests for the `fetch_document` mock tool."""

    def test_returns_string(self) -> None:
        """fetch_document returns a non-empty string."""
        result = fetch_document.invoke({"url": "https://example.com/paper"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_result_contains_url(self) -> None:
        """The result references the requested URL."""
        url = "https://example.com/unique-document"
        result = fetch_document.invoke({"url": url})
        assert url in result

    def test_different_urls_produce_different_results(self) -> None:
        """Different URLs yield distinct results."""
        result_a = fetch_document.invoke({"url": "https://example.com/doc1"})
        result_b = fetch_document.invoke({"url": "https://example.com/doc2"})
        assert result_a != result_b


class TestSummarizeFindings:
    """Tests for the `summarize_findings` tool."""

    def test_empty_findings_returns_message(self) -> None:
        """An empty findings list returns a descriptive message."""
        result = summarize_findings.invoke({"findings": []})
        assert "No findings" in result

    def test_single_finding(self) -> None:
        """A single finding is included in the summary."""
        finding = "Attention is all you need"
        result = summarize_findings.invoke({"findings": [finding]})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_multiple_findings_mentioned(self) -> None:
        """Multiple findings are counted in the output."""
        findings = ["Finding one", "Finding two", "Finding three"]
        result = summarize_findings.invoke({"findings": findings})
        # The tool reports the count
        assert "3" in result

    def test_focus_parameter_appears_in_output(self) -> None:
        """The focus parameter is reflected in the output."""
        result = summarize_findings.invoke(
            {"findings": ["some data"], "focus": "technical"}
        )
        assert "technical" in result

    def test_default_focus_is_general(self) -> None:
        """The default focus is 'general'."""
        result = summarize_findings.invoke({"findings": ["data"]})
        assert "general" in result


class TestGetDefaultTools:
    """Tests for the `get_default_tools` factory."""

    def test_returns_list(self) -> None:
        """get_default_tools returns a list."""
        tools = get_default_tools()
        assert isinstance(tools, list)

    def test_returns_three_tools(self) -> None:
        """The default set contains exactly three tools."""
        tools = get_default_tools()
        assert len(tools) == 3

    def test_tool_names(self) -> None:
        """The three expected tools are present."""
        tools = get_default_tools()
        names = {t.name for t in tools}
        assert "search_web" in names
        assert "fetch_document" in names
        assert "summarize_findings" in names

    def test_all_tools_have_descriptions(self) -> None:
        """Every tool has a non-empty description."""
        for tool in get_default_tools():
            assert tool.description.strip(), f"Tool {tool.name!r} has no description"


class TestDescribeTools:
    """Tests for the `describe_tools` helper."""

    def test_returns_dict(self) -> None:
        """describe_tools returns a dict."""
        result = describe_tools()
        assert isinstance(result, dict)

    def test_keys_match_tool_names(self) -> None:
        """The dict keys correspond to the default tool names."""
        result = describe_tools()
        expected_names = {t.name for t in get_default_tools()}
        assert set(result.keys()) == expected_names

    def test_each_entry_has_description(self) -> None:
        """Each tool entry in the description dict has a 'description' key."""
        for name, info in describe_tools().items():
            assert "description" in info, f"Missing 'description' for tool {name!r}"
