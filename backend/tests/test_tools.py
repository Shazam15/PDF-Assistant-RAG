import pytest
from app.config import Settings
from app.rag.agent import route_query
from app.rag.tools import StatisticsTool, execute_tool


def test_statistics_tool_calculates_mean_and_correlation():
    tool = StatisticsTool()
    result = tool._run(data=[1, 2, 3, 4], operation="mean")
    assert "mean" in result.lower()
    assert "2.5" in result

    corr_result = tool._run(data=[1, 2, 3, 4], operation="correlation_pearson", other_data=[1, 2, 3, 4])
    assert "pearson" in corr_result.lower()
    assert "1.0" in corr_result


def test_statistics_tool_rejects_invalid_input():
    tool = StatisticsTool()

    with pytest.raises(ValueError):
        tool._run(data=[], operation="mean")

    with pytest.raises(ValueError):
        tool._run(data=[1, "two", 3], operation="mean")

    with pytest.raises(ValueError):
        tool._run(data=[1, 2, 3], operation="unsupported")


def test_execute_tool_dispatches_statistics():
    result = execute_tool("statistics", {"data": [1, 2, 3], "operation": "mean"})
    assert "mean" in result.lower()
    assert "2.0" in result


def test_settings_parse_mcp_json_and_allowlist():
    settings = Settings(MCP_ENABLED=True, MCP_SERVERS_JSON='{"fake":{"transport":"stdio","command":"python","args":["-m","fake"]}}', MCP_TOOL_ALLOWLIST="fake.tool")
    assert settings.MCP_ENABLED is True
    assert settings.MCP_SERVERS["fake"]["command"] == "python"
    assert settings.MCP_TOOL_ALLOWLIST == ["fake.tool"]


def test_settings_reject_invalid_mcp_json():
    with pytest.raises(ValueError):
        Settings(MCP_SERVERS_JSON="not-json")


def test_routing_uses_tool_agent_for_statistical_requests():
    decision = route_query("calcula la media de estos valores", routing_mode="auto")
    assert decision.route == "tool_agent"
