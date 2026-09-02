import sys
import types

import pytest
from app.config import Settings, get_settings
from app.rag.agent import route_query
from app.rag.tools import StatisticsTool, execute_tool, _is_tool_allowed


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


def test_default_mcp_allowlist_has_no_write_capable_tools():
    """The default allowlist must stay read-only; write/delete/edit access has to be
    an explicit, deliberate opt-in via MCP_TOOL_ALLOWLIST, never a shipped default."""
    default_allowlist = get_settings().MCP_TOOL_ALLOWLIST
    dangerous_markers = ("write", "delete", "edit", "move", "create_directory")
    for tool_name in default_allowlist:
        assert not any(marker in tool_name.lower() for marker in dangerous_markers), (
            f"Default MCP_TOOL_ALLOWLIST unexpectedly allows a mutating tool: {tool_name}"
        )


def test_is_tool_allowed_requires_explicit_allowlisting(monkeypatch):
    import app.rag.tools as tools_module

    monkeypatch.setattr(tools_module.settings, "MCP_TOOL_ALLOWLIST", ["read_file"])
    monkeypatch.setattr(tools_module.settings, "MCP_TOOL_DENYLIST", [])

    assert _is_tool_allowed("read_file") is True
    # Not in the allowlist -> denied, even though it isn't in the denylist either.
    assert _is_tool_allowed("list_directory") is False
    # Denylist wins even over an allowlist entry.
    monkeypatch.setattr(tools_module.settings, "MCP_TOOL_DENYLIST", ["read_file"])
    assert _is_tool_allowed("read_file") is False


class _FakeMCPTool:
    def __init__(self, name):
        self.name = name


def test_load_mcp_tools_caches_discovery_across_calls(monkeypatch):
    """Discovering MCP tools spawns a subprocess per server; it must not be redone on
    every chat turn. The tool list should be cached after the first successful load."""
    import app.rag.tools as tools_module

    tools_module.clear_mcp_tools_cache()

    call_count = {"n": 0}

    class _FakeMultiServerMCPClient:
        def __init__(self, connections):
            self.connections = connections

        async def get_tools(self):
            call_count["n"] += 1
            return [_FakeMCPTool("read_file"), _FakeMCPTool("write_file")]

    fake_client_module = types.ModuleType("langchain_mcp_adapters.client")
    fake_client_module.MultiServerMCPClient = _FakeMultiServerMCPClient
    fake_package = types.ModuleType("langchain_mcp_adapters")
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters", fake_package)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.client", fake_client_module)

    monkeypatch.setattr(tools_module.settings, "MCP_ENABLED", True)
    monkeypatch.setattr(
        tools_module.settings,
        "MCP_SERVERS",
        {"filesystem": {"transport": "stdio", "command": "npx", "args": ["-y", "fake"]}},
    )
    monkeypatch.setattr(tools_module.settings, "MCP_TOOL_ALLOWLIST", ["read_file"])
    monkeypatch.setattr(tools_module.settings, "MCP_TOOL_DENYLIST", [])

    try:
        first = tools_module.load_mcp_tools()
        second = tools_module.load_mcp_tools()

        assert [t.name for t in first] == ["read_file"]  # write_file is filtered out
        assert first == second
        assert call_count["n"] == 1  # discovery only happened once across both calls
    finally:
        tools_module.clear_mcp_tools_cache()
