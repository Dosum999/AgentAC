"""RemoteMCPTool 어댑터 단위 테스트 (전송 계층 모킹)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_arch.agents.context import set_user_context
from agent_arch.agents.permissions.spec import GateMode, PermissionRequirement
from agent_arch.agents.tools.mcp.remote_tool import (
    RemoteMCPTool,
    _sanitize_parameters,
    _to_tool_result,
)

_PERM = PermissionRequirement(service="mail", mode=GateMode.ACCESS_ONLY)

_INPUT_SCHEMA = {
    "properties": {
        "folder": {"type": "string", "default": "INBOX", "title": "Folder"},
        "limit": {"type": "integer", "default": 20, "title": "Limit"},
    },
    "required": ["folder"],
}


def _make_tool(**overrides) -> RemoteMCPTool:
    kwargs = dict(
        server_url="http://mcp.test/mcp",
        service_token="tok",
        name="mail_list_messages",
        description="테스트 도구",
        input_schema=_INPUT_SCHEMA,
        permission=_PERM,
    )
    kwargs.update(overrides)
    return RemoteMCPTool(**kwargs)


def _call_result(*, is_error=False, structured=None, texts=()):
    content = [SimpleNamespace(text=t) for t in texts]
    return SimpleNamespace(is_error=is_error, structured_content=structured, content=content)


class TestSchemaMapping:
    def test_sanitize_strips_title(self):
        params = _sanitize_parameters(_INPUT_SCHEMA)
        assert "title" not in params["folder"]
        assert params["folder"]["type"] == "string"

    def test_tool_exposes_base_tool_interface(self):
        tool = _make_tool()
        assert tool.name == "mail_list_messages"
        assert tool.required_params == ["folder"]
        assert set(tool.parameters.keys()) == {"folder", "limit"}
        assert tool.permission_required is _PERM

    def test_empty_schema(self):
        tool = _make_tool(input_schema=None)
        assert tool.parameters == {}
        assert tool.required_params == []


class TestResultMapping:
    def test_structured_content_preferred(self):
        result = _to_tool_result("t", _call_result(structured={"count": 1}, texts=['{"x":2}']))
        assert result.success and result.output == {"count": 1}

    def test_text_json_fallback(self):
        result = _to_tool_result("t", _call_result(texts=['{"count": 3}']))
        assert result.success and result.output == {"count": 3}

    def test_text_plain_fallback(self):
        result = _to_tool_result("t", _call_result(texts=["not-json"]))
        assert result.success and result.output == "not-json"

    def test_error_maps_to_failure(self):
        result = _to_tool_result("t", _call_result(is_error=True, texts=["메일박스 없음"]))
        assert not result.success and "메일박스 없음" in result.error

    def test_error_without_text(self):
        result = _to_tool_result("my_tool", _call_result(is_error=True))
        assert not result.success and "my_tool" in result.error


class TestExecution:
    @pytest.mark.asyncio
    async def test_demo_execute_returns_mock(self):
        tool = _make_tool()
        result = await tool._demo_execute(folder="INBOX")
        assert result.success and "[DEMO]" in result.output["message"]

    @pytest.mark.asyncio
    async def test_real_execute_without_email_errors(self):
        set_user_context({})
        tool = _make_tool()
        result = await tool._real_execute(folder="INBOX")
        assert not result.success and "이메일" in result.error

    @pytest.mark.asyncio
    async def test_real_execute_passes_user_context_headers(self, monkeypatch):
        set_user_context({
            "sub": "u1", "email": "user@example.com",
            "session_id": "sid-1", "company": "(주)회사",
        })
        tool = _make_tool()
        seen = {}

        async def fake_call(arguments, user_headers):
            seen["arguments"] = arguments
            seen["headers"] = user_headers
            return _call_result(structured={"count": 0, "messages": []})

        monkeypatch.setattr(tool, "_call_remote", fake_call)
        result = await tool._real_execute(folder="INBOX", limit=5)
        assert result.success
        assert seen["headers"]["X-User-Email"] == "user@example.com"
        assert seen["headers"]["X-Session-Id"] == "sid-1"
        # 비ASCII 회사명은 percent-encoding (헤더 ascii 제약)
        assert "%" in seen["headers"]["X-Company"]
        assert seen["arguments"] == {"folder": "INBOX", "limit": 5}

    @pytest.mark.asyncio
    async def test_real_execute_transport_failure_is_tool_error(self, monkeypatch):
        set_user_context({"sub": "u1", "email": "user@example.com"})
        tool = _make_tool()

        async def boom(arguments, user_headers):
            raise ConnectionError("connection refused")

        monkeypatch.setattr(tool, "_call_remote", boom)
        result = await tool._real_execute(folder="INBOX")
        assert not result.success and "MCP 서버 호출 실패" in result.error

    @pytest.mark.asyncio
    async def test_transport_failure_is_recorded_as_mcp_tool_event(self, monkeypatch):
        """호출 실패는 WARNING 이지만 오류 이벤트(mcp_tool)로 남는다 — 서버 키와 함께."""
        import logging

        from agent_arch.services.monitoring import error_events as ee

        monkeypatch.setattr(ee, "_buffer", ee._EventBuffer())
        monkeypatch.setattr("agent_arch.core.config.settings.enable_error_events", True, raising=False)
        lg = logging.getLogger("agent_arch.agents.tools.mcp.remote_tool")
        handler = ee.ErrorEventHandler()
        lg.addHandler(handler)
        try:
            set_user_context({"sub": "u1", "email": "user@example.com"})
            tool = _make_tool(server_name="mailcow")

            async def boom(arguments, user_headers):
                raise ConnectionError("connection refused")

            monkeypatch.setattr(tool, "_call_remote", boom)
            await tool._real_execute(folder="INBOX")
        finally:
            lg.removeHandler(handler)
        (event,) = ee._buffer._items.values()
        assert (event.category, event.mcp_server, event.exc_type, event.level) == (
            "mcp_tool", "mailcow", "ConnectionError", "WARNING",
        )


def test_failure_with_data_maps_to_failure_keeping_data():
    """MCP 는 데이터를 실은 실패를 정상 응답(success:false)으로 보낸다 — 실패로 세우되 데이터는 지킨다."""
    result = _to_tool_result("t", _call_result(structured={
        "success": False, "error": "3명입니다", "ambiguous": True,
        "candidates": [{"display": "김가나", "handle": "@kgn"}],
    }))
    assert result.success is False
    assert result.error == "3명입니다"
    assert result.output == {"ambiguous": True, "candidates": [{"display": "김가나", "handle": "@kgn"}]}
