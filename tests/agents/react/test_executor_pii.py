"""executor·HITL resume — 가명화 자리표시가 인자에 남으면 실행하지 않고 모델에 되돌린다. 인자 로그는 모양만."""
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_arch.agents.react.executor import execute_tools, unresolved_placeholder_error
from agent_arch.agents.tools.base import ToolResult
from agent_arch.llm.base import ToolCall


def _tool(name="mail_send"):
    t = MagicMock()
    t.name = name
    t.execute = AsyncMock(return_value=ToolResult(success=True, output={"ok": True}))
    return t


def test_unresolved_placeholder_error_detects_variants():
    assert unresolved_placeholder_error("t", {"to": "user@example.com"}) is None
    msg = unresolved_placeholder_error("mail_send", {"to": ["[[EMAIL_0a0b0c0d]]@example.com"], "cc": {"x": "PERSON_deadbeef"}})
    assert msg.startswith("[ERROR] `mail_send` 인자에 자리표시 토큰") and "EMAIL_0a0b0c0d" in msg and "PERSON_deadbeef" in msg


@pytest.mark.asyncio
async def test_execute_tools_blocks_leftover_tokens(caplog):
    tool = _tool()
    tc = ToolCall(id="1", name="mail_send", arguments={"to": "[[EMAIL_0a0b0c0d]]@example.com", "body": "안녕"})
    with caplog.at_level(logging.DEBUG, logger="agent_arch.agents.react.executor"):
        [(ret_tc, content)] = await execute_tools([tc], [tool])
    assert ret_tc is tc and content.startswith("[ERROR]") and "자리표시" in content
    tool.execute.assert_not_called()
    # 인자 로그엔 값이 아니라 모양만
    dbg = [r.getMessage() for r in caplog.records if "args=" in r.getMessage()]
    assert dbg and "<str:" in dbg[0] and "EMAIL_0a0b0c0d" not in dbg[0] and "안녕" not in dbg[0]


@pytest.mark.asyncio
async def test_execute_tools_runs_when_restored():
    tool = _tool()
    tc = ToolCall(id="1", name="mail_send", arguments={"to": "user@example.com"})
    [(_, content)] = await execute_tools([tc], [tool])
    tool.execute.assert_awaited_once_with(to="user@example.com")
    assert "ok" in content


@pytest.mark.asyncio
async def test_hitl_execute_one_blocks_leftover_tokens(monkeypatch):
    from agent_arch.agents.hitl import graph as graph_mod

    executed = []

    class _Tool:
        async def execute(self, **kwargs):
            executed.append(kwargs)
            return ToolResult(success=True, output={"ok": True})

    monkeypatch.setattr(graph_mod.ToolRegistry, "get", staticmethod(lambda name: _Tool()))
    r = await graph_mod._execute_one("mail_send_message", {"to": "[[EMAIL_0a0b0c0d]]@example.com"}, {"confirm": True})
    assert r["success"] is False and "자리표시" in r["error"] and executed == []
    r = await graph_mod._execute_one("mail_send_message", {"to": "user@example.com"}, {"confirm": True})
    assert r["success"] is True and executed == [{"confirm": True, "to": "user@example.com"}]


@pytest.mark.asyncio
async def test_execute_tools_schedules_harvest_only_when_guard_on(monkeypatch):
    """도구 결과의 이름·이메일 수확은 가드 모드가 켜져 있을 때만, 백그라운드로 예약된다."""
    import asyncio
    from agent_arch.agents.context import set_conversation_id
    from agent_arch.core.config import settings
    from agent_arch.llm.pii import sources

    captured = []

    async def fake_harvest(conv, content, company=None):
        captured.append((conv, content, company))

    monkeypatch.setattr(sources, "harvest_tool_result", fake_harvest)
    set_conversation_id("conv-h")
    from agent_arch.agents.context import set_user_context
    set_user_context({"sub": "u", "company": "(주)회사"})
    tool = _tool("search_colleagues")
    tool.execute = AsyncMock(return_value=ToolResult(success=True, output={"colleagues": [{"name": "김보라", "email": "user@example.com"}]}))
    tc = ToolCall(id="1", name="search_colleagues", arguments={"query": "김보라"})
    await execute_tools([tc], [tool])
    await asyncio.sleep(0)
    assert captured == []                                              # 기본 off
    monkeypatch.setattr(settings, "pii_guard_mode", "shadow", raising=False)
    await execute_tools([tc], [tool])
    await asyncio.sleep(0)
    assert len(captured) == 1 and captured[0][0] == "conv-h" and "user@example.com" in captured[0][1] and captured[0][2] == "(주)회사"
    set_conversation_id(None)
    set_user_context({})
