"""orchestrator._run_react_continuation 단위 테스트 (HITL 재진입).

continuation 한 턴이 다음 승인필요 액션(pending_queue)을 회수하거나, 후속이 없으면
마무리 텍스트만 반환하는지 검증. LLM/도구/히스토리는 모킹.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_arch.agents.orchestrator import AgentOrchestrator
from agent_arch.agents.tools.base import ToolResult
from agent_arch.llm.base import Message, StreamEvent, ToolCall


def _orchestrator() -> AgentOrchestrator:
    patches = [
        patch("agent_arch.agents.orchestrator.LLMFactory.create", return_value=MagicMock()),
        patch("agent_arch.agents.orchestrator.core_tools", return_value=[]),
        patch("agent_arch.extensions.extensions.guide_tool", MagicMock()),
    ]
    for p in patches:
        p.start()
    try:
        return AgentOrchestrator(agent_mode="default")
    finally:
        for p in patches:
            p.stop()


def _llm_toolcall(name, args):
    def _stream(*_a, **_k):
        async def _gen():
            yield StreamEvent(type="tool_calls", tool_calls=[ToolCall(id="tc1", name=name, arguments=args)])
            yield StreamEvent(type="done", finish_reason="tool_calls", usage=None)
        return _gen()
    llm = MagicMock()
    llm.stream = _stream
    llm.bind_tools = AsyncMock()
    llm.stateless = False
    return llm


def _llm_text(text):
    def _stream(*_a, **_k):
        async def _gen():
            yield StreamEvent(type="text", content=text)
            yield StreamEvent(type="done", finish_reason="stop", usage=None)
        return _gen()
    llm = MagicMock()
    llm.stream = _stream
    llm.bind_tools = AsyncMock()
    llm.stateless = False
    return llm


def _mail_tool():
    t = MagicMock()
    t.name = "mail_send_message"
    t.execute = AsyncMock(return_value=ToolResult(
        success=True,
        output={"view": {"app": "mail", "kind": "draft", "label": "메일 초안",
                          "form": {"values": {"to": ["x"]}}}},
    ))
    return t


def _wire(orch, monkeypatch, tools):
    monkeypatch.setattr(orch, "_build_messages", AsyncMock(return_value=[Message(role="user", content="continue")]))
    monkeypatch.setattr(orch, "_prepare_tools", AsyncMock(return_value=tools))
    monkeypatch.setattr("agent_arch.agents.orchestrator.get_profile", AsyncMock(return_value=None))
    monkeypatch.setattr("agent_arch.agents.orchestrator.settings.use_langgraph_hitl", True)


@pytest.mark.asyncio
async def test_continuation_yields_pending_queue(monkeypatch):
    orch = _orchestrator()
    orch.llm = _llm_toolcall("mail_send_message", {"to": ["x"], "subject": "s", "body": "b", "confirm": True})
    _wire(orch, monkeypatch, [_mail_tool()])

    out = await orch._run_react_continuation("conv-x", {"sub": "u1"})
    assert out["pending_queue"]
    assert out["pending_queue"][0]["tool_name"] == "mail_send_message"


@pytest.mark.asyncio
async def test_continuation_text_only_no_pending(monkeypatch):
    orch = _orchestrator()
    orch.llm = _llm_text("다 처리했어요.")
    _wire(orch, monkeypatch, [])

    out = await orch._run_react_continuation("conv-x", {"sub": "u1"})
    assert not out["pending_queue"]
    assert out["content"] == "다 처리했어요."


@pytest.mark.asyncio
async def test_continuation_stateless_noop(monkeypatch):
    orch = _orchestrator()
    orch.llm.stateless = True
    out = await orch._run_react_continuation("conv-x", {"sub": "u1"})
    assert out == {"content": "", "pending_action": None, "pending_queue": None}
