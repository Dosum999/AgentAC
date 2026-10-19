"""_stream_react HITL 배선 통합 테스트 (Slice 2).

루프가 승인필요 액션을 confirm=False 초안으로 만들고 pending_action 을 yield → orchestrator
가 (1) conversation_id 확정 후 (2) HITL 그래프에 체크포인트하고 (3) 기존 draft 카드
(workspace.update)를 emit 하는 전체 경로를 검증한다. 실제 발송(confirm=True)은 0회.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_arch.agents.orchestrator import AgentOrchestrator, _RequestContext
from agent_arch.agents.tools.base import ToolResult
from agent_arch.llm.base import StreamEvent, ToolCall


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


def _ctx(user_sub="user-1") -> _RequestContext:
    return _RequestContext(
        user_sub=user_sub,
        user_email="user@example.com",
        user_info={"name": "홍길동", "sub": user_sub},
        langsmith_run_id=None,
        messages=[],
        input="a@b.c 로 메일 보내줘",
        augmented_input="a@b.c 로 메일 보내줘",
        t_start=0.0,
        command_result=None,
        conversation_id=None,
    )


def _llm_one_toolcall(name: str, arguments: dict) -> MagicMock:
    """단일 tool_call 회차만 내는 mock LLM."""
    def _stream(*_args, **_kwargs):
        async def _gen():
            yield StreamEvent(type="tool_calls", tool_calls=[ToolCall(id="tc1", name=name, arguments=arguments)])
            yield StreamEvent(type="done", finish_reason="tool_calls", usage=None)
        return _gen()

    llm = MagicMock()
    llm.stream = _stream
    llm.bind_tools = AsyncMock()
    return llm


def _mail_tool() -> MagicMock:
    """confirm 값을 기록하고 draft view(form 포함)를 반환하는 모의 메일 도구."""
    tool = MagicMock()
    tool.name = "mail_send_message"
    tool.execute = AsyncMock(
        return_value=ToolResult(
            success=True,
            output={
                "view": {
                    "app": "mail",
                    "kind": "draft",
                    "label": "메일 초안",
                    "form": {
                        "kind": "mail",
                        "endpoint": "/workspace/mail/send",
                        "values": {"to": ["a@b.c"], "subject": "s", "body": "b"},
                    },
                    "preview": {"title": "메일 초안", "summary": "a@b.c 에게"},
                }
            },
        )
    )
    return tool


@pytest.mark.asyncio
async def test_stream_react_hitl_checkpoints_and_emits_draft(monkeypatch):
    orch = _orchestrator()
    monkeypatch.setattr("agent_arch.agents.orchestrator.settings.use_langgraph_hitl", True)

    mail = _mail_tool()
    monkeypatch.setattr(orch, "_prepare_tools", AsyncMock(return_value=[mail]))
    orch.llm = _llm_one_toolcall(
        "mail_send_message", {"to": ["a@b.c"], "subject": "s", "body": "b", "confirm": True}
    )
    monkeypatch.setattr(
        "agent_arch.agents.orchestrator.save_logged_in_turn",
        AsyncMock(return_value=("msg-1", "conv-xyz")),
    )
    checkpoint_spy = AsyncMock()
    monkeypatch.setattr(orch, "_checkpoint_pending_action", checkpoint_spy)
    monkeypatch.setattr(orch, "_kickoff_eval_replay", MagicMock())

    out = {}
    events = [e async for e in orch._stream_react(_ctx(), out)]

    # 실제 발송은 confirm=False 초안 1회뿐 — confirm=True 실행 0회
    mail.execute.assert_awaited_once()
    _, kwargs = mail.execute.call_args
    assert kwargs["confirm"] is False

    # HITL 그래프에 conversation_id(=thread_id)로 큐 체크포인트 적재됨(단일액션=큐 길이 1)
    checkpoint_spy.assert_awaited_once()
    conv_id, pending_queue = checkpoint_spy.call_args.args
    assert conv_id == "conv-xyz"
    assert isinstance(pending_queue, list) and len(pending_queue) == 1
    assert pending_queue[0]["tool_name"] == "mail_send_message"
    assert "confirm" not in pending_queue[0]["arguments"]

    # 기존 경로로 draft 카드(workspace.update) emit
    updates = [
        json.loads(e)
        for e in events
        if isinstance(e, str) and e.lstrip().startswith("{") and '"workspace.update"' in e
    ]
    assert updates, "draft 카드 workspace.update 가 emit 되어야 한다"
    artifact = updates[0]["artifact"]
    assert artifact["actionKind"] == "draft"
    assert artifact.get("form")  # url 없는 메일 초안 → 카드 렌더 조건
