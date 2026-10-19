"""_stream_react 생성 중단(stop) 배선 테스트 (B: 부분답변 PG 저장).

react_loop_events 가 finish_reason="stopped" 를 yield 하면 orchestrator 가
부분답변을 status_code=499 + save_to_redis=False 로 저장하고(피드백/재시도 표시용,
LLM 맥락 제외), _out["stopped"]=True 와 message_id 를 세팅하는지 검증한다.
eval replay 는 건너뛴다.
"""
from __future__ import annotations

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_arch.agents.orchestrator import AgentOrchestrator, _RequestContext
from agent_arch.agents.cancellation import set_cancel_event
from agent_arch.llm.base import StreamEvent


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
        input="긴 답변이 필요한 질문",
        augmented_input="긴 답변이 필요한 질문",
        t_start=0.0,
        command_result=None,
        conversation_id=None,
    )


@pytest.fixture(autouse=True)
def _clear_cancel():
    yield
    set_cancel_event(None)


@pytest.mark.asyncio
async def test_stopped_턴_PG저장_redis제외_및_out_플래그(monkeypatch):
    """중단 시: 부분답변을 status_code=499 + save_to_redis=False 로 저장 + _out 세팅."""
    orch = _orchestrator()
    monkeypatch.setattr(orch, "_prepare_tools", AsyncMock(return_value=[]))

    ev = asyncio.Event()
    set_cancel_event(ev)

    async def _stream(*_a, **_k):
        yield StreamEvent(type="text", content="부분답변")
        ev.set()  # 첫 토큰 직후 사용자가 중단
        yield StreamEvent(type="text", content="더보기")
        yield StreamEvent(type="done", finish_reason="stop", usage=None)

    llm = MagicMock()
    llm.stream = _stream
    llm.bind_tools = AsyncMock()
    orch.llm = llm

    save_spy = AsyncMock(return_value=("msg-1", "conv-1"))
    monkeypatch.setattr("agent_arch.agents.orchestrator.save_logged_in_turn", save_spy)
    eval_spy = MagicMock()
    monkeypatch.setattr(orch, "_kickoff_eval_replay", eval_spy)

    out: dict = {}
    _ = [e async for e in orch._stream_react(_ctx(), out)]

    save_spy.assert_awaited_once()
    kwargs = save_spy.await_args.kwargs
    assert kwargs.get("status_code") == 499, "중단 턴은 status_code=499 로 마킹"
    assert kwargs.get("save_to_redis") is False, "중단 턴은 Redis(맥락) 제외"
    assert out.get("stopped") is True
    assert out.get("message_id") == "msg-1", "피드백용 message_id 가 _out 에 실려야 함"
    eval_spy.assert_not_called()


@pytest.mark.asyncio
async def test_stopped_부분텍스트_없으면_저장_생략(monkeypatch):
    """토큰이 흐르기 전(iteration 경계)에 중단되면 저장할 부분답변이 없어 save 생략."""
    orch = _orchestrator()
    monkeypatch.setattr(orch, "_prepare_tools", AsyncMock(return_value=[]))
    orch.llm = MagicMock()

    save_spy = AsyncMock(return_value=("msg-1", "conv-1"))
    monkeypatch.setattr("agent_arch.agents.orchestrator.save_logged_in_turn", save_spy)
    monkeypatch.setattr(orch, "_kickoff_eval_replay", MagicMock())

    ev = asyncio.Event()
    ev.set()  # 진입 전 중단 → 토큰 0개
    set_cancel_event(ev)

    out: dict = {}
    _ = [e async for e in orch._stream_react(_ctx(), out)]

    save_spy.assert_not_awaited()
    assert out.get("stopped") is True
