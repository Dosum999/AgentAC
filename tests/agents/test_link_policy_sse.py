"""link_policy 이벤트는 정화기에 먹이는 **내부 신호**다 — SSE 로 나가면 안 된다.

프론트는 알려진 type 만 처리하므로 낯선 이벤트가 나가면 화면에 정체불명 진행 표시가 뜨거나
무시된다. 그리고 그 정책으로 화면 토큰이 실제로 걸러지는지(지어낸 가이드 링크 제거)도 같이 본다.
"""
from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_arch.agents.orchestrator import AgentOrchestrator, _RequestContext
from agent_arch.agents.react.link_policy import TurnLinkPolicy
from agent_arch.agents.react.types import LoopResult, ProgressEvent

_FORM = "https://example.com/main/support/inquiry/write"
_FAKE = "https://example.com/user-guide/agent-deep-dive"


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


def _ctx() -> _RequestContext:
    return _RequestContext(
        user_sub=None, user_email=None, user_info={}, langsmith_run_id=None, messages=[],
        input="에이전트 사용방법 안내해줘", augmented_input="에이전트 사용방법 안내해줘",
        t_start=0.0, command_result=None, conversation_id=None,
    )


@pytest.mark.asyncio
async def test_link_policy_event_is_consumed_not_forwarded(monkeypatch):
    orch = _orchestrator()
    monkeypatch.setattr(orch, "_prepare_tools", AsyncMock(return_value=[]))
    monkeypatch.setattr(orch, "_kickoff_eval_replay", MagicMock())

    policy = TurnLinkPolicy()
    policy.observe(
        "search_user_guide_kb",
        json.dumps([{"chunks": [{"content": f"회원 문의 폼( {_FORM} )"}]}], ensure_ascii=False),
    )
    answer = (f"문의는 [문의 폼]({_FORM}) 에서 가능합니다.\n\n"
              f"더 자세한 내용은 아래 가이드를 참고해 주세요. [Agent 사용 심층 가이드]({_FAKE})\n")

    async def _events(*_a, **_k):
        yield ProgressEvent("thinking", content="생각 중")
        yield ProgressEvent("link_policy", content=policy.dumps())
        for i in range(0, len(answer), 7):                    # 토큰이 잘게 쪼개져 와도
            yield ProgressEvent("token", content=answer[i:i + 7])
        yield LoopResult(content=answer, finish_reason="stop")

    # 테스트 기본값은 두 LangGraph 플래그가 OFF(conftest) → 기본 이벤트원은 react_loop_events 다.
    monkeypatch.setattr("agent_arch.agents.orchestrator.react_loop_events", _events)

    chunks = [c async for c in orch._stream_react(_ctx(), None)]
    screen = "".join(c for c in chunks if not c.lstrip().startswith("{"))
    events = [json.loads(c) for c in chunks if c.lstrip().startswith("{")]

    assert "link_policy" not in [e.get("type") for e in events]
    assert _FAKE not in screen and "가이드를 참고해 주세요" not in screen
    assert _FORM in screen
