"""_prepare_tools 가 데이터(ReAct) 경로에서 가이드 KB **본문을 자동주입하지 않음**을 검증한다.

가이드 KB 검색/주입을 데이터 경로에서 통째로 제거했다(f54e1fb).
_prepare_tools 는 도구만 계획·bind 하고 messages[0](system 프롬프트)을 건드리지 않는다.
(근거: reports/guide_kb_action_grounding_fix.md)

※ guide/agent 통합 1단계에서 GuideKBTool 이 CORE **도구**로 노출됐지만, 이 계약과
충돌하지 않는다 — 막는 것은 "문서 본문의 프롬프트 주입"이지 "검색 도구의 bind"가
아니다. 이 테스트는 그 재유입 방지 회귀 가드로 계속 유효하다.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_arch.agents.orchestrator import AgentOrchestrator
from agent_arch.llm.base import Message


def _orchestrator() -> AgentOrchestrator:
    """__init__ 을 우회해 최소 구성의 오케스트레이터를 만든다."""
    orc = AgentOrchestrator.__new__(AgentOrchestrator)
    orc.agent_mode = "default"
    orc.llm_provider = "gemini"
    orc.llm = MagicMock(stateless=False, supports_function_calling=True)
    orc.llm.bind_tools = AsyncMock()
    orc.static_tools = []
    orc.guide_tool = MagicMock()
    return orc


@pytest.mark.asyncio
async def test_prepare_tools_does_not_inject_kb_into_system_message():
    """_prepare_tools 는 system 메시지를 KB 로 덮어쓰지 않고, 계획된 도구 리스트만 반환한다."""
    orc = _orchestrator()
    orc._sticky_prev_turn_tools = MagicMock(return_value=[])

    original_system = Message(role="system", content="원본 시스템 프롬프트")
    messages = [original_system, Message(role="user", content="근태 수정해줘")]

    planned = ["sentinel_tool"]
    with patch("agent_arch.agents.orchestrator.plan_tools", return_value=planned):
        result = await orc._prepare_tools(messages, profile=None)

    # 반환은 단일 리스트(예전 (allowed_tools, kb_docs) 튜플이 아님)
    assert result == planned
    # 계획된 도구가 그대로 bind 됨
    orc.llm.bind_tools.assert_awaited_once_with(planned)
    # system 메시지가 KB 주입으로 교체되지 않음 — 동일 객체/내용 유지
    assert messages[0] is original_system
    assert messages[0].content == "원본 시스템 프롬프트"
