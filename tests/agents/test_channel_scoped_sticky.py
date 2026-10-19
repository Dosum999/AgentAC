"""MES 앱 전용 도구가 채널 밖으로 새지 않는다 — sticky 복원 가드.

관문(find_mespoc_tools)은 전역 OFF 라 일반 채팅에서 발견 경로가 없다. 그러나 sticky
복원은 "직전 턴에 호출된 이름"을 ToolRegistry/llm_tools 캐시에서 되살리므로, 같은
대화가 일반 `/agent/*` 로 이어지면 채널 밖에서 되살아날 수 있었다. 그래서 이 턴에
관문이 바인딩돼 있을 때(= MES 앱 채팅·음성 채널일 때)만 복원한다.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_arch.agents.orchestrator import AgentOrchestrator
from agent_arch.agents import orchestrator
from agent_arch.agents.tools.registry import ToolRegistry
from agent_arch.llm.base import Message


@pytest.fixture
def mespoc_tool(monkeypatch):
    """직전 턴에 호출된 MES 앱 전용 도구를 resolve 되게 만든다."""
    fake = MagicMock()
    fake.name = "mespoc_service_list_work_orders"
    # 채널 전용 도구 접두사 → 관문. 코어 표는 비어 있고 도메인이 채운다.
    monkeypatch.setitem(orchestrator._CHANNEL_SCOPED_PREFIXES, "mespoc_", "find_mespoc_tools")
    monkeypatch.setattr(
        ToolRegistry, "get", staticmethod(lambda name: fake if name == fake.name else None)
    )
    return fake


def _prev_turn(name: str) -> list:
    return [Message(role="tool", name=name, content="{}", tool_call_id="t1")]


def test_not_restored_outside_the_mes_channel(mespoc_tool):
    """관문이 안 붙은 턴(일반 채팅)에서는 되살리지 않는다."""
    restored = AgentOrchestrator._sticky_prev_turn_tools(
        _prev_turn(mespoc_tool.name), {"calculator", "find_messenger_tools"}
    )

    assert restored == []


def test_restored_inside_the_mes_channel(mespoc_tool):
    """관문이 붙은 턴(MES 앱 채팅·음성)에서는 종전처럼 후속턴 연속성을 지킨다."""
    restored = AgentOrchestrator._sticky_prev_turn_tools(
        _prev_turn(mespoc_tool.name), {"calculator", "find_mespoc_tools"}
    )

    assert restored == [mespoc_tool]


def test_other_domains_unaffected(monkeypatch):
    """채널 전용이 아닌 도메인은 가드에 걸리지 않는다 (회귀 방지)."""
    fake = MagicMock()
    fake.name = "hr_service_list_logs"
    monkeypatch.setattr(
        ToolRegistry, "get", staticmethod(lambda name: fake if name == fake.name else None)
    )

    restored = AgentOrchestrator._sticky_prev_turn_tools(_prev_turn(fake.name), set())

    assert restored == [fake]
