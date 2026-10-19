"""전용 채널(MES 앱·SCMS)에서 장기 메모리가 채널 계약을 깨지 않는지 검증한다.

두 가지가 어긋나 있었다:
  ① 기억 로드아웃(`_memory_hint_tools`)이 회상 별칭이 못 박는 **다른 도메인** 도구를
     채널 안으로 들여보냈다 — 프롬프트는 그 도메인을 모르는데 도구만 들어온다.
  ② 회상 안정 블록의 도구 안내가 그대로 남아, 바인딩되지도 않은 메모리 도구를 부르라고
     지시했다(가이드 모드만 걷어내고 있었다).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_arch.agents.orchestrator import _MEMORY_TOOLLESS_MODES, AgentOrchestrator
from agent_arch.services.memory_v2 import loadout as L
from agent_arch.services.memory_v2 import recall as R

pytestmark = pytest.mark.usefixtures("channel_modes")


def _recall(kind: str) -> SimpleNamespace:
    return SimpleNamespace(aliases=[{"payload": {"canonical_kind": kind, "canonical_id": "x"}}])


@pytest.fixture
def recalled(monkeypatch):
    """회상 별칭이 talk 도메인을 못 박은 상태로 만든다."""
    import agent_arch.agents.context as C

    monkeypatch.setattr(C, "get_memory_recall", lambda: _recall("mattermost_channel"))
    monkeypatch.setattr(
        L, "select_tools_for_domains",
        lambda *a, **k: [SimpleNamespace(name="mattermost_skill_find_channels")],
    )


@pytest.mark.parametrize("mode", ["operations", "procurement"])
def test_channel_modes_get_no_memory_loadout(recalled, mode):
    """전용 채널은 기억 별칭이 가리키는 타 도메인 도구를 들지 않는다."""
    orc = SimpleNamespace(agent_mode=mode)

    assert AgentOrchestrator._memory_hint_tools(orc) == []


def test_general_chat_still_gets_memory_loadout(recalled):
    """일반 채팅은 종전대로 들려 준다 — 이 가드가 로드아웃 자체를 죽이면 안 된다."""
    orc = SimpleNamespace(agent_mode="agent")

    names = [t.name for t in AgentOrchestrator._memory_hint_tools(orc)]

    assert names == ["mattermost_skill_find_channels"]


def test_toolless_modes_cover_guide():
    """도구 안내를 빼는 모드 — 코어에선 가이드다(전용 채널 모드는 확장이 정한다)."""
    assert "guide" in _MEMORY_TOOLLESS_MODES


def test_stripped_block_has_no_dead_tool_instruction():
    """전용 채널에 주는 안정 블록엔 부를 수 없는 도구 지시가 없어야 한다."""
    block = R.format_system_block("## 에이전트\n- 담당: MES", [], [{"content": "표 쓰지 않는다"}])

    stripped = R.strip_tool_guidance(block)

    assert "memory_search" not in stripped and "find_memory_tools" not in stripped
    assert "  - 표 쓰지 않는다" in stripped
