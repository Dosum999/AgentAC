"""진행 중 초안이 있으면 첫 호출을 도구로 강제한다.

**왜 그래프에 있나.** 되물을 값이 남은 초안(`collecting_draft`)이나 거절된 초안
(`rejected_draft`)이 있으면 그 턴은 조회가 아니라 실행이다. 일반 채널은 supervisor 가
그 판정을 해 주지만(`_bias_to_action_for_pending_draft`), **채널 스코핑 모드(proc·logistics)
는 supervisor 를 건너뛴다**(orchestrator `_use_supervisor`). 그래서 같은 일을 여기서 한다.

실측 사고(PROC 발주 초안): 선택값을 묻는 결과를 받은 뒤 사용자가 "넘어가" 라고 답해도
모델이 도구를 다시 부르지 않고 **같은 질문을 새로 만들어** 냈다. 그때 제시한 품목 분류가
도구가 준 목록에 없는 값이었다(지어낸 증거). 같은 화면이 무한히 반복됐다.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agent_arch.agents.context import (
    set_collecting_draft_context,
    set_image_context,
    set_rejected_draft_context,
)
from agent_arch.agents.graphs.react_graph import (
    _force_tool_reason,
    _pin_tool_for_rejected_draft,
)

pytestmark = pytest.mark.usefixtures("channel_modes")

TOOL = "orders_request_draft"


@pytest.fixture(autouse=True)
def _clean():
    for reset in (lambda: set_collecting_draft_context(None),
                  lambda: set_rejected_draft_context(None),
                  lambda: set_image_context(None)):
        reset()
    yield
    for reset in (lambda: set_collecting_draft_context(None),
                  lambda: set_rejected_draft_context(None),
                  lambda: set_image_context(None)):
        reset()


def test_no_stash_leaves_the_turn_alone():
    """평소에는 강제하지 않는다 — 조회 턴까지 도구를 부르면 안 된다."""
    assert _force_tool_reason({TOOL}) is None


@pytest.mark.parametrize("setter", [set_collecting_draft_context, set_rejected_draft_context])
def test_pending_draft_forces_a_tool_call(setter):
    """수집 중이든 거절됐든 실행 턴이다 — supervisor 의 두 stash 와 같은 범위."""
    setter({"tool_name": TOOL})
    assert _force_tool_reason({TOOL}) is not None


def test_unbound_tool_does_not_force():
    """그 도구가 이 턴에 안 붙었으면 강제하지 않는다 — 엉뚱한 도구를 부르게 된다."""
    set_collecting_draft_context({"tool_name": TOOL})
    assert _force_tool_reason({"orders"}) is None


def test_broken_stash_fails_open():
    """stash 가 깨져 있어도 예외를 올리지 않는다(부가기능)."""
    set_collecting_draft_context({"no_tool_name": 1})
    assert _force_tool_reason({TOOL}) is None


def test_attachment_alone_does_not_force():
    """도면을 올렸다고 발주요청인 것은 아니다 — 첨부만으로는 강제하지 않는다.

    실측(2026-09-30): 첨부 턴을 강제했더니 "이 도면 재질이 뭐야?" 가 묻지도 않은 초안
    질문을 달고 수집 상태까지 남겼다(2/2).
    """
    set_image_context([object()])
    assert _force_tool_reason({TOOL}) is None


def test_reason_names_which_stash_fired():
    """로그로 원인을 가릴 수 있어야 한다 — 어느 조건이 걸렸는지 사유에 남긴다."""
    set_rejected_draft_context({"tool_name": TOOL})
    assert _force_tool_reason({TOOL}).startswith("거절됨:")


class TestRejectedDraftPinsTheTool:
    """거절 턴은 ANY 로 부족해 **지목**한다.

    실측: 거절 뒤 "다시 만들어줘" 턴에 ANY 가 걸렸는데도(로그 `사유=거절됨:…`) 모델이
    도구를 부르지 않고 카드를 글로 흉내 냈다. ANY 는 텍스트가 나오면 막지 못한다 —
    `_retry_any_node` 는 빈 응답일 때만 발동하기 때문이다.
    """

    def test_pins_when_a_draft_was_rejected(self):
        set_rejected_draft_context({"tool_name": TOOL})
        assert _pin_tool_for_rejected_draft({TOOL}) == TOOL

    def test_does_not_pin_without_a_rejection(self):
        """평소에는 지목하지 않는다 — 지목은 그 도구만 부르게 하는 강한 수단이다."""
        assert _pin_tool_for_rejected_draft({TOOL}) is None

    def test_does_not_pin_while_merely_collecting(self):
        """수집 중에는 지목하지 않는다 — 사용자가 다른 것을 물으면 그쪽으로 갈 수 있어야 한다."""
        set_collecting_draft_context({"tool_name": TOOL})
        assert _pin_tool_for_rejected_draft({TOOL}) is None

    def test_does_not_pin_an_unbound_tool(self):
        set_rejected_draft_context({"tool_name": TOOL})
        assert _pin_tool_for_rejected_draft({"orders"}) is None


class TestPinIsOneShot:
    """지목은 거절 직후 **한 턴만**. stash 는 TTL 5분이라 그대로 두면 그 사이 모든 턴이
    같은 도구로 묶여, 사용자가 그만두겠다고 해도 평문으로 답할 길이 없다(실측 2026-09-22).
    소비 표시는 orchestrator 가 붙인다(`rejected_pin_used`)."""

    def test_first_turn_pins(self):
        set_rejected_draft_context({"tool_name": TOOL})
        assert _pin_tool_for_rejected_draft({TOOL}) == TOOL

    def test_after_use_the_model_chooses(self):
        set_rejected_draft_context({"tool_name": TOOL, "pin_used": True})
        assert _pin_tool_for_rejected_draft({TOOL}) is None

    def test_any_force_still_applies_after_the_pin_is_used(self):
        """지목만 꺼진다 — ANY 강제는 남아야 초안 정정이 계속 도구로 간다."""
        set_rejected_draft_context({"tool_name": TOOL, "pin_used": True})
        assert _force_tool_reason({TOOL}) == f"거절됨:{TOOL}"


class TestForcedModeReachesTheProvider:
    """판정만으로는 부족하다 — **프로바이더에 실제로 전달되는지**까지 본다.

    실측 사고: `_start_node` 반환 dict 에 `forced_any_mode` 를 새로 넣었는데 아래쪽 초기화
    줄(`"forced_any_mode": False`)과 **키가 겹쳤다.** dict 리터럴은 뒤가 이기므로 조용히
    False 로 덮였고, 판정 로그는 찍히는데 강제는 안 걸리는 상태로 세 회차를 헤맸다.
    문법 오류도 경고도 없다 — 그래서 값이 끝까지 가는지를 테스트가 봐야 한다.
    """

    @staticmethod
    async def _stream_kwargs(setup, *, force: bool = True) -> dict:
        from unittest.mock import AsyncMock, MagicMock

        from agent_arch.agents.graphs.react_graph import run_react_graph
        from agent_arch.llm.base import StreamEvent

        seen: list = []

        def gen(*_a, **k):
            seen.append(dict(k))

            async def _g():
                yield StreamEvent(type="text", content="텍스트")
                yield StreamEvent(type="done", finish_reason="stop")

            return _g()

        llm = MagicMock()
        llm.stream = gen
        llm.bind_tools = AsyncMock()
        tool = MagicMock()
        tool.name = TOOL
        tool.execute = AsyncMock()
        setup()
        await run_react_graph(llm, "넘어가", [], [tool], AsyncMock(return_value=[]), 3,
                              pending_draft_force=force)
        return seen[0] if seen else {}

    @pytest.mark.asyncio
    async def test_attachment_turn_is_not_forced(self):
        kw = await self._stream_kwargs(lambda: set_image_context([object()]))
        assert kw.get("tool_mode") != "ANY"

    @pytest.mark.asyncio
    async def test_collecting_turn_sends_any_without_pinning(self):
        """수집 중에는 지목하지 않는다 — 다른 질문으로 갈 여지를 남긴다."""
        kw = await self._stream_kwargs(
            lambda: set_collecting_draft_context({"tool_name": TOOL}))
        assert kw.get("tool_mode") == "ANY"
        assert "allowed_tools" not in kw

    @pytest.mark.asyncio
    async def test_rejected_turn_pins_the_tool(self):
        kw = await self._stream_kwargs(
            lambda: set_rejected_draft_context({"tool_name": TOOL}))
        assert kw == {"tool_mode": "ANY", "allowed_tools": [TOOL]}

    @pytest.mark.asyncio
    async def test_ordinary_turn_is_untouched(self):
        kw = await self._stream_kwargs(lambda: None)
        assert kw.get("tool_mode") == "AUTO"

    @pytest.mark.asyncio
    async def test_general_channel_is_not_touched(self):
        """supervisor 가 도는 채널은 그대로 둔다 — 거기선 승격을 supervisor 가 한다.

        일반 채널까지 켜면 중복일 뿐 아니라, 거절 지목이 무관한 질문을 끌고 간다
        (캘린더 초안을 거절한 뒤 "내일 일정 뭐야?" → 생성 도구로 지목).
        """
        kw = await self._stream_kwargs(
            lambda: set_rejected_draft_context({"tool_name": TOOL}), force=False)
        assert kw.get("tool_mode") == "AUTO"
        assert "allowed_tools" not in kw


class TestTheGateIsChannelScoped:
    """강제를 켜는 조건은 환경변수가 아니라 **채널**이다.

    게이트(`orchestrator._pending_draft_force`)를 `not _use_supervisor` 로 두면 안 된다.
    `use_langgraph_supervisor` 는 **기본값이 False** 라, 그 플래그를 세우지 않은 배포에서는
    모든 채널이 강제 대상이 된다. 강제(ANY)까지는 논리가 서지만 **도구 지목**이 함께 켜지는
    것이 문제다 — 캘린더 초안을 거절한 뒤 "내일 일정 뭐야?" 가 생성 도구로 지목된다.
    게다가 `collecting_draft`·`rejected_draft` 는 전자결재도 쓰는 공유 stash 라
    (`eapproval_create_document`) 남의 도메인까지 닿는다.
    """

    @staticmethod
    def _orc(mode: str):
        from agent_arch.agents.orchestrator import AgentOrchestrator

        with (
            patch("agent_arch.agents.orchestrator.LLMFactory.create", return_value=MagicMock()),
            patch("agent_arch.agents.orchestrator.core_tools", return_value=[]),
            patch("agent_arch.extensions.extensions.guide_tool", MagicMock()),
        ):
            return AgentOrchestrator(agent_mode=mode)

    @pytest.mark.parametrize("supervisor", [True, False])
    @pytest.mark.parametrize("mode", ["procurement", "operations"])
    def test_channel_scoped_modes_always_force(self, mode, supervisor):
        """supervisor 를 건너뛰는 채널 — 설정과 무관하게 켜져야 한다."""
        orc = self._orc(mode)
        with patch("agent_arch.agents.orchestrator.settings.use_langgraph_supervisor", supervisor):
            assert orc._pending_draft_force is True

    @pytest.mark.parametrize("supervisor", [True, False])
    @pytest.mark.parametrize("mode", ["agent", "guide"])
    def test_general_channels_never_force(self, mode, supervisor):
        """**supervisor 가 꺼진 배포에서도** 일반 채널은 강제하지 않는다 — 회귀 방지선이다."""
        orc = self._orc(mode)
        with patch("agent_arch.agents.orchestrator.settings.use_langgraph_supervisor", supervisor):
            assert orc._pending_draft_force is False
