"""agent_arch.agents.react.guide_loop 생성 중단(stop) 체크포인트 단위 테스트.

guide 모드 루프(guide_loop_events)도 REACT 와 동일하게 협조적 취소(ContextVar Event)에
반응해 finish_reason="stopped" 로 즉시 종료하는지 검증한다. 중단 턴은 저장되지 않으므로
LoopResult.content 는 비어 있어야 한다.
"""
import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_arch.agents.react.guide_loop import guide_loop_events
from agent_arch.agents.react.types import LoopResult, ProgressEvent
from agent_arch.agents.cancellation import set_cancel_event
from agent_arch.llm.base import Message, StreamEvent


def _text_gen(content: str = "응답", finish_reason: str = "stop"):
    async def _gen(*_a, **_k):
        if content:
            yield StreamEvent(type="text", content=content)
        yield StreamEvent(type="done", finish_reason=finish_reason, usage=None)
    return _gen


def _llm(gen) -> MagicMock:
    calls = {"n": 0}

    def _next(*a, **k):
        calls["n"] += 1
        return gen(*a, **k)

    llm = MagicMock()
    llm.stream = _next
    llm.bind_tools = AsyncMock()
    llm._calls = calls
    return llm


async def _collect(llm):
    tool = MagicMock()
    tool.name = "guide_kb_search"
    events = [
        e async for e in guide_loop_events(
            llm, "질문", [Message(role="user", content="질문")], tool,
        )
    ]
    progress = [e for e in events if isinstance(e, ProgressEvent)]
    results = [e for e in events if isinstance(e, LoopResult)]
    return progress, results


@pytest.fixture(autouse=True)
def _clear_cancel():
    yield
    set_cancel_event(None)


class TestGuideLoopStop:
    """guide 모드 중단 검증."""

    @pytest.mark.asyncio
    async def test_iteration_경계_stop_시_즉시_종료(self):
        """루프 진입 전 취소면 LLM 호출 없이 stopped 로 끝난다."""
        ev = asyncio.Event()
        ev.set()
        set_cancel_event(ev)

        llm = _llm(_text_gen("쓰이지 않음"))
        _, results = await _collect(llm)

        assert len(results) == 1
        assert results[0].finish_reason == "stopped"
        assert results[0].content == ""
        assert llm._calls["n"] == 0

    @pytest.mark.asyncio
    async def test_token_중간_stop_시_부분만_흘리고_stopped(self):
        """토큰을 흘리다 취소되면 이후 토큰은 멈추고 stopped 로 종료한다."""
        ev = asyncio.Event()
        set_cancel_event(ev)

        async def _gen(*_a, **_k):
            yield StreamEvent(type="text", content="부분")
            ev.set()
            yield StreamEvent(type="text", content="더보기")
            yield StreamEvent(type="done", finish_reason="stop", usage=None)

        llm = MagicMock()
        llm.stream = _gen
        llm.bind_tools = AsyncMock()

        progress, results = await _collect(llm)

        tokens = [e.content for e in progress if e.event_type == "token"]
        assert tokens == ["부분"]
        assert len(results) == 1
        assert results[0].finish_reason == "stopped"
        assert results[0].content == ""

    @pytest.mark.asyncio
    async def test_stop_미설정시_정상_종료(self):
        """취소 Event 가 없으면 회귀 없이 정상 종료한다."""
        set_cancel_event(None)
        llm = _llm(_text_gen("정상 가이드 답변"))
        _, results = await _collect(llm)

        assert len(results) == 1
        assert results[0].finish_reason == "stop"
        assert results[0].content == "정상 가이드 답변"
