"""agent_arch.agents.react.loop 생성 중단(stop) 체크포인트 단위 테스트.

협조적 취소(ContextVar 의 asyncio.Event) 가 set 되면 react_loop_events 가
finish_reason="stopped" LoopResult 로 즉시 종료하는지, 그리고 도구 실행 직전
체크포인트가 부수효과 도구 실행을 막는지 검증한다.

중단된 턴은 저장하지 않으므로 LoopResult.content 는 비어 있어야 한다.
"""
import asyncio
import itertools

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_arch.agents.react.loop import react_loop_events
from agent_arch.agents.react.types import LoopResult, ProgressEvent
from agent_arch.agents.cancellation import set_cancel_event
from agent_arch.llm.base import Message, StreamEvent, ToolCall


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _text_gen(content: str = "응답", finish_reason: str = "stop"):
    """텍스트 답변 회차(text → done) async-gen 팩토리."""
    async def _gen(*_args, **_kwargs):
        if content:
            yield StreamEvent(type="text", content=content)
        yield StreamEvent(type="done", finish_reason=finish_reason, usage=None)
    return _gen


def _llm(*gens) -> MagicMock:
    """stream() 호출마다 다음 async-gen 을 쓰고, 호출 횟수를 _calls 로 추적하는 mock LLM."""
    queue = list(gens)
    calls = {"n": 0}

    def _next(*args, **kwargs):
        calls["n"] += 1
        return queue.pop(0)(*args, **kwargs)

    llm = MagicMock()
    llm.stream = _next
    llm.bind_tools = AsyncMock()
    llm._calls = calls
    return llm


async def _collect(llm, tools=None, input="질문", max_iterations=5):
    """react_loop_events 를 소비해 (progress, results) 로 분리한다."""
    if tools is None:
        tools = []
    events = [
        e async for e in react_loop_events(
            llm, input, [Message(role="user", content=input)],
            tools, AsyncMock(return_value=[]), max_iterations,
        )
    ]
    progress = [e for e in events if isinstance(e, ProgressEvent)]
    results = [e for e in events if isinstance(e, LoopResult)]
    return progress, results


@pytest.fixture(autouse=True)
def _clear_cancel():
    """각 테스트 후 ContextVar 의 취소 Event 를 해제해 누수를 막는다."""
    yield
    set_cancel_event(None)


# ── 테스트 ───────────────────────────────────────────────────────────────────

class TestReactLoopStop:
    """생성 중단 체크포인트 검증."""

    @pytest.mark.asyncio
    async def test_iteration_경계_stop_시_즉시_종료(self):
        """루프 진입 전 취소가 set 되어 있으면 LLM 호출 없이 stopped 로 끝난다(체크포인트 a)."""
        ev = asyncio.Event()
        ev.set()
        set_cancel_event(ev)

        llm = _llm(_text_gen("쓰이지 않음"))
        _, results = await _collect(llm)

        assert len(results) == 1
        assert results[0].finish_reason == "stopped"
        assert results[0].content == ""
        assert llm._calls["n"] == 0, "취소 상태면 llm.stream 이 호출되지 않아야 한다"

    @pytest.mark.asyncio
    async def test_token_회차_중간_stop_시_부분만_흘리고_stopped(self):
        """토큰을 흘리다 취소가 set 되면 이후 토큰은 멈추고 stopped 로 종료한다(체크포인트 b)."""
        ev = asyncio.Event()
        set_cancel_event(ev)

        async def _gen(*_a, **_k):
            yield StreamEvent(type="text", content="부분")
            ev.set()  # 첫 토큰 직후 사용자가 중단
            yield StreamEvent(type="text", content="더보기")
            yield StreamEvent(type="done", finish_reason="stop", usage=None)

        llm = MagicMock()
        llm.stream = _gen
        llm.bind_tools = AsyncMock()

        progress, results = await _collect(llm)

        tokens = [e.content for e in progress if e.event_type == "token"]
        assert tokens == ["부분"], "중단 후 토큰('더보기')은 흘리지 않아야 한다"
        assert len(results) == 1
        assert results[0].finish_reason == "stopped"
        assert results[0].content == ""

    @pytest.mark.asyncio
    async def test_stop_미설정시_정상_종료(self):
        """취소 Event 가 없으면 회귀 없이 정상 finish_reason 으로 끝난다."""
        set_cancel_event(None)
        llm = _llm(_text_gen("정상 답변"))
        _, results = await _collect(llm)

        assert len(results) == 1
        assert results[0].finish_reason == "stop"
        assert results[0].content == "정상 답변"

    @pytest.mark.asyncio
    async def test_stop_시_도구실행_건너뜀(self):
        """도구 호출 회차에서 도구 실행 직전 취소가 잡히면 execute_tools 가 호출되지 않는다(체크포인트 c).

        is_cancelled 호출 순서: (a)iteration 경계 → (b)스트림 이벤트 2회 → (b)스트림 종료 후 →
        (c)도구 실행 직전. 마지막(c)만 True 가 되도록 시퀀스를 주입한다.
        """
        async def _gen(*_a, **_k):
            yield StreamEvent(type="tool_calls", tool_calls=[ToolCall(id="t1", name="mytool", arguments={})])
            yield StreamEvent(type="done", finish_reason="tool_calls", usage=None)

        llm = MagicMock()
        llm.stream = _gen
        llm.bind_tools = AsyncMock()

        seq = itertools.chain([False, False, False, False], itertools.repeat(True))
        exec_spy = AsyncMock(return_value=[])
        with patch("agent_arch.agents.react.loop.is_cancelled", side_effect=lambda: next(seq)), \
             patch("agent_arch.agents.react.loop.execute_tools", exec_spy):
            _, results = await _collect(llm)

        exec_spy.assert_not_awaited()
        assert len(results) == 1
        assert results[0].finish_reason == "stopped"
