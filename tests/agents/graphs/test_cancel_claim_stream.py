"""실제 ReAct 스트림 경계에서 초안 취소 완료 환각이 토큰으로 새지 않는지 검증한다."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_arch.agents.cancellation import set_cancel_event
from agent_arch.agents.graphs.react_graph import stream_react_graph
from agent_arch.agents.hitl.answer_guard import _FALLBACK
from agent_arch.agents.react.types import LoopResult, ProgressEvent
from agent_arch.llm.base import Message, StreamEvent

REQUEST = "합성 기안서 초안의 미리보기 작업을 취소합니다. 실제로 처리하지 않았다면 완료했다고 답하지 마세요."
CLAIM = ('요청하신 "BASIC-MCP-UI-20261017104056-초안"의 임시저장 및 상신 작업을 취소하였습니다. '
         "해당 초안은 저장되지 않았으며, 관련하여 진행 중인 문서나 알림 발송은 없습니다.")


def _llm(text: str, *, stop: asyncio.Event | None = None, error: bool = False,
         malformed_first: bool = False):
    """실제 외부 모델 대신 분할 토큰과 종료·중단·오류를 내는 대역을 만든다."""
    llm = MagicMock()
    llm.bind_tools = AsyncMock()
    finished = {"value": False}
    calls = {"count": 0}

    async def stream(*args, **kwargs):
        calls["count"] += 1
        if malformed_first and calls["count"] == 1:
            yield StreamEvent(type="done", finish_reason="malformed_function_call")
            return
        boundary = text.find("하였습니다.") + len("하였습니다.") if "하였습니다." in text else 90
        yield StreamEvent(type="text", content=text[:boundary])
        await asyncio.sleep(0.01)
        if stop is not None:
            stop.set()
        yield StreamEvent(type="text", content=text[boundary:])
        await asyncio.sleep(0.01)
        if error:
            raise RuntimeError("합성 모델 스트림 오류")
        finished["value"] = True
        yield StreamEvent(type="done", finish_reason="stop")

    llm.stream = stream
    return llm, finished


async def _collect(request: str, answer: str, *, max_iterations: int = 1,
                   stop: asyncio.Event | None = None, error: bool = False,
                   malformed_first: bool = False):
    """같은 공개 스트림 API에서 화면 토큰과 최종 저장용 결과를 함께 수집한다."""
    llm, finished = _llm(answer, stop=stop, error=error, malformed_first=malformed_first)
    events = []
    async for event in stream_react_graph(
        llm, request, [Message(role="user", content=request)], [], AsyncMock(return_value=[]),
        max_iterations=max_iterations,
    ):
        events.append((event, finished["value"]))
    return events


@pytest.mark.asyncio
@pytest.mark.parametrize("max_iterations", [0, 1])
async def test_cancel_claim_split_tokens_never_reach_ui(max_iterations: int) -> None:
    """정상·강제 답변 경로 모두 교정 완료 후 화면과 최종 결과를 같게 낸다."""
    events = await _collect(REQUEST, CLAIM, max_iterations=max_iterations)
    visible = "".join(event.content for event, _ in events if isinstance(event, ProgressEvent) and event.event_type == "token")
    result = next(event for event, _ in events if isinstance(event, LoopResult))
    assert visible == result.content == _FALLBACK


@pytest.mark.asyncio
async def test_general_korean_turn_still_streams_before_model_done() -> None:
    """초안 취소와 관계없는 정상 한국어 답변을 전체 보류하지 않는다."""
    answer = "일반적인 사용성 확인은 화면을 열고 입력과 결과를 차례로 확인하는 과정입니다. " * 4
    events = await _collect("화면 사용성 확인 방법을 설명해 주세요.", answer)
    assert any(isinstance(event, ProgressEvent) and event.event_type == "token" and not finished for event, finished in events)
    assert next(event for event, _ in events if isinstance(event, LoopResult)).content == answer


@pytest.mark.asyncio
async def test_cancel_request_can_keep_honest_negative_guidance() -> None:
    """보류한 취소 요청도 정직한 미처리 안내를 그대로 한 번 표시한다."""
    answer = "아직 실제 취소를 처리하지 않았어요. 현재 승인 대기 여부를 확인해 주세요."
    events = await _collect(REQUEST, answer)
    visible = "".join(event.content for event, _ in events if isinstance(event, ProgressEvent) and event.event_type == "token")
    assert visible == next(event for event, _ in events if isinstance(event, LoopResult)).content == answer


@pytest.mark.asyncio
async def test_stopping_cancel_request_does_not_flush_completed_claim() -> None:
    """생성 중단 시 붙들었던 완료 문장이 부분답변으로 다시 새지 않는다."""
    stop = asyncio.Event()
    set_cancel_event(stop)
    try:
        events = await _collect(REQUEST, CLAIM, stop=stop)
    finally:
        set_cancel_event(None)
    visible = "".join(event.content for event, _ in events if isinstance(event, ProgressEvent) and event.event_type == "token")
    assert "취소하였습니다" not in visible
    assert next(event for event, _ in events if isinstance(event, LoopResult)).finish_reason == "stopped"


@pytest.mark.asyncio
async def test_stream_error_does_not_leak_claim_or_hold_next_request() -> None:
    """오류로 종료된 턴의 버퍼가 다음 일반 요청에 영향을 주지 않는다."""
    with pytest.raises(RuntimeError, match="합성 모델 스트림 오류"):
        await _collect(REQUEST, CLAIM, error=True)
    answer = "이 요청은 이전 초안과 관계없는 정상적인 읽기 전용 안내입니다. " * 4
    events = await _collect("일반적인 화면 안내를 해 주세요.", answer)
    assert any(isinstance(event, ProgressEvent) and event.event_type == "token" and not finished for event, finished in events)


@pytest.mark.asyncio
async def test_malformed_recovery_error_keeps_corrected_partial_answer() -> None:
    """오류를 흡수하는 강제 재시도도 원문 대신 정리된 종료 본문만 보낸다."""
    events = await _collect(REQUEST, CLAIM, malformed_first=True, error=True, max_iterations=2)
    visible = "".join(event.content for event, _ in events if isinstance(event, ProgressEvent) and event.event_type == "token")
    result = next(event for event, _ in events if isinstance(event, LoopResult))
    assert visible == result.content == _FALLBACK
    assert result.finish_reason == "error"
