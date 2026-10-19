"""스트리밍 run 의 협조적(cooperative) 취소 신호.

요청 단위로 `asyncio.Event` 를 ContextVar 에 심어, ReAct 루프가 시그니처 변경 없이
취소 여부를 폴링할 수 있게 한다. 같은 task 로 구동되는 async 제너레이터 체인
(generate → orchestrator.stream → react_loop_events)에서 동일 ContextVar 값이 보인다.

엔드포인트(`generate()`)가 `set_cancel_event(event)` 로 심고, 중단 엔드포인트는
`run_registry` 가 보관한 **같은 Event 객체**를 `.set()` 한다. 루프는 `is_cancelled()` 로
이 Event 를 폴링한다.

명시적 파라미터 대신 ContextVar 를 쓰는 이유: `react_loop_events` 는 호출처가 여럿이고
포지셔널 인자 테스트가 있어 시그니처 변경 영향 범위가 크기 때문이다. 코드베이스의
`app/agents/context.py` 가 사용하는 ContextVar 패턴과 일관된다.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Optional

_cancel_event: ContextVar[Optional[asyncio.Event]] = ContextVar(
    "cancel_event", default=None
)


def set_cancel_event(event: Optional[asyncio.Event]) -> None:
    """현재 요청 컨텍스트에 취소 Event 를 심는다(또는 None 으로 해제).

    Args:
        event: 이 run 의 취소 신호 Event. 정리 시 None 을 전달한다.
    """
    _cancel_event.set(event)


def get_cancel_event() -> Optional[asyncio.Event]:
    """현재 컨텍스트의 취소 Event 를 반환한다.

    Returns:
        설정된 Event, 없으면 None.
    """
    return _cancel_event.get()


def is_cancelled() -> bool:
    """현재 run 이 중단 요청을 받았는지 여부.

    Event 가 없으면(취소 비대상 경로) 항상 False 다. bool 읽기라 토큰 단위로
    호출해도 비용이 무시할 만하다.

    Returns:
        중단 요청이 set 되었으면 True.
    """
    event = _cancel_event.get()
    return event is not None and event.is_set()
