"""진행 중인 스트리밍 run 레지스트리.

`/agent/stream` 이 시작될 때 run 을 등록해 `run_id` 와 취소용 `asyncio.Event` 를 발급하고,
`/agent/runs/{run_id}/stop` 이 같은 Event 를 `.set()` 해 협조적 중단을 트리거한다.

배포가 uvicorn 단일 프로세스 + 단일 컨테이너(Dockerfile 에 --workers 없음, compose
replicas 없음)라 모듈 전역 dict 로 충분하다. 기존 `app/queue/runner.py` 의
`TaskRunner._tasks` 와 같은 in-memory 패턴이다.

주의: **메모리 전용** — 프로세스 재시작 시 진행 중 run 은 복구되지 않는다.
멀티 워커/멀티 컨테이너로 확장하면 Redis pub/sub가 필요해진다.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class _RunHandle:
    """등록된 run 한 건의 핸들.

    Attributes:
        user_sub: run 소유자(Keycloak sub). 중단 요청 시 소유권 검증에 쓴다.
        cancel_event: 협조적 취소 신호. 중단 엔드포인트가 set 하면 루프가 폴링해 멈춘다.
    """

    user_sub: str
    cancel_event: asyncio.Event


_RUNS: Dict[str, _RunHandle] = {}


def register_run(user_sub: str) -> Tuple[str, asyncio.Event]:
    """새 run 을 등록하고 run_id 와 취소 Event 를 발급한다.

    Args:
        user_sub: run 소유자(Keycloak sub).

    Returns:
        (run_id, cancel_event) 튜플. run_id 는 클라이언트에 내려보내고,
        cancel_event 는 호출자가 ContextVar 에 심는다.
    """
    run_id = uuid.uuid4().hex
    event = asyncio.Event()
    _RUNS[run_id] = _RunHandle(user_sub=user_sub, cancel_event=event)
    return run_id, event


def request_stop(run_id: str, user_sub: str) -> bool:
    """run 중단을 요청한다. 본인 소유 run 만 중단 가능.

    미존재 run 과 타인 소유 run 을 구분하지 않고 모두 False 를 반환한다
    (호출처에서 동일하게 404 처리 → run 존재 여부 정보 누출 방지).

    Args:
        run_id: 중단할 run 식별자.
        user_sub: 요청자(Keycloak sub).

    Returns:
        Event 를 set 했으면 True, run 이 없거나 소유자가 다르면 False.
    """
    handle = _RUNS.get(run_id)
    if handle is None or handle.user_sub != user_sub:
        return False
    handle.cancel_event.set()
    return True


def unregister_run(run_id: str) -> None:
    """run 등록을 해제한다(스트림 종료 시 finally 에서 호출).

    Args:
        run_id: 해제할 run 식별자. 없으면 무시한다.
    """
    _RUNS.pop(run_id, None)
